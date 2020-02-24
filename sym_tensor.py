import numpy as np
from numpy import linalg as LA
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable
# requires the SymIndex class to be imported 
from sym_index import SymIndex
"""
-----------------------------------------------------------------------------
Glen's SymTensor class
-----------------------------------------------------------------------------
Defines an object class for (element-wise encoded) blocksparse tensors, which
store only the structually non-zero tensor elements. The class also contains 
the functions and methods for manipulating these tensors. Designed to share 
common functionality with numpy arrays where-ever possible.

Quick reference:
Definig attributes: (*.data, *.indices, *.arrows, *.partitions, *.divergence)

Important properties: (*.ndim, *.shape, *.size, *.dtype, *.norm)

Important methods: 
  Overloaded operators: ==, !=, +, -, *, /, @)
  A.copy() - create a copy of tensor `A` in memory.
  A.conj() - take complex conjugate of data in `A` and reverse direction of 
    its arrows. 
  A.toarray() - export a SymTensor to a dense numpy array. 
  A.reshape() - reshape to new dimensions (fusing or unfusing indices)
  A.transpose() - transpose index ordering.  
  A.retrieve_blocks() - extract the symmetry blocks from tensor `A` w.r.t a 
    specified bipartition of the indices.
  
Important class methods (for creation of tensors):
  fromarray - create a SymTensor from a dense numpy array. 
  rand - create a SymTensor with randomly set elements.
  zeros - create a SymTensor with all zeros elements.
  ones - create a SymTensor with all unit elements.
  eye - create a SymTensor equivalent to the identity matrix.
    
Important static methods (helper functions for SymTensor class):
  fuse_indices
  fuse_indices_unique
  fuse_indices_reduced
  compute_num_nonzero
  find_balanced_partition
""" 

class SymTensor:
  def __init__(self, 
               data: Union[Type, np.ndarray], 
               indices_orig: List[SymIndex], 
               arrows_orig: np.ndarray, 
               partitions: Optional[np.ndarray] = None, 
               divergence: Optional[SymIndex] = None):
    """
    Args:
      data (np.ndarray): structually non-zero tensor elements (in row-major 
        order).
      orig_indices (List[SymIndex]): list of SymIndex. 
      arrows (np.ndarray): boolean array describing each index as incoming 
        (False) or outgoing (True).
      partitions (np.ndarray, optional): 1d array describing the grouping of 
        indices, e.g [1,1,1] for a three index tensor and [2,1] for a three
        index tensor that has been reshaped into a matrix by combining the 
        first two indices.
      divergence (np.ndarray, optional): total quantum number sector of the 
        tensor (currently only implemented for zero divergence) 
    """
    syms = indices_orig[0].syms
    ndim_orig = len(indices_orig)
    
    # initialize partitions and divergence if necessary
    if partitions is None:
      self.partitions = [np.arange(n, n+1, dtype=np.int16) for n in range(ndim_orig)]
    else:
      self.partitions = [np.asarray(partitions[n], dtype=np.int16) for n in range(len(partitions))]
    
    if divergence is None:
      divergence = SymIndex.identity(1,syms)
    
    self.divergence = divergence
    
    if type(data) == (np.dtype or type):
      num_nonzero = SymTensor.compute_num_nonzero(indices_orig,arrows_orig,divergence)
      self.data = np.zeros(num_nonzero, dtype=data)
    else:
      self.data = data.ravel()
      
    self.indices_orig = [indices_orig[n].copy() for n in range(ndim_orig)]
    self.arrows_orig = np.asarray(arrows_orig, dtype=bool)
  
  
  @property
  def ndim_orig(self) -> int:
    """ original number of tensor dimensions (i.e. before any reshape) """
    return len(np.concatenate(self.partitions))
  
  @property
  def ndim(self) -> int:
    """ number of tensor dimensions (cf. numpy ndarray.ndim) """
    return len(self.partitions)
  
  @property
  def shape_orig(self) -> np.ndarray:
    """ shape of the original dense tensor before reshape or transposition """
    return np.asarray([ind.dim for ind in self.indices_orig],dtype=np.uint32)
  
  @property
  def shape_trans(self) -> np.ndarray:
    """ shape of the original dense tensor after transposition but before reshape """
    return np.asarray(self.shape_orig[np.concatenate(self.partitions)],dtype=np.uint32)
  
  @property
  def shape(self) -> np.ndarray:
    """ shape of dense tensor (cf. numpy ndarray.shape) after reshape and transposition"""
    return np.asarray([np.prod(self.shape_orig[self.partitions[n]]) for n in range(self.ndim)],dtype=np.uint32)
  
  @property
  def size(self) -> int:
    """ total number of elements in the dense array (cf. numpy ndarray.size) """
    return np.prod(self.shape_orig)
  
  @property
  def norm(self) -> int:
    """ norm of a SymTensor (cf LA.norm)"""
    return np.sqrt(np.dot(self.data, self.data))

  @property
  def dtype(self) -> Type[np.number]:
    """ data type of tensor elements (cf. numpy ndarray.dtype) """
    return self.data.dtype
  
  @property
  def syms(self) -> List[str]:
    """ data type of tensor elements (cf. numpy ndarray.dtype) """
    return self.indices_orig[0].syms
  
  @property
  def index_groups(self) -> np.ndarray:
    """ return number of original indices in each ind grouping """
    return np.asarray([len(self.partitions[n]) for n in range(self.ndim)], dtype=np.int16)
  
  @property
  def indices_trans(self) -> List[SymIndex]:
    """ return unreshaped tensor indices after transposition"""
    return [self.indices_orig[n] for n in np.concatenate(self.partitions)] 
  
  @property
  def indices(self) -> List[SymIndex]:
    """ return transposed and reshaped tensor indices"""
    indices_trans = self.indices_trans
    arrows_trans = self.arrows_trans
    
    cumul_grouping = np.insert(np.cumsum(self.index_groups),0,0)
    indices = [0]*self.ndim
    for n in range(self.ndim):
      curr_group = slice(cumul_grouping[n],cumul_grouping[n+1])
      # Note: arrow for combined indices taken from the first index in the group.
      indices[n] = SymTensor.fuse_indices(indices_trans[curr_group], (arrows_trans[curr_group] != arrows_trans[cumul_grouping[n]]))
    return indices
  
  @property
  def arrows_trans(self) -> np.ndarray:
    """ return unreshaped tensor arrows after transposition"""
    return np.asarray([self.arrows_orig[n] for n in np.concatenate(self.partitions)], dtype=bool)
  
  @property
  def arrows(self) -> np.ndarray:
    """ return the tensor arrows after reshape and transposition"""
    arrows_trans = self.arrows_trans
    return  np.asarray([arrows_trans[self.partitions[n][0]] for n in range(self.ndim)], dtype=bool)
  
  
  def __eq__(self,other):
    """ determine if two tensors are equal """
    if (self.ndim != other.ndim):
      return False
    
    for n in range(self.ndim):
      if not (len(self.partitions[n]) == len(other.partitions[n])):
        return False
    
    if not self.divergence == other.divergence:
      return False
    
    if not (np.array_equal(self.arrows_trans,other.arrows_trans)):
      return False
    
    for n in range(self.ndim_orig):
      if (self.indices_trans[n] != other.indices_trans[n]):
        return False
  
    if not (np.array_equal(self.transpose_data().data, other.transpose_data().data)):
      return False
    
    return True
  
  def __ne__(self,other):
     """ determine if two tensors are not equal """
     return not(self==other)
   
  def __repr__(self):
    """ define repl output """
    return (str(type(self)) +", dtype:" +str(self.dtype)+", shape:" +str(self.shape)+ "\n")
    
  def __add__(self, other) -> "SymTensor":
    """ Addition between two SymTensors """
    new_tensor = self.copy()
    new_tensor.data = self.transpose_data().data + other.transpose_data().data
    return new_tensor
  
  def __sub__(self, other) -> "SymTensor":
    """ Subtractrion between two SymTensors """
    new_tensor = self.copy()
    new_tensor.data = self.transpose_data().data - other.transpose_data().data
    return new_tensor
  
  def __mul__(self, other) -> "SymTensor":
    """ Multiplication between a scalar and a SymTensor """
    new_tensor = self.copy()
    new_tensor.data = other*new_tensor.data
    return new_tensor
  
  def __rmul__(self, other) -> "SymTensor":
    """ Multiplication between a scalar and a SymTensor """
    new_tensor = self.copy()
    new_tensor.data = other*new_tensor.data
    return new_tensor
  
  def __truediv__(self, scalar: int) -> "SymTensor":
    """ Division of SymTensor with scalar """
    new_tensor = self.copy()
    new_tensor.data = new_tensor.data / scalar
    return new_tensor
  
  def copy(self) -> "SymTensor":
    """ Create a copy in memory of a SymTensor (including all the underlying data) """
    return SymTensor(self.data.copy(), self.indices_orig, self.arrows_orig, self.partitions, self.divergence)
  
  def conj(self) -> "SymTensor":
    """ Take complex conjugation of the tensor data and reverse the arrows """
    return SymTensor(self.data.conj(), self.indices_orig, np.logical_not(self.arrows_orig), self.partitions, self.divergence.dual())
  
  def normalize(self) -> "SymTensor":
    """ Normalize a SymTensor """
    return SymTensor(self.data/self.norm, self.indices_orig, self.arrows_orig, self.partitions, self.divergence)
  
  def toarray(self, pivot: Optional[int] = None) -> np.ndarray:
    """ Export a SymTensor to a dense np.ndarray """
    dense_array = np.zeros(self.size, dtype=self.data.dtype)
    if (self.size != 0):
      dense_pos = SymTensor.fuse_indices_reduced(self.indices_trans, self.arrows_trans, self.divergence, return_locs = True)[1] 
      dense_array[dense_pos] = self.transpose_data().data
      
    if pivot is not None:
      final_shape = [np.prod(self.shape[:pivot]),np.prod(self.shape[pivot:])]
    else:
      final_shape = self.shape
     
    return dense_array.reshape(final_shape)
  
  def toarray_orig(self, pivot: Optional[int] = None) -> np.ndarray:
    """ Export a SymTensor to a dense np.ndarray (without eval of lazy permutation)"""
    dense_array = np.zeros(self.size, dtype=self.data.dtype)
    if (self.size != 0):
      dense_pos = SymTensor.fuse_indices_reduced(self.indices_orig, self.arrows_orig, self.divergence, return_locs = True)[1] 
      dense_array[dense_pos] = self.data
      
    if pivot is not None:
      final_shape = [np.prod(self.shape_orig[:pivot]),np.prod(self.shape_orig[pivot:])]
    else:
      final_shape = self.shape_orig
     
    return dense_array.reshape(final_shape)
  
  def __matmul__(self, other) -> "SymTensor":
    """ Multiply two SymTensor matrices """
    if (self.ndim > 2) or (other.ndim > 2):
      raise ValueError("SymTensors must be matrices (ndim = 2) or vectors "
                       "(ndim = 1) in order to use matrix multiply")
    
    num_cont = len(self.partitions[1])
    left_num_free = len(self.partitions[0])
    right_num_free = len(other.partitions[1])
      
    # initialize output tensor 
    C_indices = [self.indices_orig[n] for n in self.partitions[0]] + [other.indices_orig[n] for n in other.partitions[1]]
    C_arrows = np.concatenate([self.arrows_orig[self.partitions[0]], other.arrows_orig[other.partitions[1]]])
    C_divergence = self.divergence @ other.divergence
    C_num_nonzero = SymTensor.compute_num_nonzero(C_indices, C_arrows, C_divergence)
    C_partitions = [np.arange(0,left_num_free,dtype=np.int16), np.arange(left_num_free,left_num_free+right_num_free,dtype=np.int16)]
    C_tensor = SymTensor(np.zeros(C_num_nonzero, dtype=self.dtype), C_indices, C_arrows,partitions=C_partitions, divergence=C_divergence)
    
    if ((len(self.data) > 0) and (len(other.data) > 0)) and (len(C_tensor.data) > 0):
      # find locations of symmetry blocks
      A_block_maps, A_block_ind, A_block_dims = self.retrieve_blocks(left_num_free, leading_div=self.divergence)
      B_block_maps, B_block_ind, B_block_dims = other.retrieve_blocks(num_cont, lagging_div=other.divergence)
      C_block_maps, C_block_ind, C_block_dims = C_tensor.retrieve_blocks(left_num_free, leading_div=self.divergence, lagging_div=other.divergence)
     
      # construct map between qnum labels for each tensor and the common qnums
      common_ind, A_to_common, B_to_common = SymIndex.intersect_inds(A_block_ind, B_block_ind)
      C_to_common = SymIndex.intersect_inds(C_block_ind, common_ind)[1]
      
      # perform tensor contraction oone block at a time
      for n in range(common_ind.dim):
        nA = A_to_common[n]
        nB = B_to_common[n] 
        nC = C_to_common[n] 
        
        C_tensor.data[C_block_maps[nC].ravel()] = (self.data[A_block_maps[nA].reshape(A_block_dims[:,nA])] @ 
                                            other.data[B_block_maps[nB].reshape(B_block_dims[:,nB])]).ravel()
    return C_tensor
  
  def reshape(self, *dims_new: Union[tuple,np.ndarray]) -> "SymTensor":
    """
    Reshape a SymTensor object (cf. np.ndarray.reshape). Does not manipulate 
    the tensor data, only changes the `self.partitions` field to reflect the
    new grouping of indices.
    Args: 
      new_dims: either tuple or np.ndarray describing the new tensor shape
    Returns:
      SymTensor: reshaped SymTensor
    """
    dims_new = np.asarray(dims_new, dtype=np.int16).ravel()
    dims_orig = self.shape_trans
    
    # find new grouping of indices that matches `dims_new`
    ind_grouping = []
    for n in range(len(dims_new)):
      group_end = np.flatnonzero(np.cumprod(dims_orig[sum(ind_grouping):]) == dims_new[n])
      if len(group_end) == 0:
        raise ValueError("Reshape of tensor from original shape {} "
                         "into new shape {} is not possible.".format(
                         tuple(self.shape),tuple(dims_new)))
      else:
        # complicated stuff to properly deal with leading and trailing dim-1 indices 
        if n < (len(dims_new)-1):
          if dims_new[n+1] == 1:
            ind_grouping.append(group_end[-2]+1)
          else:
            ind_grouping.append(group_end[-1]+1)
        else:
          ind_grouping.append(group_end[-1]+1)
    
    ind_grouping = np.asarray(ind_grouping,dtype=np.int16)
    if np.array_equal(self.index_groups,ind_grouping): # trivial reshape
      return self
    else:
      cumul_grouping = np.insert(np.cumsum(ind_grouping),0,0)
      flat_partitions = np.concatenate(self.partitions)
      new_partitions = [flat_partitions[cumul_grouping[n]:cumul_grouping[n+1]] for n in range(len(ind_grouping))]
    
      return SymTensor(data=self.data, indices_orig=self.indices_orig, arrows_orig=self.arrows_orig, 
                     partitions=new_partitions, divergence=self.divergence)
    
  def reshape_data(self) -> "SymTensor":
    """
    Evalautes a lazily-reshaped tensor (by explicitly fusing the tensor 
    indices according to the partitions). 
    """
    new_tensor = SymTensor(np.zeros(0, dtype=self.data.dtype), indices_orig=self.indices, 
                           arrows_orig=self.arrows, divergence=self.divergence)
    new_tensor.data = self.data.copy()
    return new_tensor
  
  def transpose(self, *perm_ord: tuple) -> "SymTensor":
    """
    Implements a lazy transpose (only updates the tensor meta-data to reflect 
    that a transposition has taken place, but doesn't transpose the data).
    Args: 
      perm_ord: either tuple or np.ndarray describing new index order
    Returns:
      SymTensor: permuted SymTensor
    """
    perm_ord = np.asarray(perm_ord, dtype=np.int16).ravel()
    new_partitions = [self.partitions[n] for n in perm_ord]
    return SymTensor(self.data, self.indices_orig, self.arrows_orig, new_partitions, self.divergence)
    
  def transpose_data(self) -> "SymTensor":
    """
    Evalautes a lazily-transposed tensor (by explicitly reordering the data 
    in memory). 
    """
    if np.array_equal(np.concatenate(self.partitions),np.arange(self.ndim_orig)):
      # trivial permutation
      return self
    
    else: # non-trivial permutation
      # initialize new tensor
      cumul_groups = np.insert(np.cumsum(self.index_groups),0,0)
      new_partitions = [np.arange(cumul_groups[n], cumul_groups[n+1], dtype=np.int16) for n in range(len(self.index_groups))]
      trans_tensor = SymTensor(np.zeros(len(self.data), dtype=self.data.dtype), indices_orig=self.indices_trans, 
                               arrows_orig=self.arrows_trans, divergence=self.divergence, partitions=new_partitions)
      
      if self.data.size == 0:
        # special case: trivial tensor
        trans_tensor.data = np.asarray([],dtype=self.data.dtype)
        
      else: # general case
        pivot = SymTensor.find_balanced_partition(self.shape_trans)
        
        # find block maps for original and permuted tensors
        block_maps0, block_ind0, block_dims0 = self.retrieve_blocks(pivot, transpose_order=np.concatenate(self.partitions), lagging_div=self.divergence)
        block_maps1, block_ind1, block_dims1 = trans_tensor.retrieve_blocks(pivot, lagging_div=self.divergence)
        trans_tensor.data[np.concatenate(block_maps1)] = self.data[np.concatenate(block_maps0)]
        
      return trans_tensor
  
  def retrieve_blocks(self,
                      pivot: int, 
                      transpose_order: Optional[np.ndarray]=None,
                      leading_div: Optional[SymIndex]=None,
                      lagging_div: Optional[SymIndex]=None
                      ) -> Tuple[List[np.ndarray], SymIndex, np.ndarray]:
    """
    Find the location of all non-trivial symmetry blocks within the data 
    vector of a SymTensor (accounting for a potential transposition) when the 
    tensor is viewed as a matrix across some prescribed index bi-partition.
    Args:
      pivot: location of tensor partition (i.e. such that the tensor is 
        viewed as a matrix between first `pivot` indices and the remaining 
        indices).
      transpose_order: order with which to permute the tensor axes. If 
        omitted defaults to the order given by the tensor partitions.
      leading_div: optional dim-1 SymIndex to be added before the first tensor 
        index, used for tensors with non-trivial divergence.
      lagging_div: optional dim-1 SymIndex to be added after the last tensor 
        index, used for tensors with non-trivial divergence.
    Returns:
      List[np.ndarray]: list of integer arrays containing the locations of 
        symmetry blocks in the data vector.
      SymIndex: containing the quantum numbers of the symmetry blocks.
        block, with 'n' the number of symmetries and 'm' the number of blocks.
      np.ndarray: 2-by-m array describing the dims each block, with 'm' the 
        number of blocks.
    """
    # initialize optional arguments
    syms = self.syms
    if transpose_order is None:
      transpose_order = np.concatenate(self.partitions)
     
    if leading_div is None:
      leading_div = SymIndex.identity(1,syms)
      
    if lagging_div is None:
      lagging_div = SymIndex.identity(1,syms)
      
    # modify indices, arrows, pivot, transpose order to account for div's
    indices = [leading_div] + self.indices_orig + [lagging_div]
    arrows = np.concatenate((np.asarray([True],dtype=bool),self.arrows_orig,np.asarray([True],dtype=bool)))
    pivot = pivot+1
    transpose_order = np.concatenate((np.asarray([0]),np.asarray(transpose_order)+1,np.asarray([len(transpose_order)+1])))
    
    # compute tensor dimensions and default strides
    num_inds = len(indices)
    tensor_dims = np.array([indices[n].dim for n in range(num_inds)],dtype=int)
    strides = np.append(np.flip(np.cumprod(np.flip(tensor_dims[1:]))),1)
    
    # define properties of new tensor resulting from transposition
    new_strides = strides[transpose_order]
    new_row_indices = [indices[n] for n in transpose_order[:pivot]]
    new_col_indices = [indices[n] for n in transpose_order[pivot:]]
    new_row_arrows = [arrows[n] for n in transpose_order[:pivot]]
    new_col_arrows = [arrows[n] for n in transpose_order[pivot:]]
    
    # compute qnums of row/cols in transposed tensor
    unique_row_ind, new_row_degen = SymTensor.fuse_indices_unique(new_row_indices, new_row_arrows, return_degens=True)
    unique_col_ind, new_col_degen = SymTensor.fuse_indices_unique(new_col_indices, np.logical_not(new_col_arrows), return_degens=True)
    block_ind, new_row_map, new_col_map = SymIndex.intersect_inds(unique_row_ind, unique_col_ind)
    block_dims = np.asarray([new_row_degen[new_row_map],new_col_degen[new_col_map]], dtype=np.uint32)
    num_blocks = block_ind.dim
  
    if np.array_equal(np.asarray(transpose_order, dtype=np.int16), np.arange(len(indices), dtype=np.int16)):
      if num_blocks == 0: # special case of tensor with no structually non-zero elements
        return np.zeros(0,dtype=np.uint32), block_ind, np.asarray([], dtype=np.uint32)
      
      elif num_blocks == 1: # special case of tensor with only a single block
        return [np.arange(np.prod(block_dims), dtype=np.uint32).ravel()], block_ind, block_dims
      
      else: # general case of tensor with multiple blocks
        # calculate number of non-zero elements in each row of the matrix
        row_ind = SymTensor.fuse_indices_reduced(indices[:pivot], arrows[:pivot], block_ind)
        row_num_nz = new_col_degen[new_col_map[row_ind.ind_labels]]
        cumulate_num_nz = np.insert(np.cumsum(row_num_nz[0:-1]),0,0).astype(np.uint32)
        
        # calculate mappings for the position in data-vector of each block 
        if num_blocks < 15:
          # faster method for small number of blocks
          row_locs = np.concatenate([(row_ind.ind_labels==n) for n in range(num_blocks)]).reshape(num_blocks,row_ind.dim)
        else:
          # faster method for large number of blocks
          row_locs = np.zeros([num_blocks,row_ind.dim],dtype=bool)
          row_locs[row_ind.ind_labels,np.arange(row_ind.dim)] = np.ones(row_ind.dim,dtype=bool)
        
        block_dims = np.array([[new_row_degen[new_row_map[n]],new_col_degen[new_col_map[n]]] for n in range(num_blocks)],dtype=np.uint32).T
        block_maps = [(cumulate_num_nz[row_locs[n,:]][:,None] + np.arange(block_dims[1,n], dtype=np.uint32)[None,:]).ravel() for n in range(num_blocks)]
      
        return block_maps, block_ind, block_dims
    
    else: # general case: non-trivial transpose order
      # compute qnums of row/cols in original tensor
      orig_pivot = SymTensor.find_balanced_partition(tensor_dims)
      orig_width = np.prod(tensor_dims[orig_pivot:])
      orig_unique_row_ind = SymTensor.fuse_indices_unique(indices[:orig_pivot], arrows[:orig_pivot])
      orig_unique_col_ind, orig_col_degen = SymTensor.fuse_indices_unique(indices[orig_pivot:], np.logical_not(arrows[orig_pivot:]), return_degens=True)
      orig_block_ind, row_map, col_map = SymIndex.intersect_inds(orig_unique_row_ind, orig_unique_col_ind)
      orig_num_blocks = orig_block_ind.dim
      
      if orig_num_blocks == 0: # special case: trivial number of non-zero elements
        return [], SymIndex.identity(0,syms), np.array([], dtype=np.uint32)
      
      orig_row_ind = SymTensor.fuse_indices(indices[:orig_pivot], arrows[:orig_pivot])
      orig_col_ind = SymTensor.fuse_indices(indices[orig_pivot:], np.logical_not(arrows[orig_pivot:]))
      inv_row_map = -np.ones(orig_unique_row_ind.dim, dtype=np.int16)
      for n in range(len(row_map)):
        inv_row_map[row_map[n]] = n
      
      # compute row degens (i.e. number of non-zero elements per row)
      all_degens = np.append(orig_col_degen[col_map],0)[inv_row_map[orig_row_ind.ind_labels]]
      all_cumul_degens = np.cumsum(np.insert(all_degens[:-1],0,0)).astype(np.uint32)
      
      # generate vector which translates from dense row position to sparse row position
      dense_to_sparse = np.zeros(orig_width,dtype=np.uint32)
      for n in range(orig_num_blocks):
        dense_to_sparse[orig_col_ind.ind_labels == col_map[n]] = np.arange(orig_col_degen[col_map[n]],dtype=np.uint32)
        
      row_ind, row_locs = SymTensor.fuse_indices_reduced(new_row_indices, new_row_arrows, block_ind, return_locs=True, strides=new_strides[:pivot])
      col_ind, col_locs = SymTensor.fuse_indices_reduced(new_col_indices, np.logical_not(new_col_arrows), block_ind, return_locs=True, strides=new_strides[pivot:])
      
      # find location of blocks in transposed tensor (w.r.t positions in original)
      block_maps = [0]*num_blocks
      for n in range(num_blocks):
        orig_row_posL, orig_col_posL = np.divmod(row_locs[row_ind.ind_labels == n], orig_width)
        orig_row_posR, orig_col_posR = np.divmod(col_locs[col_ind.ind_labels == n], orig_width)
        block_maps[n] = (all_cumul_degens[np.add.outer(orig_row_posL,orig_row_posR)] +
                         dense_to_sparse[np.add.outer(orig_col_posL,orig_col_posR)]).ravel()
       
      return block_maps, block_ind, block_dims
  
  
  @classmethod
  def rand(cls, indices: List[SymIndex], arrows: np.ndarray, divergence: Optional[Union[List, np.ndarray]] = None):
    """ Construct a SymTensor with uniformly distributed random elements 
    (cf. numpy.random.rand). """
    if divergence is None: 
      divergence = SymIndex.identity(1,indices[0].syms)
    elif type(divergence) is not SymIndex:
      divergence = SymIndex.create(divergence,indices[0].syms)
    
    num_nz = SymTensor.compute_num_nonzero(indices, arrows, divergence)
    return cls(data=np.random.rand(num_nz), indices_orig=indices, arrows_orig=arrows, divergence=divergence)
  
  @classmethod
  def zeros(cls, indices: List[SymIndex], arrows: np.ndarray, divergence: Optional[Union[List, np.ndarray]] = None):
    """ Construct a SymTensor with all elements initialized to zero (cf. 
    numpy.zeros). """
    if divergence is None: 
      divergence = SymIndex.identity(1,indices[0].syms)
    elif type(divergence) is not SymIndex:
      divergence = SymIndex.create(divergence,indices[0].syms)
    
    num_nz = SymTensor.compute_num_nonzero(indices, arrows, divergence)
    return cls(data=np.zeros(num_nz,dtype=float), indices_orig=indices, arrows_orig=arrows, divergence=divergence)
  
  @classmethod
  def ones(cls, indices: List[SymIndex], arrows: np.ndarray, divergence: Optional[Union[List, np.ndarray]] = None):
    """ Construct a SymTensor with all (structually non-zero) elements 
    initialized to unity (cf. numpy.ones). """
    if divergence is None: 
      divergence = SymIndex.identity(1,indices[0].syms)
    elif type(divergence) is not SymIndex:
      divergence = SymIndex.create(divergence,indices[0].syms)
    
    num_nz = SymTensor.compute_num_nonzero(indices, arrows, divergence)
    return cls(data=np.ones(num_nz,dtype=float), indices_orig=indices, arrows_orig=arrows, divergence=divergence)
  
  @classmethod
  def eye(cls, indices: List[SymIndex], arrows: np.ndarray, pivot: Optional[int]=None):
    """ Construct a SymTensor where each block is an the identity matrix 
    (under bipartition specified by `pivot` or between first N/2 and final N/2 
    indices if `pivot` is omitted). """
    if pivot is None:
      pivot = (len(indices)//2)
    
    # create SymTensor
    num_nz = SymTensor.compute_num_nonzero(indices, arrows)
    eye_tensor = cls(data=np.zeros(num_nz,dtype=float), indices_orig=indices, arrows_orig=arrows)
  
    # find block locations
    block_maps, block_ind, block_dims = eye_tensor.retrieve_blocks(pivot=pivot)
    
    # set each block to identity
    for n in range(block_ind.dim):
      eye_tensor.data[block_maps[n]] = np.eye(block_dims[0,n],block_dims[1,n],dtype=float).ravel()
    
    return eye_tensor
  
  @classmethod
  def fromarray(cls, arr: np.ndarray, indices: List[SymIndex], arrows: np.ndarray, 
                divergence: Optional[Union[List, np.ndarray, SymIndex]] = None):
    """ Construct a SymTensor from a dense np.ndarray `arr`. Dimensions of 
    `indices` must match `arr.shape`. """
    if divergence is None: 
      divergence = SymIndex.identity(1,indices[0].syms)
    elif type(divergence) is not SymIndex:
      divergence = SymIndex.create(divergence,indices[0].syms)
    
    ele_locs = SymTensor.fuse_indices_reduced(indices, arrows, divergence, return_locs = True)[1]
    data = np.asarray((arr.ravel())[ele_locs],dtype=arr.dtype)
    
    return cls(data=data, indices_orig=indices, arrows_orig=arrows, divergence=divergence)
  
  @staticmethod
  def find_balanced_partition(index_dims: np.ndarray) -> int:
    """
    Find the location of the tensor index bipartition that most closely balances
    the total dimensions of each partition.
    Args:
      index_dims: list of dim of each index.
    Returns:
      int: location of the index partition (i.e. number of indices contained in 
        first side of partition).
    """
    index_dims = np.asarray(index_dims, dtype=np.int64)
    num_ind = len(index_dims)
    
    # trivial cases
    if num_ind == 0:
      return 0
    elif num_ind == 1:
      return 1
    
    imbalance = [np.abs(np.prod(index_dims[:n]) - np.prod(index_dims[n:])) for n in range(num_ind)]
    pivot = np.argmin(np.array(imbalance))
    
    if pivot == 0:
      return 1 #always include first index in partition
    else:
      return pivot
  
  @staticmethod
  def fuse_indices(indices: List[SymIndex], arrows: np.ndarray) -> SymIndex:
    """
    Fuse multiple SymIndex into a single SymIndex. 
    Args:
      indices: list of SymIndex to combine.
      arrows: 1d array of bools describing index orientations. 
    Returns:
      SymIndex: the fused index.
    """
    comb_ind = indices[0].dual(arrows[0])
    for n in range(1,len(indices)):
      comb_ind = comb_ind @ indices[n].dual(arrows[n])
    
    return comb_ind

  @staticmethod
  def fuse_indices_unique(indices: List[SymIndex], 
                           arrows: np.ndarray, 
                           return_degens: Optional[bool] = False
                           ) -> Tuple[SymIndex, np.ndarray]:
    """
    Fuse multiple SymIndex into a single SymIndex. 
    Args:
      indices: list of SymIndex to combine.
      arrows: 1d array of bools describing index orientations. If omitted 
        defaults to all incoming (or False) indices.
      only_unique: if True then return an index containing only the unique 
        qnums from the fused index.
      return_degens: if True the function will return a 1d array specifying the
        degeneracy of each of the unique qnums in the fused index.
    Returns:
      SymIndex: the fused index.
    """
    comb_ind = indices[0].dual(arrows[0])
    if return_degens:
      unique_comb_degen = comb_ind.degens  
      for n in range(1,len(indices)):
        
        comb_ind = comb_ind.remove_degen() @ indices[n].dual(arrows[n]).remove_degen()
      
        # compute new degeneracies from kronecker product of old degens
        comb_degen = np.kron(unique_comb_degen, indices[n].degens)
        unique_comb_degen = np.array([np.sum(comb_degen[comb_ind.ind_labels == n]) for n in range(comb_ind.num_unique)])
      return comb_ind.remove_degen(), unique_comb_degen
      
    else:
      for n in range(1,len(indices)):
        comb_ind = comb_ind.remove_degen() @ indices[n].dual(arrows[n]).remove_degen()
        
      # print(comb_ind.unique_qnums)
      return comb_ind.remove_degen()
    
  @staticmethod
  def fuse_indices_reduced(indices: List[SymIndex], 
                          arrows: np.ndarray,
                          ind_kept: SymIndex, 
                          return_locs: Optional[bool] = False, 
                          strides: Optional[np.ndarray] = None,
                          ) -> Tuple[SymIndex, np.ndarray]:
    """
    Fuse two or more SymIndex into a single SymIndex, but truncate the values of 
    the fused index to retain only the index values whose qnums are common to 
    one appearing in `reduce_ind`. This function is equilvalent to using 
    `fuse_indices` followed by `.reduce(ind_kept)` on the output, but is more 
    efficient.
    Args:
      indices: list of SymIndex to combine.
      arrows: vector of bools describing index orientations.
      ind_kept: SymIndex describing the qnums to be kept in the fused index.
      return_locs: if True, also return the index locations of kept values.
      strides: index strides with which to compute the `return_locs` of the kept 
        elements. Defaults to standard row-major order strides if ommitted (i.e. 
        given a four index tensor of dims [d0,d1,d2,d3] the default strides are 
        [d1*d2*d3,d2*d3,d3,1]).
    Returns:
      SymIndex: the fused index after reduction.
      np.ndarray: locations of the fused SymIndex qnums that were kept. 
    """
    num_inds = len(indices)
    tensor_dims = [indices[n].dim for n in range(num_inds)]
    
    if strides is None:
      trivial_strides = True
    else:
      trivial_strides = False
    
    if num_inds == 1: # reduce single index
      if strides is None:
        strides = np.array([1], dtype=np.uint32)
      return indices[0].dual(arrows[0]).reduce(ind_kept, return_locs, strides)
    
    else: 
      # find size-balanced partition of indices 
      pivot = SymTensor.find_balanced_partition(tensor_dims)
     
      # compute quantum numbers for each partition
      ind_L = SymTensor.fuse_indices(indices[:pivot], arrows[:pivot])
      ind_R = SymTensor.fuse_indices(indices[pivot:], arrows[pivot:])
      
      # fuse left/right indices and intersect with kept index to find common qnums
      ind_fused = ind_L.remove_degen() @ ind_R.remove_degen()
      ind_C, fused_to_common = SymIndex.intersect_inds(ind_fused,ind_kept)[:2]
      
      # invert map from common to fused qnums (more convenient this way)
      common_to_fused = -np.ones(ind_fused.num_unique, dtype=np.int16)
      for n in range(len(fused_to_common)):
        common_to_fused[fused_to_common[n]] = n
      
      # build mapping from left/right ind labels to kept ind labels
      map_to_kept = common_to_fused[ind_fused.ind_labels].reshape([ind_L.num_unique,ind_R.num_unique])
      
      if not trivial_strides:
        # combine index strides (necessary to compute position in final index)
        pos_L = np.arange(0,strides[0]*tensor_dims[0],strides[0],dtype=np.uint32)
        for n in range(1,pivot):
          pos_L = np.add.outer(pos_L,np.arange(0,strides[n]*tensor_dims[n],strides[n],dtype=np.uint32)).ravel()
        
        pos_R = np.arange(0,strides[pivot]*tensor_dims[pivot],strides[pivot],dtype=np.uint32)
        for n in range(pivot+1,len(tensor_dims)):
          pos_R = np.add.outer(pos_R,np.arange(0,strides[n]*tensor_dims[n],strides[n],dtype=np.uint32)).ravel()
      
      # for each unique value of ind_L, find values of ind_R that fuse into a kept qnum
      reduced_labels = [0]*ind_L.num_unique
      row_locs = [0]*ind_L.num_unique
      for n in range(ind_L.num_unique):
        kept_labels = map_to_kept[n,ind_R.ind_labels]
        kept_positions = (kept_labels >= 0)
        reduced_labels[n] = kept_labels[kept_positions]
        
        if return_locs:
          if trivial_strides:
            row_locs[n] = np.flatnonzero(kept_positions)
          else:
            row_locs[n] = pos_R[kept_positions]
      
      reduced_labels = np.concatenate([reduced_labels[n] for n in ind_L.ind_labels])
      if return_locs:
        # compute the index values that are kept after the reduction
        if trivial_strides:
          reduced_locs = np.concatenate([n*ind_R.dim + row_locs[ind_L.ind_labels[n]] for n in range(ind_L.dim)])
        else:
          reduced_locs = np.concatenate([pos_L[n] + row_locs[ind_L.ind_labels[n]] for n in range(ind_L.dim)])
        
        return SymIndex(ind_C.unique_qnums, reduced_labels, indices[0].syms), reduced_locs
      else:
        return SymIndex(ind_C.unique_qnums, reduced_labels, indices[0].syms)

  @staticmethod
  def compute_num_nonzero(indices: List[SymIndex], 
                          arrows: Optional[np.ndarray] = None, 
                          divergence: Optional[SymIndex] = None) -> int:
    """
    Compute the number of non-zero elements in a SymTensor
    Args:
      indices: list of SymIndex.
      arrows: vector of bools describing index orientations.
      divergence: optional SymIndex to specify the divergence if non-trivial.
    Returns:
      int: The number of non-zero elements.
    """
    if arrows is None:
      arrows = np.asarray([False]*len(indices), dtype=bool)
      
    if divergence is None:
      divergence = SymIndex.identity(dim=1, syms=indices[0].syms)
    
    # find unique qnums of fused indices and their degens
    comb_ind, comb_degen = SymTensor.fuse_indices_unique(indices, arrows, return_degens=True) 
    
    # evaluate the divergence in the appropriate qnum sector
    num_nonzero = comb_degen[comb_ind.reduce(divergence, return_locs=True)[1]]
    if (num_nonzero.size > 0):
      return num_nonzero.item()
    else:
      return 0




