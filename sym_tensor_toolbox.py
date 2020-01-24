
import time
import numpy as np
from numpy import linalg as LA
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable

"""
-----------------------------------------------------------------------------
Glen's symmetric tensor toolbox: a toolbox for the efficient representation 
and manipulation of tensors that are invariant under the action of an abelian 
group.
-----------------------------------------------------------------------------

Core design features:
* designed to be fast even for tensors with many indices and/or many different
  qnum sectors per index.
* designed such that SymTensor objects behave in a similar manner to numpy
  ndarrays (under transpose, rehape, matmul, tensordot etc), such that tensor
  network codes require minimal modifications when switching between the two.
  
Techinical Notes:
* uses element-wise encoding strategy (with row-major order) to store only the
  structually non-zero tensor elements in a dense array (or datavector).
* built in support for common symmetries ('U1', 'Z2', 'Z3'), although the 
  toolbox is designed to allow the user to add any custom symmetry (or 
  custom representation). To add a custom symmetry the user must supply the 
  fusion rule in 'fuse_qnums' and the duality transform in 'SymIndex.dual'.
* supports tensors which are simultaneously symmetric under multiple groups 
  (e.g. tensors symmetric under both 'U1' and 'Z2' with distinct index qnums 
  for each symmetry group).
* uses np.int16 to represent both quantum numbers and index labels (which 
  assumes that no index will have more than 32767 distinct symmetry sectors).
* uses np.uint32 to represent the positions that non-zero tensor elements 
  would occupy within the dense array corresponding to a SymTensor (which 
  assumes the size of the dense array would not exceed 2**32 elements)
""" 

###########################################################################
###########################################################################
###########################################################################
class SymIndex:
  """
  Object for storing symmetry information about an index 
  Attributes:
    (.syms, .unique_qnums, .ind_labels)
  Properties:
    (.dim, .num_syms, .num_unique, .qnums, .degens)
  Class methods:
    (.create, .rand, .identity)
  Methods:
    (.dual, .copy, .__matmul__, .reduce)
  """
  
  def __init__(self, 
               unique_qnums: np.ndarray, 
               ind_labels: np.ndarray, 
               syms: List[str]):
    """
    Args:
      unique_qnums: np.ndarray of shape (m,n) with `m` the number of symmetries 
        in use and `n` the number of distinct symmetry sectors in the index.
      ind_labels: 1d array labelling the qnum of each index value in terms of 
        the unique_qnums.
      syms: list of strings describing the symmetries in use (currently 
        supported: 'U1', 'Z2', 'Z3').
    """
    # do some consistancy checks
    if type(syms) != list:
      raise ValueError('syms must be a list of symmetries')
      
    if len(syms) != unique_qnums.shape[0]:
      raise ValueError(('height of qnums array (%i) should match number of '
                        'symmetries in use (%i)')%(unique_qnums.shape[0],len(syms)))
    
    if unique_qnums.dtype != np.int16:
      raise ValueError('qnums must be of type np.int16')
      
    if ind_labels.dtype != np.int16:
      raise ValueError('index labels must be of type np.int16')
    
    self.unique_qnums = unique_qnums
    self.ind_labels = ind_labels
    self.syms = syms
    
  #########################################
  @property
  def dim(self) -> int:
    """ index dimension """
    return len(self.ind_labels)
  
  @property
  def num_syms(self) -> int:
    """ number of symmetries in use """
    return len(self.syms)
  
  @property
  def num_unique(self) -> int:
    """ number of unique quantum numbers (or symmetry sectors) """
    return self.unique_qnums.shape[1]
  
  @property
  def qnums(self) -> np.ndarray:
    """ np.ndarray of shape (self.num_syms,self.dim) explicitly describing 
    qnums for each index value """
    return self.unique_qnums[:,self.ind_labels]
  
  @property
  def degens(self) -> np.ndarray:
    """ degeneracy of each of the unique quantum numbers """
    return np.asarray([np.sum(self.ind_labels == n) for n in range(self.num_unique)], dtype=np.uint32)
  
  #########################################
  @classmethod
  def create(cls, 
             qnums: Union[List[List], np.ndarray, List[int]], 
             syms: Union[List[str], str]):
    """
    Create a SymIndex from an array of quantum numbers.
    Args:
      qnums: quantum numbers for each value of the index. Either a list of 
        `m` lists of length `D` or an np.ndarray of shape (m,D), where `m` is 
        the number of symmetries and `D` is the index dimension. Can also be 
        a 1d np.ndarray or list of ints when using only a single symmetry.
      syms (List[str]): list of strings describing the symmetries in use. Can 
        also be a single string when using only a single symmetry.
    """
    # put syms into a list if necessary
    if type(syms) == list:
      num_syms = len(syms)
    elif type(syms) == str:
      num_syms = 1
      syms = [syms]
    
    # reshape qnums if necessary
    qnums = np.array(qnums,dtype=np.int16)
    if qnums.ndim == 1:
      qnums = qnums.reshape([1,qnums.size])
    
    # check consistancy
    if qnums.shape[0] != num_syms:
      raise ValueError(('height of qnums array (%i) should match number '
                        'of symmetries in use (%i)')%(qnums.shape[0],num_syms))
    
    # find unique values
    [unique_qnums, ind_labels] = np.unique(qnums, return_inverse=True, axis=1)
    
    return cls(np.asarray(unique_qnums, dtype=np.int16), np.asarray(ind_labels, dtype=np.int16) , syms)
  
  @classmethod
  def rand(cls, dim: int, syms: List[str]):
    """
    Create a SymIndex of dimension `dim` with random qnums according to 
    symmetries in `syms`.
    """
    # put syms into a list if necessary
    if type(syms) == list:
      num_syms = len(syms)
    elif type(syms) == str:
      num_syms = 1
      syms = [syms]
    
    # generate random qnums for each symmetry
    qnums = [0]*num_syms
    for n in range(num_syms):
      if syms[n] == 'U1':
        qnums[n] = (2*np.random.randn(dim)).astype(np.int16)
      elif syms[n] == 'Z2':
        qnums[n] = np.round(np.random.rand(dim)).astype(np.int16)
      elif syms[n] == 'Z3':
        qnums[n] = np.round(2*np.random.rand(dim)-1).astype(np.int16)
      elif syms[n] == 'custom_sym':
        # define how random qnums are generated here
        raise NotImplementedError("Please write your own rule for generating random qnums here")
      else:
        raise NotImplementedError("Unable to generate random qnums for unknown symmetry type")
      
    # find unique values
    qnums = np.concatenate(qnums,axis = 0).reshape(num_syms,dim)
    [unique_qnums, ind_labels] = np.unique(qnums, return_inverse=True, axis=1)
      
    return cls(np.asarray(unique_qnums, dtype=np.int16), np.asarray(ind_labels, dtype=np.int16), syms)
  
  @classmethod
  def identity(cls, dim: int, syms: List[str]):
    """
    Create a SymIndex of dimension `dim` with trivial qnums (i.e. equal to the 
    group identity element of `syms`).
    """
    # put syms into a list if necessary
    if type(syms) == list:
      num_syms = len(syms)
    elif type(syms) == str:
      num_syms = 1
      syms = [syms]
    
    unique_qnums = np.asarray(identity_charges(syms).reshape([num_syms,1]), dtype=np.int16)
    ind_labels = np.zeros(dim,dtype=np.int16)
    
    return cls(unique_qnums, ind_labels, syms)
  
  #########################################
  def dual(self, take_dual: bool=True) -> "SymIndex":
    """
    Return the dual index of `self` if 'take_dual=True'.
    """
    if take_dual:
      new_unique_qnums = [0]*self.num_syms
      for n in range(self.num_syms):
        if self.syms[n] == 'U1':
          new_unique_qnums[n] = -self.unique_qnums[n,:]
        elif self.syms[n] == 'Z2':
          new_unique_qnums[n] = self.unique_qnums[n,:]
        elif self.syms[n] == 'Z3':
          new_unique_qnums[n] = -self.unique_qnums[n,:]
        elif self.syms[n] == 'custom_sym':
          # write your duality transformation for custom fusion rule here
          raise NotImplementedError("Please write your own fusion duality rule here")
        else:
          raise NotImplementedError("Unknown symmetry type. Please write your own fusion duality rule here")
      
      return SymIndex(np.concatenate(new_unique_qnums,axis=0).reshape(self.num_syms,self.num_unique), self.ind_labels, self.syms)
    else:
      return self
  
  #########################################
  def copy(self) -> "SymIndex":
    """
    Create a copy of a SymIndex 
    """
    return SymIndex(self.unique_qnums.copy(), self.ind_labels.copy(), self.syms.copy())
  
  #########################################
  def __matmul__(self, other: "SymIndex") -> "SymIndex":
    """
    Define matrix multiplication for two SymIndex as their kronecker product, 
    i.e. such that `(ind1 @ ind2) = combine_indices([ind1,ind2])`.
    """
    # check that combination is valid
    if (self.syms != other.syms):
      raise ValueError("Symmetries of indices to be combined do not match")
    
    # fuse the unique charges from each index, then compute new unique charges
    comb_qnums = fuse_qnums(self.unique_qnums, other.unique_qnums, self.syms)
    [unique_qnums, new_labels] = np.unique(comb_qnums, return_inverse=True, axis=1)
    new_labels = np.asarray(new_labels.reshape(self.num_unique,other.num_unique), dtype=np.int16)
    
    # find new labels using broadcasting (could use np.tile but less efficient)
    ind_labels = new_labels[(self.ind_labels[:,None] + np.zeros([1,other.dim],dtype=np.int16)).ravel(),
                            (other.ind_labels[None,:] + np.zeros([self.dim,1],dtype=np.int16)).ravel()]
  
    return SymIndex(unique_qnums, ind_labels, self.syms)
  
  #########################################
  def reduce(self, 
             kept_qnums: np.ndarray, 
             return_locs: bool = False, 
             strides: int = 1) -> Tuple["SymIndex", np.ndarray]:
    """
    Reduce the dim of a SymIndex to keep only the index values that intersect 
    an ele of `kept_qnums`.
    Args:
      kept_qnums: array of shape (m,n) with `m` number of symmetries in use 
        and each column describing a set of unique quantum numbers to keep.
      return_locs: if True, also return the index locations of kept values.
      strides: index strides with which to compute `return_locs`.
    Returns:
      SymIndex: the SymIndex of reduced dimension.
      np.ndarray: locations of kept values in the output index.
    """
    # find intersection of index qnums and kept qnums
    reduced_qnums, label_to_unique, label_to_kept = intersect2d(self.unique_qnums, kept_qnums, axis=1, return_indices = True) 
    num_unique = len(label_to_unique)
    
    # construct the map to the reduced qnums 
    map_to_reduced = -np.ones(self.dim, dtype=np.int16)
    for n in range(num_unique):
      map_to_reduced[label_to_unique[n]] = n
    
    # construct the map to the reduced qnums 
    reduced_ind_labels = map_to_reduced[self.ind_labels]
    reduced_locs = (reduced_ind_labels >= 0)
    new_ind_labels = np.asarray(reduced_ind_labels[reduced_locs], dtype=np.int16)
    
    if return_locs:
      return (SymIndex(reduced_qnums, new_ind_labels, self.syms), 
              np.asarray(strides*np.flatnonzero(reduced_locs), dtype=np.uint32))
    else:
      return SymIndex(reduced_qnums, new_ind_labels, self.syms)



###########################################################################
###########################################################################
###########################################################################
class SymTensor:
  """
  Object for efficiently storing data and metadata of symmetric tensors 
  Attributes:
    (.data, .indices, .arrows, .partitions, .divergence)
  Properties:
    (.ndim, .shape, .dtype)
  Class methods:
    (.rand, .zeros, .ones, .from_array)
  Methods:
    (.reshape, .__matmul__, .reduce, .deepcopy, .copy, .__mul__, .__mul__, 
    .toarray, .transpose)
  """
  
  def __init__(self, 
               data: np.ndarray, 
               indices: List[SymIndex], 
               arrows: np.ndarray, 
               partitions: np.ndarray = np.zeros(0), 
               divergence: np.ndarray = np.zeros(0)):
    """
    Args:
      data (np.ndarray): structually non-zero tensor elements (in row-major 
        order).
      indices (List[SymIndex]): list of SymIndex. 
      arrows (np.ndarray): boolean array describing each index as incoming 
        (False) or outgoing (True).
      partitions (np.ndarray, optional): 1d array describing the grouping of 
        indices, e.g [1,1,1] for a three index tensor and [2,1] for a three
        index tensor that has been reshaped into a matrix by combining the 
        first two indices.
      divergence (np.ndarray, optional): total quantum number sector of the 
        tensor (currently only implemented for zero divergence)
    """
    syms = indices[0].syms
    num_ind = len(indices)
    
    # initialize partitions and divergence if necessary
    if (len(partitions) == 0):
      partitions = np.ones(num_ind,dtype=np.int16)
    
    if (len(divergence) == 0):
      divergence = np.array([identity_charges(sym_type) for sym_type in syms], dtype=np.int16)
      
    self.data = data.ravel()
    self.indices = [indices[n].copy() for n in range(num_ind)]
    self.arrows = np.array(arrows.copy(), dtype=bool)
    self.partitions = np.array(partitions.copy(),dtype=np.int16)
    self.divergence = divergence.copy()
    
  #########################################
  @property
  def ndim(self) -> int:
    """ number of tensor dimensions (cf. numpy ndarray.ndim) """
    return len(self.partitions)
  
  @property
  def ndim_orig(self) -> int:
    """ original number of tensor dimensions (i.e. before any reshape) """
    return np.sum(self.partitions)
  
  @property
  def size(self) -> int:
    """ total number of elements in the dense array (cf. numpy ndarray.size) """
    return np.prod(self.shape_orig)
  
  @property
  def shape(self) -> Tuple:
    """ shape of dense tensor (cf. numpy ndarray.shape) """
    all_dims = np.array([ind.dim for ind in self.indices],dtype=np.uint32)
    cumul_parts = np.cumsum(np.concatenate([np.array([0]),self.partitions]))
    return tuple([np.prod(all_dims[cumul_parts[n]:cumul_parts[n+1]]) 
                  for n in range(len(self.partitions))])
  
  @property
  def shape_orig(self) -> Tuple:
    """ shape of the original dense tensor (before any reshapes) """
    return np.array([ind.dim for ind in self.indices],dtype=np.uint32)

  @property
  def dtype(self) -> Type[np.number]:
    """ data type of tensor elements (cf. numpy ndarray.dtype) """
    return self.data.dtype
  
  @property
  def joint_indices(self) -> List[SymIndex]:
    """ compute combined tensor indices as specified by self.partitions"""
    cumul_parts = np.cumsum(np.concatenate([np.array([0]),self.partitions]))
    new_arrows = np.array([self.arrows[cumul_parts[n]] for n in range(len(self.partitions))],dtype=bool)  
    new_indices = [0]*len(self.partitions)
    for n in range(len(self.partitions)):
      index_group = slice(cumul_parts[n],cumul_parts[n+1])
      if new_arrows[n]:
        new_indices[n] = combine_indices(self.indices[index_group], np.logical_not(self.arrows[index_group])) 
      else:
        new_indices[n] = combine_indices(self.indices[index_group], self.arrows[index_group]) 
    return new_indices
  
  @property
  def joint_arrows(self) -> np.ndarray:
    """ compute combined tensor arrows as specified by self.partitions"""
    cumul_parts = np.cumsum(np.concatenate([np.array([0]),self.partitions]))
    # Note: the arrow for each combined index is taken from the first index in the combination.
    return np.array([self.arrows[cumul_parts[n]] for n in range(len(self.partitions))], dtype=bool)
  
  #########################################
  @classmethod
  def rand(cls, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor with uniformly distributed random elements (cf. 
    numpy.random.rand). 
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.random.rand(num_nz), indices=indices, arrows=arrows)
  
  @classmethod
  def zeros(cls, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor with all elements initialized to zero (cf. 
    numpy.zeros).
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.zeros(num_nz,dtype=float), indices=indices, arrows=arrows)
  
  @classmethod
  def ones(cls, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor with all (structually non-zero) elements initialized 
    to unity (cf. numpy.ones).
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.ones(num_nz,dtype=float), indices=indices, arrows=arrows)
  
  @classmethod
  def eye(cls, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor where each block is an the identity matrix (under 
    reshape between first N/2 and final N/2 indices). Requires and even number 
    N of indices. 
    """
    # initialize datavector
    num_nz = compute_num_nonzero(indices, arrows)
    data = np.zeros(num_nz,dtype=float)
    # find block locations
    partition_loc = len(indices) // 2
    block_maps, block_qnums, block_dims = retrieve_blocks(indices, arrows, partition_loc)
    # set each block to identity
    for n in range(block_qnums.shape[1]):
      data[block_maps[n]] = np.eye(block_dims[0,n],block_dims[1,n],dtype=float).ravel()
    
    return cls(data=data, indices=indices, arrows=arrows)
  
  @classmethod
  def from_array(cls, arr: np.ndarray, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor from a dense np.ndarray `arr`. Dimensions of 
    `indices` must match `arr.shape`.
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.ones(num_nz,dtype=arr.dtype), indices=indices, arrows=arrows)

  #########################################
  def reshape(self, *new_dims: Union[tuple,np.ndarray]) -> "SymTensor":
    """
    Reshape a SymTensor object (cf. np.ndarray.reshape). Does not manipulate 
    the tensor data, only changes the `self.partitions` field to reflect the
    new grouping of indices.
    Args: 
      new_dims: either tuple or np.ndarray describing the new tensor shape
    Returns:
      SymTensor: reshaped SymTensor
    """
    new_dims = np.asarray(new_dims, dtype=np.int16).ravel()
    original_dims = np.asarray([ind.dim for ind in self.indices], dtype=np.int16)
    
    if np.array_equal(original_dims,new_dims):
      # trivial case; no reshape
      new_partitions = np.ones(len(original_dims),dtype=np.int16)
    else:
      # compute new partitions
      new_partitions = []
      for n in range(len(new_dims)):
        temp_partition = np.flatnonzero(np.cumprod(original_dims[sum(new_partitions):]) == new_dims[n])
        if len(temp_partition) == 0:
          raise ValueError("Reshape of tensor from original shape {} "
                           "into new shape {} is not possible.".format(
                           tuple(original_dims),tuple(new_dims)))
        else:
          # complicated stuff to properly deal with dim-1 indices 
          if n < (len(new_dims)-1):
            if new_dims[n+1] == 1:
              new_partitions.append(temp_partition[-2]+1)
            else:
              new_partitions.append(temp_partition[-1]+1)
          else:
            new_partitions.append(temp_partition[-1]+1)
    
    new_partitions = np.asarray(new_partitions,dtype=np.int16)
  
    return SymTensor(data=self.data, indices=self.indices, arrows=self.arrows, 
                     partitions=new_partitions)
  
  #########################################
  def transpose(self, *perm_ord: tuple) -> "SymTensor":
    """
    Returns a SymTensor with axes permuted (cf. np.ndarray.transpose).
    Args: 
      perm_ord: either tuple or np.ndarray describing new index order
    Returns:
      SymTensor: permuted SymTensor
    """
    perm_ord = np.asarray(perm_ord, dtype=np.int16).ravel()
    
    if np.array_equal(perm_ord,np.arange(self.ndim)):
      # trivial permutation
      return SymTensor(self.data, self.indices, self.arrows, self.partitions, self.divergence)
    
    else: # non-trivial permutation
      # find permutation order in terms of original indices
      full_ord = unreshape_order(perm_ord, self.partitions)
      
      # generate indices and arrows permuted tensor
      full_num_inds = len(self.indices)
      new_dims = np.array(self.shape_orig)[full_ord]
      new_partition_loc = find_balanced_partition(new_dims)
      new_indices = [self.indices[full_ord[n]] for n in range(full_num_inds)]
      new_arrows = self.arrows[full_ord]
      
      if self.data.size == 0:
        # special case: trivial tensor
        new_data = np.array([],dtype=self.data.dtype)
        
      else: # general case
        # find block maps for original and permuted tensors
        block_maps0, block_qnums0, block_dims0 = retrieve_transpose_blocks(self.indices, self.arrows, new_partition_loc, full_ord)
        block_maps1, block_qnums1, block_dims1 = retrieve_blocks(new_indices, new_arrows, new_partition_loc)
        
        new_data = np.zeros(len(self.data), dtype=self.data.dtype)
        new_data[np.concatenate(block_maps1)] = self.data[np.concatenate(block_maps0)]
      
      return SymTensor(new_data, new_indices, new_arrows, self.partitions[perm_ord], self.divergence)
  
  #########################################
  def __matmul__(self, other) -> "SymTensor":
    """
    Multiply two SymTensor matrices
    """
    if (self.ndim > 2) or (other.ndim > 2):
      raise ValueError("SymTensors must be matrices (ndim = 2) or vectors "
                       "(ndim = 1) in order to use matrix multiply")
    
    return tensordot(self, other, axes=1)
  
  #########################################
  def deepcopy(self) -> "SymTensor":
    """
    Create a deepcopy of a SymTensor, which copies the data-vector in memory 
    """
    return SymTensor(self.data.copy(), self.indices, self.arrows, self.partitions, self.divergence)
  
  def copy(self) -> "SymTensor":
    """
    Create a shallow copy of a SymTensor which does not copy the data-vector
    """
    return SymTensor(self.data, self.indices, self.arrows, self.partitions, self.divergence)
  
  def conj(self) -> "SymTensor":
    """
    Take complex conjugation of the tensor data and reverse the arrows
    """
    return SymTensor(self.data.conj(), self.indices, np.logical_not(self.arrows),
                     self.partitions, np.logical_not(self.divergence))
  
  def __mul__(self, other) -> "SymTensor":
    """
    Multiplication between a scalar and a SymTensor 
    """
    new_tensor = self.deepcopy()
    new_tensor.data = other*new_tensor.data
    return new_tensor
  
  def __rmul__(self, other) -> "SymTensor":
    """
    Multiplication between a scalar and a SymTensor 
    """
    new_tensor = self.deepcopy()
    new_tensor.data = other*new_tensor.data
    return new_tensor
  
  def toarray(self) -> np.ndarray:
    """
    Export a SymTensor to a dense np.ndarray 
    """
    dense_array = np.zeros(np.prod(self.shape), dtype=self.data.dtype)
    if (self.size != 0):
      dense_pos = compute_dense_pos(self.indices, self.arrows) 
      dense_array[dense_pos] = self.data
    
    return dense_array.reshape(self.shape)
  
  
  
###########################################################################
###########################################################################
###########################################################################
def fuse_qnums(qnums_A: np.ndarray, 
               qnums_B: np.ndarray, 
               syms: List[str]) -> np.ndarray:
  """
  Fuse the quantum numbers of two indices under their kronecker product, using 
  the fusion rule of the correpsonding symmetry type.
  Args:
    qnums_A: np.ndarray of ints with shape (m,d1) describing the index quantum 
      numbers, with `m` the number of symmetries and `d1` the index dimension.
    qnums_B: np.ndarray of ints with shape (m,d2) describing the index quantum 
      numbers, with `m` the number of symmetries and `d2` the index dimension.
    syms: list of symmetries in use.
  Returns:
    np.ndarray: np.ndarray of ints with shape (m,d1*d2) describing the index 
      qnum of the fused index (in row-major order).
  """
  comb_qnums = [0]*len(syms)
  for n in range(len(syms)):
    if syms[n] == 'U1':
      comb_qnums[n] = np.add.outer(qnums_A[n,:],qnums_B[n,:]).ravel()
    elif syms[n] == 'Z2':
      comb_qnums[n] = np.abs(np.add.outer(qnums_A[n,:],-qnums_B[n,:]).ravel())
    elif syms[n] == 'Z3':
      comb_qnums[n] = np.mod(np.add.outer(qnums_A[n,:],qnums_B[n,:]).ravel() + 1,3) - 1
    elif syms[n] == 'custom_sym':
      # write your own custom fusion rule here
      raise NotImplementedError("Please write your own fusion rule here")
    else:
      raise NotImplementedError("Unknown symmetry type. Please write your own fusion rule here")
  
  return np.asarray(np.concatenate(comb_qnums,axis = 0).reshape(len(syms),len(comb_qnums[0])), dtype=np.int16)
  
#########################################
def identity_charges(syms: List[str]) -> np.ndarray:
  """
  Give the identity charge associated to symmetries in `syms` (usually, but 
  not necessarily, correspond to '0' elements).
  Args:
    syms: list of symmetries in use.
  Returns:
    nd.array: 1d array containing identity charges for each symmetry in `syms` 
  """
  identity_charges = np.zeros(len(syms),dtype=np.int16)
  for n in range(len(syms)):
    if syms[n] == 'U1':
      identity_charges[n] = 0
    elif syms[n] == 'Z2':
      identity_charges[n] = 0
    elif syms[n] == 'Z3':
      identity_charges[n] = 0
    elif syms[n] == 'custom_sym':
      # write your the identity for your custom fusion rule here
      raise NotImplementedError("Please specify the identity element for your custom symmetry here")
    else:
      # default to '0' for unknown symmetry
      identity_charges[n] = 0
      
  return np.asarray(identity_charges.reshape([len(syms),1]), dtype = np.int16)

#########################################
def combine_indices(indices: List[SymIndex], 
                    arrows: Optional[np.ndarray] = np.zeros(0)
                    ) -> SymIndex:
  """
  Combine multiple SymIndex into a single SymIndex.
  Args:
    indices: list of SymIndex to combine.
    arrows: 1d array of bools describing index orientations. If omitted 
      defaults to all incoming (or False) indices.
  Returns:
    SymIndex: combined index.
  """
  if arrows.size == 0:
    arrows = np.asarray([False]*len(indices), dtype=bool)
    
  comb_index = indices[0].dual(arrows[0])
  for n in range(1,len(indices)):
    comb_index = comb_index @ indices[n].dual(arrows[n])
      
  return comb_index
  
#########################################
def combine_indices_reduced(indices: List[SymIndex], 
                            arrows: np.ndarray, 
                            kept_qnums: np.ndarray, 
                            return_locs: Optional[bool] = False, 
                            strides: Optional[np.ndarray] = np.zeros(0)
                            ) -> Tuple[SymIndex, np.ndarray]:
  """
  Add quantum numbers arising from combining two or more indices into a 
  single index, keeping only the quantum numbers that appear in `kept_qnums`.
  Equilvalent to using `combine_indices` followed by `reduce`, but is 
  generally much more efficient.
  Args:
    indices: list of SymIndex to combine.
    arrows: vector of bools describing index orientations.
    kept_qnums: np.ndarray of shape (m,n) describing qauntum numbers of the 
      qnums which should be kept with 'm' the number of symmetries in use.
    return_locs: if True then return the location of the kept values of the 
      fused indices
    strides: index strides with which to compute the `return_locs` of the kept 
      elements. Defaults to standard strides (based on row-major order) if 
      ommitted. Non-standard strides are used for finding `return_locs` of 
      transposed tensors.  
  Returns:
    SymIndex: the fused index after reduction.
    np.ndarray: locations of the fused SymIndex qnums that were kept. Only 
      provided if `return_locs=True`).
  """
  num_inds = len(indices)
  tensor_dims = [indices[n].dim for n in range(num_inds)]
  
  if num_inds == 1:
    # reduce single index
    if strides.size == 0:
      strides = np.array([1], dtype=np.uint32)
    return indices[0].dual(arrows[0]).reduce(kept_qnums, return_locs=return_locs, strides = strides[0])
  
  else:
    # find size-balanced partition of indices 
    partition_loc = find_balanced_partition(tensor_dims)
    
    # compute quantum numbers for each partition
    left_ind = combine_indices(indices[:partition_loc], arrows[:partition_loc])
    right_ind = combine_indices(indices[partition_loc:], arrows[partition_loc:])
    
    # compute combined qnums
    comb_qnums = fuse_qnums(left_ind.unique_qnums, right_ind.unique_qnums, indices[0].syms) 
    [unique_comb_qnums, comb_labels] = np.unique(comb_qnums, return_inverse=True, axis=1)
    num_unique = unique_comb_qnums.shape[1]

    # intersect combined qnums and kept_qnums
    reduced_qnums, label_to_unique, label_to_kept = intersect2d(unique_comb_qnums, kept_qnums, axis = 1, return_indices = True) 
    map_to_kept = -np.ones(num_unique, dtype=np.int16)
    for n in range(len(label_to_unique)):
      map_to_kept[label_to_unique[n]] = n
      
    new_comb_labels = map_to_kept[comb_labels].reshape([left_ind.num_unique,right_ind.num_unique])
  
  if return_locs:
    if (strides.size != 0):
      # computed locations based on non-trivial strides
      row_pos = combine_index_strides(tensor_dims[:partition_loc], strides[:partition_loc])
      col_pos = combine_index_strides(tensor_dims[partition_loc:], strides[partition_loc:])
      
      # reduce combined qnums to include only those in kept_qnums
      reduced_rows = [0]*left_ind.num_unique
      row_locs = [0]*left_ind.num_unique
      for n in range(left_ind.num_unique):
        temp_label = new_comb_labels[n,right_ind.ind_labels]
        temp_keep = temp_label >= 0
        reduced_rows[n] = temp_label[temp_keep]
        row_locs[n] = col_pos[temp_keep]
      
      reduced_labels = np.concatenate([reduced_rows[n] for n in left_ind.ind_labels])
      reduced_locs = np.concatenate([row_pos[n] + row_locs[left_ind.ind_labels[n]] for n in range(left_ind.dim)])
      
      return SymIndex(reduced_qnums, reduced_labels, indices[0].syms), reduced_locs
    
    else: # trivial strides
      # reduce combined qnums to include only those in kept_qnums
      reduced_rows = [0]*left_ind.num_unique
      row_locs = [0]*left_ind.num_unique
      for n in range(left_ind.num_unique):
        temp_label = new_comb_labels[n,right_ind.ind_labels]
        temp_keep = temp_label >= 0
        reduced_rows[n] = temp_label[temp_keep]
        row_locs[n] = np.where(temp_keep)[0]
      
      reduced_labels = np.concatenate([reduced_rows[n] for n in left_ind.ind_labels])
      reduced_locs = np.concatenate([n*right_ind.dim + row_locs[left_ind.ind_labels[n]] for n in range(left_ind.dim)])
      
      return SymIndex(reduced_qnums, reduced_labels, indices[0].syms), reduced_locs
    
  else:
    # reduce combined qnums to include only those in kept_qnums
    reduced_rows = [0]*left_ind.num_unique
    for n in range(left_ind.num_unique):
      temp_label = new_comb_labels[n,right_ind.ind_labels]
      reduced_rows[n] = temp_label[temp_label >= 0]
    
    reduced_labels = np.concatenate([reduced_rows[n] for n in left_ind.ind_labels])
    
    return SymIndex(reduced_qnums, reduced_labels, indices[0].syms)

#########################################
def compute_qnum_degen(indices: List[SymIndex], 
                       arrows: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
  """
  Add quantum numbers arising from combining two or more indices into a single 
  index, computing only the unique qnums and their degeneracies.
  Args:
    indices: list of SymIndex to be combined.
    arrows: 1d array of bools describing index orientations.
  Returns:
    np.ndarray: array of shape (m,n) describing unique qauntum numbers, with 
      `m` the number of symmetries and `n` the number of unique values.
    np.ndarray: 1d array specifying the degeneracies for each unique quantum 
      number.
  """
  # initialize arrays containing unique qnums and their degens
  unique_comb_degen = indices[0].degens
  unique_comb_qnums = indices[0].dual(arrows[0]).unique_qnums
  
  for n in range(1,len(indices)):
    # fuse the unique charges from each index
    comb_degen = np.kron(unique_comb_degen, indices[n].degens)
    comb_qnums = fuse_qnums(unique_comb_qnums, indices[n].dual(arrows[n]).unique_qnums, indices[n].syms)
    
    # reduce to unique values only
    unique_comb_qnums, ind_labels = np.unique(comb_qnums, return_inverse=True, axis=1)
    unique_comb_degen = np.array([np.sum(comb_degen[ind_labels == n]) for n in range(unique_comb_qnums.shape[1])])
    
  return np.asarray(unique_comb_qnums, dtype=np.int16), np.asarray(unique_comb_degen, np.uint32)

#########################################
def compute_num_nonzero(indices: List[SymIndex], arrows: np.ndarray) -> int:
  """
  Compute the number of non-zero elements in a SymTensor
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
  Returns:
    int: The number of non-zero elements.
  """
  unique_comb_qnums, unique_comb_degen = compute_qnum_degen(indices, arrows) 
  num_nonzero = unique_comb_degen[sum(abs(unique_comb_qnums)) == 0]
  if (num_nonzero.size > 0):
    return num_nonzero.item()
  else:
    return 0

#########################################
def find_balanced_partition(index_dims: np.ndarray) -> int:
  """
  Find the location of the tensor index bipartition that most closely balances
  the total dimensions of each partition.
  Args:
    index_dims (np.ndarray): list of dim of each index.
  Returns:
    int: location of the index partition (i.e. number of indices contained in 
      first side of partition).
  """
  index_dims = np.array(index_dims, dtype=np.int64)
  num_ind = len(index_dims)
  
  # trivial cases
  if num_ind == 0:
    return 0
  elif num_ind == 1:
    return 1
  
  imbalance = [np.abs(np.prod(index_dims[:n]) - np.prod(index_dims[n:])) for n in range(num_ind)]

  return np.argmin(np.array(imbalance))
  
#########################################
def combine_index_strides(index_dims: np.ndarray, strides: np.ndarray) -> np.ndarray:
  """
  Combine multiple indices of some dimensions and strides into a single index, 
  based on row-major order. Used when transposing SymTensors.
  Args:
    index_dims (np.ndarray): list of dim of each index.
    strides (np.ndarray): list of strides of each index.
  Returns:
    np.ndarray: strides of combined index.
  """
  num_ind = len(index_dims)  
  comb_ind_locs = np.arange(0,strides[0]*index_dims[0],strides[0],dtype=np.uint32)
  for n in range(1,num_ind):
    comb_ind_locs = np.add.outer(comb_ind_locs,np.arange(0,strides[n]*index_dims[n],strides[n],dtype=np.uint32)).ravel()

  return comb_ind_locs
    
#########################################
def intersect2d(A: np.ndarray, B: np.ndarray, axis=0, assume_unique=False, return_indices=False) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Extends numpy's intersect1d to find the row or column-wise intersection of 
  two 2d arrays. Takes identical input to numpy intersect1d.
  Args:
    A, B (np.ndarray): arrays of matching widths and datatypes
  Returns:
    ndarray: sorted 1D array of common rows/cols between the input arrays
    ndarray: the indices of the first occurrences of the common values in A. 
      Only provided if return_indices is True.
    ndarray: the indices of the first occurrences of the common values in B. 
      Only provided if return_indices is True.
  """
  if A.ndim == 1:
    return np.intersect1d(A, B,assume_unique=assume_unique,return_indices=return_indices)
  
  elif A.ndim == 2:
    
    if axis == 0:
      ncols = A.shape[1]
      if A.shape[1] != B.shape[1]:
        raise ValueError("array widths must match to intersect")
      
      dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}
      if return_indices:
        C, A_locs, B_locs = np.intersect1d(A.view(dtype), B.view(dtype),assume_unique=assume_unique,return_indices=return_indices)
        return C.view(A.dtype).reshape(-1, ncols), A_locs, B_locs
      else:
        C = np.intersect1d(A.view(dtype), B.view(dtype),assume_unique=assume_unique)
        return C.view(A.dtype).reshape(-1, ncols)
    
    elif axis == 1: 
      A = A.T.copy()
      B = B.T.copy()
      ncols = A.shape[1]
      if A.shape[1] != B.shape[1]:
        raise ValueError("array widths must match to intersect")
      
      dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}
      if return_indices:
        C, A_locs, B_locs = np.intersect1d(A.view(dtype), B.view(dtype),assume_unique=assume_unique,return_indices=return_indices)
        return C.view(A.dtype).reshape(-1, ncols).T, A_locs, B_locs
      else:
        C = np.intersect1d(A.view(dtype), B.view(dtype),assume_unique=assume_unique)
        return C.view(A.dtype).reshape(-1, ncols).T
      
    else:
      raise NotImplementedError("intersection can only be performed on first or second axis")
    
  else:
    raise NotImplementedError("intersect2d is only implemented for 2d arrays")
 
#########################################
def retrieve_blocks(indices: List[SymIndex], arrows: np.ndarray, partition_loc: int) -> (np.ndarray, np.ndarray, np.ndarray):
  """
  Find the location of all non-trivial symmetry blocks from the data vector of
  of SymTensor (when viewed as a matrix across some prescribed index 
  bi-partition).
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
    partition_loc (int): location of tensor partition (i.e. such that the 
      tensor is viewed as a matrix between first partition_loc indices and 
      the remaining indices).
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, each of which 
      contain the location of a symmetry block in the data vector.
    block_qnums (np.ndarray): n-by-m array describing qauntum numbers of each 
      block, with 'n' the number of symmetries and 'm' the number of blocks.
    block_dims (np.ndarray): 2-by-m array describing the dims each block, 
      with 'm' the number of blocks).
  """
  num_inds = len(indices)
  num_syms = indices[0].num_syms
  
  if (partition_loc == 0) or (partition_loc == num_inds):
    # special cases (matrix of trivial height or width)
    num_nonzero = compute_num_nonzero(indices, arrows)
    block_maps = [np.arange(0, num_nonzero, dtype=np.uint32).ravel()]
    block_qnums = np.zeros([num_syms,1],dtype=np.int16)
    block_dims = np.array([[1],[num_nonzero]])
    
    if partition_loc == len(arrows):
        block_dims = np.flipud(block_dims)
        
    return block_maps, block_qnums, block_dims
  
  else:
    # compute qnums of non-trivial blocks
    unique_row_qnums, row_degen = compute_qnum_degen(indices[:partition_loc], arrows[:partition_loc])
    unique_col_qnums, col_degen = compute_qnum_degen(indices[partition_loc:], np.logical_not(arrows[partition_loc:]))
    block_qnums, row_to_block, col_to_block = intersect2d(unique_row_qnums, unique_col_qnums, axis=1, return_indices=True)
    num_blocks = block_qnums.shape[1]
    if num_blocks == 0:
      # special case of tensor with no structually non-zero elements
      return np.zeros(0,dtype=np.uint32), np.zeros(0,dtype=np.int16), np.zeros(0,dtype=np.uint32)
      
    else:
      # calculate number of non-zero elements in each row of the matrix
      row_ind = combine_indices_reduced(indices[:partition_loc], arrows[:partition_loc], block_qnums)
      row_num_nz = col_degen[col_to_block[row_ind.ind_labels]]
      cumulate_num_nz = np.insert(np.cumsum(row_num_nz[0:-1]),0,0).astype(np.uint32)
      
      # calculate mappings for the position in data-vector of each block 
      if num_blocks < 15:
        # faster method for small number of blocks
        row_locs = np.concatenate([(row_ind.ind_labels==n) for n in range(num_blocks)]).reshape(num_blocks,row_ind.dim)
      else:
        # faster method for large number of blocks
        row_locs = np.zeros([num_blocks,row_ind.dim],dtype=bool)
        row_locs[row_ind.ind_labels,np.arange(row_ind.dim)] = np.ones(row_ind.dim,dtype=bool)
      
      block_dims = np.array([[row_degen[row_to_block[n]],col_degen[col_to_block[n]]] for n in range(num_blocks)],dtype=np.uint32).T
      block_maps = [(cumulate_num_nz[row_locs[n,:]][:,None] + np.arange(block_dims[1,n], dtype=np.uint32)[None,:]).ravel() for n in range(num_blocks)]
    
      return block_maps, block_qnums, block_dims
  
#########################################
def retrieve_transpose_blocks(indices: List[SymIndex], arrows: np.ndarray, partition_loc: int, transpose_order: np.ndarray=None) -> List[np.ndarray]:
  """
  Find the location of all non-trivial symmetry blocks from the data vector of
  of SymTensor after transposition (and then viewed as a matrix across some 
  prescribed index bi-partition). Produces and equivalent result to 
  retrieve_blocks acting on a transposed SymTensor, but is much faster.
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
    partition_loc (int): location of tensor partition (i.e. such that the 
      tensor is viewed as a matrix between first partition_loc indices and 
      the remaining indices).
    transpose_order (np.ndarray): order with which to permute the tensor axes. 
  Returns:
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
    block_qnums (np.ndarray): n-by-m array describing qauntum numbers of each 
      block, with 'n' the number of symmetries and 'm' the number of blocks.
    block_dims (np.ndarray): 2-by-m array describing the dims each block, 
      with 'm' the number of blocks).
  """
  if np.array_equal(transpose_order,None) or (np.array_equal(np.array(transpose_order), np.arange(len(indices)))):
    #  special case: no transpose order
    return retrieve_blocks(indices, arrows, partition_loc)
  
  # general case: non-trivial transposition is required
  num_inds = len(indices)
  tensor_dims = np.array([indices[n].dim for n in range(num_inds)],dtype=int)
  strides = np.append(np.flip(np.cumprod(np.flip(tensor_dims[1:]))),1)
    
  # compute qnums of row/cols in original tensor
  orig_partition_loc = find_balanced_partition(tensor_dims)
  orig_width = np.prod(tensor_dims[orig_partition_loc:])
  
  orig_unique_row_qnums = compute_qnum_degen(indices[:orig_partition_loc], arrows[:orig_partition_loc])[0]
  orig_unique_col_qnums, orig_col_degen = compute_qnum_degen(indices[orig_partition_loc:], np.logical_not(arrows[orig_partition_loc:]))
  orig_block_qnums, row_map, col_map = intersect2d(orig_unique_row_qnums, orig_unique_col_qnums, axis=1, return_indices = True)
  orig_num_blocks = orig_block_qnums.shape[1]
  
  if orig_num_blocks == 0:
    # special case: trivial number of non-zero elements
    return [], np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)
  
  orig_row_ind = combine_indices(indices[:orig_partition_loc], arrows[:orig_partition_loc])
  orig_col_ind = combine_indices(indices[orig_partition_loc:], np.logical_not(arrows[orig_partition_loc:]))
  
  # compute row degens (i.e. number of non-zero elements per row)
  inv_row_map = -np.ones(orig_unique_row_qnums.shape[1],dtype=np.int16)
  for n in range(len(row_map)):
    inv_row_map[row_map[n]] = n
    
  all_degens = np.append(orig_col_degen[col_map],0)[inv_row_map[orig_row_ind.ind_labels]]
  all_cumul_degens = np.cumsum(np.insert(all_degens[:-1],0,0)).astype(np.uint32)
  
  # generate vector which translates from dense row position to sparse row position
  dense_to_sparse = np.zeros(orig_width,dtype=np.uint32)
  for n in range(orig_num_blocks):
    dense_to_sparse[orig_col_ind.ind_labels == col_map[n]] = np.arange(orig_col_degen[col_map[n]],dtype=np.uint32)
    
  # define properties of new tensor resulting from transposition
  new_strides = strides[transpose_order]
  new_row_indices = [indices[n] for n in transpose_order[:partition_loc]]
  new_col_indices = [indices[n] for n in transpose_order[partition_loc:]]
  new_row_arrows = arrows[transpose_order[:partition_loc]]
  new_col_arrows = arrows[transpose_order[partition_loc:]]
  
  # print(all_cumul_degens.dtype,dense_to_sparse.dtype)
  
  if (partition_loc == 0):
    # special case: reshape into row vector
    
    # compute qnums of row/cols in transposed tensor
    unique_col_qnums, new_col_degen = compute_qnum_degen(new_col_indices, np.logical_not(new_col_arrows))
    block_qnums, new_row_map, new_col_map = intersect2d(identity_charges(indices[0].syms), unique_col_qnums, axis=1, return_indices = True)
    block_dims = np.array([[1],new_col_degen[new_col_map]], dtype=np.uint32)
    num_blocks = 1
    col_ind, col_locs = combine_indices_reduced(new_col_indices, np.logical_not(new_col_arrows), block_qnums, return_locs=True, strides=new_strides[partition_loc:])
    
    # find location of blocks in transposed tensor (w.r.t positions in original)
    orig_row_posR, orig_col_posR = np.divmod(col_locs[col_ind.ind_labels == 0], orig_width)
    block_maps = [(all_cumul_degens[orig_row_posR] + dense_to_sparse[orig_col_posR]).ravel()]

  elif (partition_loc == len(indices)):
    # special case: reshape into col vector
    
    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_qnum_degen(new_row_indices, new_row_arrows)
    
    block_qnums, new_row_map, new_col_map = intersect2d(unique_row_qnums, identity_charges(indices[0].syms), axis=1, return_indices = True)
    block_dims = np.array([new_row_degen[new_row_map],[1]], dtype=np.uint32)
    num_blocks = 1
    row_ind, row_locs = combine_indices_reduced(new_row_indices, new_row_arrows, block_qnums, return_locs=True, strides=new_strides[:partition_loc])
    
    # find location of blocks in transposed tensor (w.r.t positions in original)
    orig_row_posL, orig_col_posL = np.divmod(row_locs[row_ind.ind_labels == 0], orig_width)
    block_maps = [(all_cumul_degens[orig_row_posL] + dense_to_sparse[orig_col_posL]).ravel()]
    
  else:
    # general case: reshape into a matrix
    
    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_qnum_degen(new_row_indices, new_row_arrows)
    unique_col_qnums, new_col_degen = compute_qnum_degen(new_col_indices, np.logical_not(new_col_arrows))
    block_qnums, new_row_map, new_col_map = intersect2d(unique_row_qnums, unique_col_qnums, axis=1, return_indices = True)
    block_dims = np.array([new_row_degen[new_row_map],new_col_degen[new_col_map]], dtype=np.uint32)
    num_blocks = len(new_row_map)
    
    row_ind, row_locs = combine_indices_reduced(new_row_indices, new_row_arrows, block_qnums, return_locs=True, strides=new_strides[:partition_loc])
    col_ind, col_locs = combine_indices_reduced(new_col_indices, np.logical_not(new_col_arrows), block_qnums, return_locs=True, strides=new_strides[partition_loc:])
    
    # find location of blocks in transposed tensor (w.r.t positions in original)
    block_maps = [0]*num_blocks
    for n in range(num_blocks):
      orig_row_posL, orig_col_posL = np.divmod(row_locs[row_ind.ind_labels == n], orig_width)
      orig_row_posR, orig_col_posR = np.divmod(col_locs[col_ind.ind_labels == n], orig_width)
      block_maps[n] = (all_cumul_degens[np.add.outer(orig_row_posL,orig_row_posR)] +
                       dense_to_sparse[np.add.outer(orig_col_posL,orig_col_posR)]).ravel()
  
  return block_maps, block_qnums, block_dims

#########################################
def tensordot(A: SymTensor, B: SymTensor, axes: int=2) -> SymTensor:
  """
  Compute tensor dot product of two SymTensor along specified axes, using 
  equivalent input to the numpy tensordot function. Reverts to numpy tensordot
  if A and B are numpy arrays.
  Args:
    A (SymTensor): first SymTensor in contraction.
    B (SymTensor): second SymTensor in contraction.
    axes (int or array_like): if integer_like, sum over the last N axes of A 
      and the first N axes of B in order. If array_like, either a list of axes 
      to be summed over or a pair of lists with the first applying to A axes 
      and the second to B axes. 
  Returns:
    SymTensor: SymTensor corresponding to the tensor dot product of the input.
  """
  # using numpy tensordot for numpy arrays
  if (type(A) == np.ndarray) and (type(B) == np.ndarray):
    return np.tensordot(A, B, axes)
  
  # transform input the standard form
  if type(axes) == int:
    axes = [np.arange(A.ndim-axes,A.ndim,dtype=np.int16),np.arange(0,axes,dtype=np.int16)]
  elif type(axes[0]) == int:
    axes = [np.array(axes,dtype=np.int16),np.array(axes,dtype=np.int16)]  
  else:
    axes = [np.array(axes[0],dtype=np.int16),np.array(axes[1],dtype=np.int16)]
  
  # find free indices and index permutation orders (in reshaped indices)
  A_free_res = np.array([np.arange(A.ndim)[n] for n in range(A.ndim) if (np.intersect1d(axes[0],n).size == 0)], dtype=np.int16)
  B_free_res = np.array([np.arange(B.ndim)[n] for n in range(B.ndim) if (np.intersect1d(axes[1],n).size == 0)], dtype=np.int16)
  A_perm_ord_res = np.concatenate([A_free_res, axes[0]])
  B_perm_ord_res = np.concatenate([axes[1], B_free_res])
  
  if axes[0].size == 0: 
    # special case: do outer product
    return outerproduct(A.transpose(A_perm_ord_res),B.transpose(B_perm_ord_res))
    
  elif (len(axes[0]) == A.ndim) and (len(axes[1]) == B.ndim):
    # special case: do inner product
    return innerproduct(A.transpose(A_perm_ord_res),B.transpose(B_perm_ord_res))
      
  else:
    # general case: do matrix product
    
    # find free indices and index permutation orders (in original indices)
    A_axes = unreshape_order(axes[0], A.partitions)
    B_axes = unreshape_order(axes[1], B.partitions)
    A_free = np.array([np.arange(A.ndim_orig)[n] for n in range(A.ndim_orig) if (np.intersect1d(A_axes,n).size == 0)], dtype=np.int16)
    B_free = np.array([np.arange(B.ndim_orig)[n] for n in range(B.ndim_orig) if (np.intersect1d(B_axes,n).size == 0)], dtype=np.int16)
    A_perm_ord = np.concatenate([A_free,A_axes])
    B_perm_ord = np.concatenate([B_axes,B_free])
    
    # initialize output tensor properties
    A_free_ind = [A.indices[A_free[nA]] for nA in range(len(A_free))]
    B_free_ind = [B.indices[B_free[nB]] for nB in range(len(B_free))]
    
    C_indices = A_free_ind + B_free_ind
    C_arrows = np.concatenate([A.arrows[A_free], B.arrows[B_free]])
    C_data = np.zeros(compute_num_nonzero(C_indices, C_arrows), dtype=A.data.dtype)
    C_partitions = np.concatenate([A.partitions[A_free_res], B.partitions[B_free_res]])
    
    if ((len(A.data) > 0) and (len(B.data) > 0)) and (len(C_data) > 0):
      # find blocks from each tensor
      ### t0 = time.time()
      A_block_maps, A_block_qnums, A_block_dims = retrieve_transpose_blocks(A.indices, A.arrows, len(A_free), A_perm_ord)
      B_block_maps, B_block_qnums, B_block_dims = retrieve_transpose_blocks(B.indices, B.arrows, len(B_axes), B_perm_ord)
      C_block_maps, C_block_qnums, C_block_dims = retrieve_blocks(C_indices, C_arrows, len(A_free))
      ### print("time for block comp:", time.time() - t0)  
      
      # construct map between qnum labels for each tensor
      common_qnums, A_to_common, B_to_common = intersect2d(A_block_qnums, B_block_qnums, axis=1, return_indices=True)
      C_to_common = intersect2d(C_block_qnums, common_qnums, axis=1, return_indices=True)[1]
      
      # perform tensor contraction oone block at a time
      ### t1 = time.time()
      for n in range(common_qnums.shape[1]):
        nA = A_to_common[n]
        nB = B_to_common[n] 
        nC = C_to_common[n] 
        
        C_data[C_block_maps[nC].ravel()] = (A.data[A_block_maps[nA].reshape(A_block_dims[:,nA])] @ 
                                           B.data[B_block_maps[nB].reshape(B_block_dims[:,nB])]).ravel()
    
    ### print("time for mults:", time.time() - t1)  
    return SymTensor(C_data, C_indices, C_arrows, C_partitions)
  
#########################################
def outerproduct(A: SymTensor, B: SymTensor) -> SymTensor:
  """
  Compute the outer product of two SymTensor.
  Args:
    A (SymTensor): first input SymTensor (automatically flattened if not 
      already 1-dimensional).
    B (SymTensor): second input SymTensor (automatically flattened if not 
      already 1-dimensional).
  Returns:
    SymTensor: SymTensor corresponding to the outer product.
  """
  # initialize fields of output tensor
  C_indices = A.indices + B.indices
  C_arrows = np.concatenate([A.arrows, B.arrows])
  C_data = np.zeros(compute_num_nonzero(C_indices, C_arrows), dtype=A.data.dtype)
  C_partitions = np.concatenate([A.partitions, B.partitions])
  
  if ((len(A.data) > 0) and (len(B.data) > 0)) and (len(C_data) > 0):
    # find the location of the zero block in the output
    C_block_maps, C_block_qnums, C_block_dims = retrieve_blocks(C_indices, C_arrows, A.ndim)
    zero_block_label = intersect2d(C_block_qnums, identity_charges(A.indices[0].syms), axis=1, return_indices=True)[1].item()
  
    # store the result of the outer product in the output tensor data
    C_data[C_block_maps[zero_block_label].ravel()] = np.outer(A.data, B.data).ravel()
  
  return SymTensor(C_data, C_indices, C_arrows, partitions=C_partitions)

#########################################
def innerproduct(A: SymTensor, B: SymTensor) -> float:
  """
  Compute the inner product of two SymTensor of matching dimensions.
  Args:
    A (SymTensor): first input SymTensor (automatically flattened if not 
      already 1-dimensional).
    B (SymTensor): second input SymTensor (automatically flattened if not 
      already 1-dimensional).
  Returns:
    float: the scalar product of the two tensors.
  """
  return np.dot(A.data, B.data)
  
#########################################
def compute_dense_pos(indices: List[SymIndex], arrows: np.ndarray) -> np.ndarray:
  """
  Compute the location that the elements of a data vector would appear in the 
  dense tensor representation. Used in exporting SymTensor to dense tensors.
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
  Returns:
    np.ndarray: interger vector denoting the location that each element of the 
      data vector would appear in the corresponding the dense tensor (with 
      row-major order assumed).
  """
  num_inds = len(arrows)
  if num_inds == 1:
    # reduce single index
    return indices[0].dual(arrows[0]).reduce(identity_charges(indices[0].syms),return_locs=True)[1]
    
  else:
    # find the size-balanced bipartition
    tensor_dims = [indices[n].dim for n in range(num_inds)]
    partition_loc = find_balanced_partition(tensor_dims)
    mat_width = np.prod(tensor_dims[partition_loc:])
        
    # compute qnums of non-trivial blocks
    unique_row_qnums = compute_qnum_degen(indices[:partition_loc], arrows[:partition_loc])[0]
    unique_col_qnums = compute_qnum_degen(indices[partition_loc:], np.logical_not(arrows[partition_loc:]))[0]
    block_qnums = intersect2d(unique_row_qnums, unique_col_qnums, axis=1)
    if block_qnums.size == 0:
      # trivial case: no structually non-zero elements
      return np.zeros(0, dtype=np.uint32)
      
    else:
      num_blocks = block_qnums.shape[1]
      
      # calculate row/col quantum numbers (enumerated in basis of non-trivial block qnums)
      left_ind, left_locs = combine_indices_reduced(indices[:partition_loc], arrows[:partition_loc], block_qnums, return_locs=True)
      right_ind, right_locs = combine_indices_reduced(indices[partition_loc:], np.logical_not(arrows[partition_loc:]), block_qnums, return_locs=True)
      
      # positions of each element
      qnum_col_pos = [right_locs[right_ind.ind_labels==n] for n in range(num_blocks)]
      
      return np.concatenate([mat_width*left_locs[n] + qnum_col_pos[left_ind.ind_labels[n]] for n in range(left_ind.dim)])
  
#########################################
def unreshape_order(order: np.ndarray, partitions: np.ndarray) -> np.ndarray:
  """
  Takes an index order and produces the equivalent order if the tensor were to
  be reshaped into its original indices (i.e. with trivial index partitions).
  Args:
    order (np.ndarray): an index permutation order.
    partitions (np.ndarray): 1d array describing the grouping of indices.
  Returns:
    np.ndarray: equivalent order expressed for the original (un-paritioned)
      indices
  """
  num_ind = np.sum(partitions)
  partitions = np.array(partitions,dtype=np.int16)
  
  if (len(partitions) == num_ind):
    # only trivial partitions
    return order
  
  else:
    # non-trivial partitions
    trivial_ord = np.arange(num_ind)
    cumul_ind_num = np.insert(np.cumsum(partitions),0,0)
    return np.concatenate([trivial_ord[cumul_ind_num[n]:cumul_ind_num[n+1]] for n in order])
  
#########################################
def ncon(tensors: List[SymTensor], connects_in: List[np.ndarray], 
         cont_order: np.ndarray=np.array([]), check_network: bool=True, check_dense: bool=False) -> SymTensor:
  """
  Network CONtractor based on that of https://arxiv.org/abs/1402.0939. 
  Evaluates a tensor network via a sequence of pairwise contractions using 
  tensordot. Can perform both partial traces and outer products. Valid both 
  for networks of SymTensor and for networks composed of numpy arrays.
  Args:
    tensors (List[SymTensor]): list of tensors in the network (either of type 
      SymTensor or of type np.ndarray).
    connects_in (List[np.ndarray]): list of 1d arrays (or lists) specifying 
      the index labels on the corresponding tensor.
    cont_order (np.ndarray, optional): 1d array specifying the order to 
      contract the internal indices of the network. Defaults to ascending
      order.
    check_network (bool, optional): sets whether to check the consistancy of 
      the input network. 
    check_dense (bool, optional): if True then ncon routine will evaluate the 
      network twice, once with SymTensor and once after exporting to tensors 
      numpy arrays. Useful for testing SymTensor routines.
  Returns:
    SymTensor: result of contracting the network.
  """
  # check inputs if enabled
  if check_network:
    check_ncon_inputs(tensors, connects_in, cont_order)
  
  # put inputs into a list if necessary
  if type(tensors) is not list:
    tensors = [tensors]
    connects_in = [connects_in]  
  
  # make sure that each element of connects is an array 
  connects = [np.array(connects_in[ele], dtype=int) for ele in range(len(connects_in))]

  # generate contraction order if necessary
  flat_connect = np.concatenate(connects)
  if len(cont_order) == 0:
    cont_order = np.unique(flat_connect[flat_connect > 0])
  else:
    cont_order = np.array(cont_order)

  # check whether to use ncon for SymTensors or for np.ndarray
  sym_in_use = (type(tensors[0]) == SymTensor)

  # do dense calculation (for testing purposes)
  if sym_in_use and check_dense:
    dense_tensors = [tensor.toarray() for tensor in tensors]
    t0 = time.time()
    final_dense_tensor = ncon(dense_tensors, connects, cont_order, check_network=False)
    time_dense = time.time() - t0
    
  # do all partial traces
  for ele in range(len(tensors)):
    num_cont = len(connects[ele]) - len(np.unique(connects[ele]))
    if num_cont > 0:
      raise NotImplementedError('partial traces still under contruction')
      # tensors[ele], connects[ele], cont_ind = partial_trace(tensors[ele], connects[ele])
      # cont_order = np.delete(cont_order, np.intersect1d(cont_order,cont_ind,return_indices=True)[1])

  # do all binary contractions
  while len(cont_order) > 0:
    # identify tensors to be contracted
    cont_ind = cont_order[0]
    locs = [ele for ele in range(len(connects)) if sum(connects[ele] == cont_ind) > 0]

    # do binary contraction using tensordot
    cont_many, A_cont, B_cont = np.intersect1d(connects[locs[0]], connects[locs[1]], assume_unique=True, return_indices=True)
    tensors.append(tensordot(tensors[locs[0]], tensors[locs[1]], axes=(A_cont, B_cont)))
    connects.append(np.append(np.delete(connects[locs[0]], A_cont), np.delete(connects[locs[1]], B_cont)))

    # remove contracted tensors from list and update cont_order
    del tensors[locs[1]]
    del tensors[locs[0]]
    del connects[locs[1]]
    del connects[locs[0]]
    cont_order = np.delete(cont_order,np.intersect1d(cont_order,cont_many, assume_unique=True, return_indices=True)[1])

  # do all outer products
  while len(tensors) > 1:
    tensors[-2] = tensordot(tensors[-2], tensors[-1], axes=0)
    connects[-2] = np.append(connects[-2],connects[-1])
    del tensors[-1]
    del connects[-1]

  # do final permutation
  if len(connects[0]) > 0:
    final_tensor = tensors[0].transpose(np.argsort(-connects[0]))
  else:
    final_tensor = tensors[0]
    if not sym_in_use:
      # take 0-dim numpy array to scalar 
      final_tensor = final_tensor.item()
    
  # check correctness against dense contraction (for testing purposes)
  if sym_in_use and check_dense:
    time_sym = time.time() - time_dense - t0
    tolerance = 1e-10
    if len(connects[0]) > 0:
      cont_error = LA.norm(final_tensor.toarray() - final_dense_tensor) / max(LA.norm(final_dense_tensor),tolerance)
    else:
      cont_error = LA.norm(final_tensor - final_dense_tensor) / max(LA.norm(final_dense_tensor),tolerance)
      
    print("contraction error: ", cont_error)    
    print("cont time for Sym: ", time_sym)
    print("cont time for dense: ", time_dense)    
    assert cont_error <= tolerance
    
  # return the contracted network
  return final_tensor

#########################################
def check_ncon_inputs(tensors: List[SymTensor], connects_in: List[np.ndarray], cont_order: np.ndarray=np.array([])) -> bool:
  """
  Function for checking that a tensor network is defined consistently, taking
  the same inputs as the ncon routine. Can detect many common errors (e.g. 
  mis-matched tensor dimensions and mislabelled tensors) and for networks of 
  SymTensors also checks that quantum numbers and index arrows match. This 
  routine is automatically called by ncon if check_network is enabled.
  Args:
    tensors (List[SymTensor]): list of SymTensor in the contraction.
    connects_in (List[np.ndarray]): list of arrays, each of which contains the 
      index labels of the corresponding tensor.
    cont_order (np.ndarraym optional): 1d array describing the order with 
      which tensors are to be contracted.
  Returns:
    bool: True if network is consistant.
  """
  # put inputs into a list if necessary
  if type(tensors) is not list:
    tensors = [tensors]
    connects = [connects_in]  
    
  # check whether to use ncon for SymTensors or for np.ndarray
  sym_in_use = (type(tensors[0]) == SymTensor)
  
  # make sure that each element of connects is an array 
  connects = [np.array(connects_in[ele], dtype=int) for ele in range(len(connects_in))]

  # generate contraction order if necessary
  flat_connect = np.concatenate(connects)
  if len(cont_order) == 0:
    cont_order = np.unique(flat_connect[flat_connect > 0])
  else:
    cont_order = np.array(cont_order)
  
  # generate dimensions, find all positive and negative labels
  dims_list = [np.array(tensor.shape, dtype=int) for tensor in tensors]
  flat_connect = np.concatenate(connects)
  pos_ind = flat_connect[flat_connect > 0]
  neg_ind = flat_connect[flat_connect < 0]

  # check that lengths of lists match
  if len(dims_list) != len(connects):
    raise ValueError(('NCON error: %i tensors given but %i index sublists given')
                     %(len(dims_list), len(connects)))
  
  # check that tensors have the right number of indices
  for ele in range(len(dims_list)):
    if len(dims_list[ele]) != len(connects[ele]):
      raise ValueError(('NCON error: number of indices does not match number of labels on tensor %i: '
                        '%i-indices versus %i-labels')%(ele,len(dims_list[ele]),len(connects[ele])))

  # check that contraction order is valid
  if not np.array_equal(np.sort(cont_order),np.unique(pos_ind)):
    raise ValueError(('NCON error: invalid contraction order'))

  # check that negative indices are valid
  for ind in np.arange(-1,-len(neg_ind)-1,-1):
    if sum(neg_ind == ind) == 0:
      raise ValueError(('NCON error: no index labelled %i') %(ind))
    elif sum(neg_ind == ind) > 1:
      raise ValueError(('NCON error: more than one index labelled %i')%(ind))

  # check that positive indices are valid and contracted tensor dimensions match
  flat_dims = np.concatenate(dims_list)
  for ind in np.unique(pos_ind):
    if sum(pos_ind == ind) == 1:
      raise ValueError(('NCON error: only one index labelled %i')%(ind))
    elif sum(pos_ind == ind) > 2:
      raise ValueError(('NCON error: more than two indices labelled %i')%(ind))

    cont_dims = flat_dims[flat_connect == ind]
    if cont_dims[0] != cont_dims[1]:
      raise ValueError(('NCON error: tensor dimension mismatch on index labelled %i: '
                        'dim-%i versus dim-%i')%(ind,cont_dims[0],cont_dims[1]))
  
  if sym_in_use:
    # locate tensor and index that each positive label appears on
    for curr_ind in cont_order:
      locs = []
      for tensor_pos in range(len(connects)):
        for ind_pos in range(len(connects[tensor_pos])):
          if connects[tensor_pos][ind_pos] == curr_ind:
            locs.append(tensor_pos)
            locs.append(ind_pos)
  
      # check quantum numbers on joining indices match up
      if not np.array_equal(tensors[locs[0]].joint_indices[locs[1]].qnums,tensors[locs[2]].joint_indices[locs[3]].qnums):
        raise ValueError(('Quantum numbers mismatch between index %i of tensor '
                          '%i and index %i of tensor %i')%(locs[1],locs[0],locs[3],locs[2]))
            
      # check arrows on joining indices match up (incoming to outgoing)
      if tensors[locs[0]].joint_arrows[locs[1]] == tensors[locs[2]].joint_arrows[locs[3]]:
        raise ValueError(('Arrow mismatch between index %i of tensor '
                          '%i and index %i of tensor %i')%(locs[1],locs[0],locs[3],locs[2]))
      
  # network is valid!
  return True

  
  
  
  