
import time
import numpy as np
from typing import List, Tuple, Type, Optional

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
  
  def __init__(self, unique_qnums: np.ndarray, ind_labels: np.ndarray, syms: List[str]) -> None:
    """
    Args:
      unique_qnums (np.ndarray): np.ndarray of shape (m,D,) with m the number 
        of symmetries in use and D the number of distinct symmetry sectors of 
        the index. Each col describes the qnums relative to the corresponding 
        symmetry in syms.
      ind_labels (np.ndarray): 1d array labelling the qnum of each index value 
        in terms of the cols of the unique_qnums.
      syms (List[str]): list of strings describing the symmetries in use 
        (currently supported: 'U1', 'Z2', 'Z3').
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
    return np.array([np.sum(self.ind_labels == n) for n in range(self.num_unique)], dtype=np.uint32)
  
  #########################################
  @classmethod
  def create(cls, qnums: np.ndarray, syms: List[str]):
    """
    Create a SymIndex from an array of quantum numbers.
    Args:
      qnums (np.ndarray): np.ndarray of shape (m,D,), where 'm' is the number 
        of symmetries and 'D' is the bond dimension. Can also be a 1d array 
        when using only a single symmetry.
      syms (List[str]): list of strings describing the symmetries in use. Can 
        also be a string when using only a single symmetry.
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
      raise ValueError(('height of qnums array (%i) should match number of symmetries in use (%i)')%(qnums.shape[0],num_syms))
    
    # find unique values
    [unique_qnums, ind_labels] = np.unique(qnums, return_inverse=True, axis=1)
    
    return cls(unique_qnums, ind_labels.astype(np.int16), syms)
  
  @classmethod
  def rand(cls, dim: int, syms: List[str]):
    """
    Create a SymIndex of random qnums.
    Args:
      dim (int): dimension of the index to create
      syms (List[str]): list of strings describing the symmetries in use. 
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
      
    return cls(unique_qnums, ind_labels.astype(np.int16), syms)
  
  @classmethod
  def identity(cls, dim: int, syms: List[str]):
    """
    Create a SymIndex of trivial qnums (i.e. the group identity element).
    Args:
      dim (int): dimension of the index to create.
      syms (List[str]): list of strings describing the symmetries in use.
    """
    # put syms into a list if necessary
    if type(syms) == list:
      num_syms = len(syms)
    elif type(syms) == str:
      num_syms = 1
      syms = [syms]
    
    unique_qnums = identity_charges(syms).reshape([num_syms,1]).astype(np.int16)
    ind_labels = np.zeros(dim,dtype=np.int16)
    
    return cls(unique_qnums, ind_labels, syms)
  
  #########################################
  def dual(self, take_dual: bool = True) -> "SymIndex":
    """
    Compute the dual charges of a SymIndex.
    Args:
      take_dual (bool, optional): sets whether to take dual; defaults to True.
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
  def __matmul__(self, other) -> "SymIndex":
    """
    Overload matrix multiplication for SymIndex's to take their kronecker product
    Args:
      self: a SymIndex of dim d1
      other: a SymIndex of dim d2
    Returns:
      SymIndex: a SymIndex of dim d1*d2
    """
    # check that combination is valid
    if (self.syms != other.syms):
      raise ValueError("Symmetries of indices to be combined do not match")
    
    # fuse the unique charges from each index, then compute new unique charges
    comb_qnums = fuse_qnums(self.unique_qnums, other.unique_qnums, self.syms)
    [unique_qnums, new_labels] = np.unique(comb_qnums, return_inverse=True, axis=1)
    new_labels = new_labels.reshape(self.num_unique,other.num_unique).astype(np.int16)
    
    # find new labels using broadcasting (could use np.tile but less efficient)
    ind_labels = new_labels[(self.ind_labels[:,None] + np.zeros([1,other.dim],dtype=np.int16)).ravel(),
                            (other.ind_labels[None,:] + np.zeros([self.dim,1],dtype=np.int16)).ravel()]
  
    return SymIndex(unique_qnums, ind_labels, self.syms)
  
  #########################################
  def reduce(self, kept_qnums: np.ndarray, return_locs: bool = False, strides: int = 1) -> ("SymIndex", np.ndarray):
    """
    Reduce the dim of a SymIndex to keep only the index values that intersect kept_qnums
    Args:
      kept_qnums (np.ndarray): array of unique quantum numbers to keep.
      return_locs (bool, optional): if True, also return the output index 
        locations of kept values.
    Returns:
      SymIndex: index of reduced dimension.
      np.ndarray: output index locations of kept values.
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
    new_ind_labels = reduced_ind_labels[reduced_locs].astype(np.int16)
    
    if return_locs:
      return (SymIndex(reduced_qnums, new_ind_labels, self.syms), 
              strides*np.flatnonzero(reduced_locs).astype(np.uint32))
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
               divergence: np.ndarray = np.zeros(0)) -> None:
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
    self.arrows = arrows.copy()
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
    numpy.random.rand) 
    Args:
      indices (List[SymIndex]): list of SymIndex.
      arrows (np.ndarray): 1d array of bool denoting index orientations.
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.random.rand(num_nz), indices=indices, arrows=arrows)
  
  @classmethod
  def zeros(cls, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor with all elements initialized to zero (cf. 
    numpy.zeros) 
    Args:
      indices (List[SymIndex]): list of SymIndex.
      arrows (np.ndarray): 1d array of bool denoting index orientations.
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.zeros(num_nz,dtype=float), indices=indices, arrows=arrows)
  
  @classmethod
  def ones(cls, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor with all (structually non-zero) elements initialized 
    to unity (cf. numpy.ones) 
    Args:
      indices (List[SymIndex]): list of SymIndex.
      arrows (np.ndarray): 1d array of bool denoting index orientations.
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.ones(num_nz,dtype=float), indices=indices, arrows=arrows)
  
  @classmethod
  def from_array(cls, arr: np.ndarray, indices: List[SymIndex], arrows: np.ndarray):
    """
    Construct a SymTensor from a dense np.ndarray
    Args:
      arr (np.ndarray): array to convert.
      indices (List[SymIndex]): list of SymIndex.
      arrows (np.ndarray): 1d array of bool denoting index orientations.
    """
    num_nz = compute_num_nonzero(indices, arrows)
    return cls(data=np.ones(num_nz,dtype=arr.dtype), indices=indices, arrows=arrows)

  #########################################
  def reshape(self, *new_dims: tuple) -> "SymTensor":
    """
    Reshape a SymTensor object (cf. np.ndarray.reshape). Does not manipulate 
    the tensor data, only changes the self.partitions field.
    Args: 
      new_dims: either tuple or np.ndarray describing the new tensor shape
    Returns:
      SymTensor: reshaped SymTensor
    """
    new_dims = np.array(new_dims, dtype=np.int16).ravel()
    original_dims = np.array([ind.dim for ind in self.indices], dtype=np.int16)
    
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
    
    new_partitions = np.array(new_partitions,dtype=np.int16)
  
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
    if type(perm_ord[0]) == np.ndarray:
      perm_ord = perm_ord[0]
    
    perm_ord = np.array(tuple(perm_ord),dtype=np.int16)
    # find permutation order in terms of original indices
    full_ord = unreshape_order(perm_ord, self.partitions)
    
    # generate indices and arrows permuted tensor
    full_num_inds = len(self.indices)
    new_dims = np.array(self.shape_orig)[full_ord]
    new_partition_loc = find_balanced_partition(new_dims)
    new_indices = [self.indices[full_ord[n]] for n in range(full_num_inds)]
    new_arrows = self.arrows[full_ord]
    
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
    Args:
      self: a SymTensor matrix (i.e. SymTensor with ndim = 2)
      other: a SymTensor matrix (i.e. SymTensor with ndim = 2)
    Returns:
      SymTensor: a SymTensor matrix corresponding to: (self @ other)
    """
    
    if (self.ndim > 2) or (other.ndim > 2):
      raise ValueError("SymTensors must be matrices (ndim = 2) or vectors "
                       "(ndim = 1) in order to use matrix multiply")
    
    return tensordot(self, other, axes=1)
  
  #########################################
  def deepcopy(self) -> "SymTensor":
    """
    Create a deepcopy of a SymTensor, which copies the datavector in memory 
    """
    return SymTensor(self.data.copy(), self.indices, self.arrows, self.partitions, self.divergence)
  
  def copy(self) -> "SymTensor":
    """
    Create a shallow copy of a SymTensor which does not duplicate the datavector
    """
    return SymTensor(self.data, self.indices, self.arrows, self.partitions, self.divergence)
  
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
def fuse_qnums(qnums_A: np.ndarray, qnums_B: np.ndarray, syms: List[str]) -> np.ndarray:
  """
  Fuse the quantum numbers of two indices under their kronecker product, using 
  the fusion rule of the correpsonding symmetry type.
  Args:
    qnums_A (np.ndarray): n-by-d1 dimensional array of ints describing the 
      index quantum numbers, with n the number of symmetries and d1 the index 
      dimension.
    qnums_B (np.ndarray): n-by-d2 dimensional array of quantum numbers.
  Returns:
    np.ndarray: n-by-(d1*d2) dimensional array of the fused qnums from the 
      kroncker product of the indices
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
  
  return np.concatenate(comb_qnums,axis = 0).reshape(len(syms),len(comb_qnums[0]))
  
#########################################
def identity_charges(syms) -> np.ndarray:
  """
  Give the identity charge associated to a symmetries of a SymIndex 
  (usually correspond to '0' elements)
  Args:
    self: a SymIndex
  Returns:
    nd.array: vector of identity charges for each symmetry in self 
  """
  identity_charges = np.zeros(len(syms),dtype=int)
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
      
  return identity_charges.reshape([len(syms),1])

#########################################
def combine_indices(indices: List[SymIndex], arrows: np.ndarray) -> SymIndex:
  """
  Combine multiple SymIndex into a single SymIndex.
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
  Returns:
    SymIndex: combined index.
  """
  comb_index = indices[0].dual(arrows[0])
  for n in range(1,len(indices)):
    comb_index = comb_index @ indices[n].dual(arrows[n])
      
  return comb_index
  
#########################################
def combine_indices_reduced(indices: List[SymIndex], arrows: np.ndarray, kept_qnums: np.ndarray, return_locs: Optional[bool] = False, 
                            strides: Optional[np.ndarray] = np.zeros(0)) -> (SymIndex, np.ndarray):
  """
  Add quantum numbers arising from combining two or more indices into a 
  single index, keeping only the quantum numbers that appear in 'kept_qnums'.
  Equilvalent to using "combine_indices" followed by "reduce", but is 
  generally much more efficient.
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
    kept_qnums (np.ndarray): n-by-m array describing qauntum numbers of the 
      qnums which should be kept with 'n' the number of symmetries.
    return_locs (bool, optional): if True then return the location of the kept
      values of the fused indices
    strides (np.ndarray, optional): index strides with which to compute the 
      return_locs of the kept elements. Defaults to trivial strides (based on
      row major order) if ommitted.
  Returns:
    SymIndex: the fused index after reduction.
    np.ndarray: locations of the fused SymIndex qnums that were kept.
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
def compute_qnum_degen(indices: List[SymIndex], arrows: np.ndarray) -> (np.ndarray, np.ndarray):
  """
  Add quantum numbers arising from combining two or more indices into a single 
  index, computing only the unique qnums and their degeneracies
  Args:
    indices (List[SymIndex]): list of SymIndex.
    arrows (np.ndarray): vector of bools describing index orientations.
  Returns:
    np.ndarray: n-by-m array describing unique qauntum numbers, with 'n' the 
      number of symmetries and 'm' the number of unique values.
    np.ndarray: vector of degeneracies for each unique quantum number.
  """
  # initialize arrays containing unique qnums and their degens
  unique_comb_degen = indices[0].degens
  unique_comb_qnums = indices[0].dual(arrows[0]).unique_qnums
  
  for n in range(1,len(indices)):
    # fuse the unique charges from each index
    comb_degen = np.kron(unique_comb_degen,indices[n].degens)
    comb_qnums = fuse_qnums(unique_comb_qnums, indices[n].dual(arrows[n]).unique_qnums, indices[n].syms)
    
    # reduce to unique values only
    unique_comb_qnums, ind_labels = np.unique(comb_qnums, return_inverse=True, axis=1)
    unique_comb_degen = np.array([np.sum(comb_degen[ind_labels == n]) for n in range(unique_comb_qnums.shape[1])])
    
  return unique_comb_qnums, unique_comb_degen

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
    block_maps (List[np.ndarray]): list of integer arrays, which each 
      containing the location of a symmetry block in the data vector.
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
    block_maps = [np.arange(0, num_nonzero, dtype=np.uint64).ravel()]
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
      
      # calculate mappings for the position in datavector of each block 
      if num_blocks < 15:
        # faster method for small number of blocks
        row_locs = np.concatenate([(row_ind.ind_labels==n) for n in range(num_blocks)]).reshape(num_blocks,row_ind.dim)
      else:
        # faster method for large number of blocks
        row_locs = np.zeros([num_blocks,row_ind.dim],dtype=bool)
        row_locs[row_ind.ind_labels,np.arange(row_ind.dim)] = np.ones(row_ind.dim,dtype=bool)
      
      # block_dims = np.array([row_degen[row_to_block],col_degen[col_to_block]], dtype=np.uint32)
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
    # no transpose order
    return retrieve_blocks(indices, arrows, partition_loc)
  
  else:
    # non-trivial transposition is required
    num_inds = len(indices)
    tensor_dims = np.array([indices[n].dim for n in range(num_inds)],dtype=int)
    strides = np.append(np.flip(np.cumprod(np.flip(tensor_dims[1:]))),1)
    
    # define properties of new tensor resulting from transposition
    new_strides = strides[transpose_order]
    new_row_indices = [indices[n] for n in transpose_order[:partition_loc]]
    new_col_indices = [indices[n] for n in transpose_order[partition_loc:]]
    new_row_arrows = arrows[transpose_order[:partition_loc]]
    new_col_arrows = arrows[transpose_order[partition_loc:]]
    
    # compute qnums of row/cols in transposed tensor
    unique_row_qnums, new_row_degen = compute_qnum_degen(new_row_indices, new_row_arrows)
    unique_col_qnums, new_col_degen = compute_qnum_degen(new_col_indices, np.logical_not(new_col_arrows))
    block_qnums, new_row_map, new_col_map = intersect2d(unique_row_qnums, unique_col_qnums, axis=1, return_indices = True)
    block_dims = np.array([new_row_degen[new_row_map],new_col_degen[new_col_map]], dtype=np.uint32)
    num_blocks = len(new_row_map)
    
    row_ind, row_locs = combine_indices_reduced(new_row_indices, new_row_arrows, block_qnums, return_locs=True, strides=new_strides[:partition_loc])
    col_ind, col_locs = combine_indices_reduced(new_col_indices, np.logical_not(new_col_arrows), block_qnums, return_locs=True, strides=new_strides[partition_loc:])
    
    # compute qnums of row/cols in original tensor
    orig_partition_loc = find_balanced_partition(tensor_dims)
    orig_width = np.prod(tensor_dims[orig_partition_loc:])
    
    orig_unique_row_qnums = compute_qnum_degen(indices[:orig_partition_loc], arrows[:orig_partition_loc])[0]
    orig_unique_col_qnums, orig_col_degen = compute_qnum_degen(indices[orig_partition_loc:], np.logical_not(arrows[orig_partition_loc:]))
    orig_block_qnums, row_map, col_map = intersect2d(orig_unique_row_qnums, orig_unique_col_qnums, axis=1, return_indices = True)
    orig_num_blocks = orig_block_qnums.shape[1]
    
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
  equivalent input to the numpy tensordot function.
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
  if axes == 0: # do outer product
    return outerproduct(A,B)
    
  else: # do tensor contraction
    # transform input the standard form
    if type(axes) == int:
      axes = [np.arange(A.ndim-axes,A.ndim,dtype=np.int16),np.arange(0,axes,dtype=np.int16)]
    elif type(axes[0]) == int:
      axes = [np.array(axes,dtype=np.int16),np.array(axes,dtype=np.int16)]  
    else:
      axes = [np.array(axes[0],dtype=np.int16),np.array(axes[1],dtype=np.int16)]
      
    # find free indices and index permutation orders (in original indices)
    A_axes = unreshape_order(axes[0], A.partitions)
    B_axes = unreshape_order(axes[1], B.partitions)
    A_free = np.array([np.arange(A.ndim_orig)[n] for n in range(A.ndim_orig) if (np.intersect1d(A_axes,n).size == 0)], dtype=np.int16)
    B_free = np.array([np.arange(B.ndim_orig)[n] for n in range(B.ndim_orig) if (np.intersect1d(B_axes,n).size == 0)], dtype=np.int16)
    A_perm_ord = np.concatenate([A_free,A_axes])
    B_perm_ord = np.concatenate([B_axes,B_free])
    
    # find free indices and index permutation orders (in reshaped indices)
    A_free_res = np.array([np.arange(A.ndim)[n] for n in range(A.ndim) if (np.intersect1d(axes[0],n).size == 0)], dtype=np.int16)
    B_free_res = np.array([np.arange(B.ndim)[n] for n in range(B.ndim) if (np.intersect1d(axes[1],n).size == 0)], dtype=np.int16)
    
    # initialize output tensor properties
    A_free_ind = [A.indices[A_free[nA]] for nA in range(len(A_free))]
    B_free_ind = [B.indices[B_free[nB]] for nB in range(len(B_free))]
    
    C_indices = A_free_ind + B_free_ind
    C_arrows = np.concatenate([A.arrows[A_free], B.arrows[B_free]])
    C_data = np.zeros(compute_num_nonzero(C_indices, C_arrows), dtype=A.data.dtype)
    C_partitions = np.concatenate([A.partitions[A_free_res], B.partitions[B_free_res]])
    
    # find blocks from each tensor
    t0 = time.time()
    A_block_maps, A_block_qnums, A_block_dims = retrieve_transpose_blocks(A.indices, A.arrows, len(A_free), A_perm_ord)
    B_block_maps, B_block_qnums, B_block_dims = retrieve_transpose_blocks(B.indices, B.arrows, len(B_axes), B_perm_ord)
    C_block_maps, C_block_qnums, C_block_dims = retrieve_blocks(C_indices, C_arrows, len(A_free))
    print("time for block comp:", time.time() - t0)  
    
    # construct map between qnum labels for each tensor
    common_qnums, A_to_common, B_to_common = intersect2d(A_block_qnums, B_block_qnums, axis=1, return_indices=True)
    C_to_common = intersect2d(C_block_qnums, common_qnums, axis=1, return_indices=True)[1]
    
    # perform tensor contraction oone block at a time
    t1 = time.time()
    for n in range(common_qnums.shape[1]):
      nA = A_to_common[n]
      nB = B_to_common[n] 
      nC = C_to_common[n] 
      
      C_data[C_block_maps[nC].ravel()] = (A.data[A_block_maps[nA].reshape(A_block_dims[:,nA])] @ 
                                         B.data[B_block_maps[nB].reshape(B_block_dims[:,nB])]).ravel()
    
    print("time for mults:", time.time() - t1)  
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
  
  # find the location of the zero block in the output
  C_block_maps, C_block_qnums, C_block_dims = retrieve_blocks(C_indices, C_arrows, A.ndim)
  zero_block_label = intersect2d(C_block_qnums, identity_charges(A.indices[0].syms), axis=1, return_indices=True)[1].item()

  # store the result of the outer product in the output tensor data
  C_data[C_block_maps[zero_block_label].ravel()] = np.outer(A.data, B.data).ravel()
  
  return SymTensor(C_data, C_indices, C_arrows, partitions=C_partitions)
  
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
    

  
  
  
  
  
  
  