import numpy as np
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable
"""
-----------------------------------------------------------------------------
Glen's SymIndex class
-----------------------------------------------------------------------------
Defines an object class for tensor indices, which are used to store information
about the symmetry groups in use and the qauntum numbers (or qnums) of an 
index w.r.t. the symmetries. The class also contains the functions and methods 
necessary for fusing and conjugating indices.

Quick reference:
Important attributes: (*.syms, *.unique_qnums, *.ind_labels, *.qnums, *.dim)

Important methods:
  q1.reduce(q2) - reduce index q1 to contain only the index values whose qnum 
    overlaps with a qnum from q2.
  q1.dual() - tansform index q1 dual qnums.
  
Important static methods (or helper functions for indices):
  fuse(q1,q2) - fuse index q1 with index q2 (equivalent notation: q1 @ q2).
  intersect_inds(q1, q2) - create a new index of unique qnums that contains 
    only the common qnums from q1 and q2.
  
Important class methods (for creation of indices):
  create - create an index from a numpy array of qnums and list of symmetries.
  rand - create an index of some specified dim, populated with random qnums 
    from the specified symmetry.
  identity - create an index of some specified dim, populated with the 
    identity qnum from the specified symmetry.
""" 


class SymIndex:
  def __init__(self, unique_qnums: np.ndarray, ind_labels: np.ndarray, syms: List[str]):
    """
    Args:
      unique_qnums: np.ndarray of shape (m,n) with `m` the number of symmetries 
        in use and `n` the number of distinct symmetry sectors in the index.
      ind_labels: 1d array labelling the qnum of each index value in terms of 
        the unique_qnums.
      syms: list of strings describing the symmetries in use (currently 
        supported: 'U1', 'Z2', 'Z3').
    """
    if type(syms) != list:
      self.syms = [syms]
    else:
      self.syms = syms
    
    self.unique_qnums = np.asarray(unique_qnums, dtype=np.int16).reshape(len(syms),-1)
    self.ind_labels = np.asarray(ind_labels, dtype=np.int16)
  
  
  @staticmethod
  def fuse(indA:"SymIndex", indB:"SymIndex") -> "SymIndex":
    """
    Fuse the quantum numbers of two indices under their kronecker product, 
    using the fusion rule of the correpsonding symmetry type.
    """
    # fuse the unique quantum numbers from each index
    num_syms = len(indA.syms)
    comb_qnums = [0]*num_syms
    for n in range(num_syms):
      qnums_A = indA.unique_qnums[n,:]
      qnums_B = indB.unique_qnums[n,:]
      
      """
      Implement fusion for each symmetry type. Fusions rules should take a 
      pair of 1d arrays `qnums_A` and `qnums_B` of dims dA and dB, and return 
      the 1d array of dim dA*dB representing the kronecker product under the 
      prescribed rule.  
      """
      if indA.syms[n] == 'U1':
        # U1 fusion rule is addition
        comb_qnums[n] = np.add.outer(qnums_A,qnums_B).ravel()
      elif indA.syms[n] == 'Z2':
        # Z2 fusion rule is addition modulo 2
        comb_qnums[n] = np.mod(np.add.outer(qnums_A,qnums_B),2).ravel()
      elif indA.syms[n] == 'Z3':
        # Z3 fusion rule is addition modulo 3
        comb_qnums[n] = np.mod(np.add.outer(qnums_A,qnums_B).ravel() + 1,3) - 1
      elif indA.syms[n] == 'custom_sym':
        # write your own custom fusion rule here
        raise NotImplementedError("Please write your own fusion rule here")
      else:
        raise NotImplementedError("Unknown symmetry type. Please write your own fusion rule here")
    
    #find the unique qnums of the fused index
    comb_qnums = np.asarray(np.concatenate(comb_qnums,axis = 0).reshape(num_syms,indA.num_unique*indB.num_unique), dtype=np.int16)
    [unique_qnums, new_labels] = np.unique(comb_qnums, return_inverse=True, axis=1)
    new_labels = np.asarray(new_labels.reshape(indA.num_unique,indB.num_unique), dtype=np.int16)
      
    # find new labels using broadcasting (could use np.tile but less efficient)
    ind_labels = new_labels[(indA.ind_labels[:,None] + np.zeros([1,indB.dim],dtype=np.int16)).ravel(),
                            (indB.ind_labels[None,:] + np.zeros([indA.dim,1],dtype=np.int16)).ravel()]
  
    return SymIndex(unique_qnums, ind_labels, indA.syms)
  
  
  def dual(self, take_dual: bool=True) -> "SymIndex":
    """
    Return the dual index of `self` if 'take_dual=True'.
    """
    if take_dual:
      new_unique_qnums = [0]*self.num_syms
      for n in range(self.num_syms):
        if self.syms[n] == 'U1':
          # U1 duality is negation of sign
          new_unique_qnums[n] = -self.unique_qnums[n,:]
        elif self.syms[n] == 'Z2':
          # Z2 elements are self-dual
          new_unique_qnums[n] = self.unique_qnums[n,:]
        elif self.syms[n] == 'Z3':
          # Z3 duality is negation of sign
          new_unique_qnums[n] = -self.unique_qnums[n,:]
        elif self.syms[n] == 'custom_sym':
          # write your duality transformation for custom fusion rule here
          raise NotImplementedError("Please write your own fusion duality rule here")
        else:
          raise NotImplementedError("Unknown symmetry type. Please write your own fusion duality rule here")
      
      return SymIndex(np.concatenate(new_unique_qnums,axis=0).reshape(self.num_syms,self.num_unique), self.ind_labels, self.syms)
    else:
      return self
  
  
  @classmethod
  def identity(cls, dim: int, syms: List[str]):
    """
    Create a SymIndex of dimension `dim` with qnums equal to the group 
    identity element of the index `syms` (usually, but not always, equal to 
    the `0` element)
    """
    # put syms into a list if necessary
    if type(syms) == list:
      num_syms = len(syms)
    elif type(syms) == str:
      num_syms = 1
      syms = [syms]
    
    num_syms = len(syms)
    idn_charges = np.zeros([num_syms,1],dtype=np.int16)
    for n in range(num_syms):
      if syms[n] == 'U1':
        idn_charges[n,:] = 0
      elif syms[n] == 'Z2':
        idn_charges[n,:] = 0
      elif syms[n] == 'Z3':
        idn_charges[n,:] = 0
      elif syms[n] == 'custom_sym':
        # write your the identity for your custom fusion rule here
        raise NotImplementedError("Please specify the identity element for your custom symmetry here")
      else:
        # default to '0' for unknown symmetry
        idn_charges[n,:] = 0
        
    ind_labels = np.zeros(dim,dtype=np.int16)
    
    return cls(idn_charges, ind_labels, syms)
  
  
  @classmethod
  def rand(cls, dim: int, syms: List[str]):
    """
    Create a SymIndex of dimension `dim` with random permissible qnums 
    according to symmetries specified by `syms`.
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
        # random integers selected from gaussian distribution about 0
        qnums[n] = (2*np.random.randn(dim)).astype(np.int16)
      elif syms[n] == 'Z2':
        # random integers from set {0,1}
        qnums[n] = np.round(np.random.rand(dim)).astype(np.int16)
      elif syms[n] == 'Z3':
        # random integers from set {-1,0,1}
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
      syms: list of strings describing the symmetries in use. Can also be a 
        single string when using only a single symmetry.
    """
    # put syms into a list if necessary
    if type(syms) == list:
      num_syms = len(syms)
    elif type(syms) == str:
      num_syms = 1
      syms = [syms]
    
    # reshape qnums if necessary
    qnums = np.array(qnums,dtype=np.int16).reshape(num_syms,-1)
    # check consistancy
    if qnums.shape[0] != num_syms:
      raise ValueError(('height of qnums array (%i) should match number '
                        'of symmetries in use (%i)')%(qnums.shape[0],num_syms))
    # find unique values
    [unique_qnums, ind_labels] = np.unique(qnums, return_inverse=True, axis=1)
    
    return cls(np.asarray(unique_qnums, dtype=np.int16), np.asarray(ind_labels, dtype=np.int16) , syms)
  
  
  def remove_degen(self) -> "SymIndex":
    """ Create a new SymIndex where each unique qnum has singular degeneracy
    (i.e. such that index_out.qnums == index_in.unique_qnums). """
    return SymIndex(self.unique_qnums, np.arange(self.num_unique, dtype=np.int16), self.syms)
  
  def copy(self) -> "SymIndex":
    """ Create a copy of a SymIndex """
    return SymIndex(self.unique_qnums.copy(), self.ind_labels.copy(), self.syms.copy())
  
  def __matmul__(self, other: "SymIndex") -> "SymIndex":
    """ Define matrix multiplication for two SymIndex as their fusion. """
    return SymIndex.fuse(self, other)
  
  def __repr__(self):
    """ Define display output for SymIndex objects """
    return (str(type(self)) +", sym:"+ str(self.syms)+", dim:" +str(self.dim)+ "\n")
  
  def __eq__(self,other):
    """ Check whether two SymIndex are equal """
    if not np.array_equal(self.unique_qnums,other.unique_qnums):
      return False
    
    if not np.array_equal(self.ind_labels,other.ind_labels):
      return False
    
    if len(self.syms) != len(other.syms):
      return False
    
    for k in range(len(self.syms)):
      if self.syms[k] != other.syms[k]:
        return False  
    return True
      
  def __ne__(self,other):
    """ Check whether two SymIndex are different """
    return not(self == other)
  
  @staticmethod
  def intersect_inds(indA: "SymIndex", indB: "SymIndex") -> Tuple["SymIndex", np.ndarray, np.ndarray]:
    """
    Create a new SymIndex of unique qnums that contains only the intersection 
    of the unique quantum numbers from `indA` and `indB`. 
    Returns:
      SymIndex: the index containing only the intersection of qnums.
      np.ndarray: the `label_map_A' from the qnums of indA to the intersected 
        qnums, such that label_map_A[indA.ind_labels] produces labels in terms
        of the intersected labels.
      np.ndarray: the `label_map_B' from the qnums of indB to the intersected 
        qnums, such that label_map_B[indB.ind_labels] produces labels in terms
        of the intersected labels. 
    """
    # find intersection of unique qnums
    if indA.num_syms == 1:
      common_qnums, indA_to_common, indB_to_common = np.intersect1d(indA.unique_qnums, indB.unique_qnums, return_indices=True)
    
    else:
      # recast intersection of matrix cols into form compatible with intersect1d
      # (arrays need to be C-order contigous in memory for `view` to work properly)
      ele_type = indA.qnums.dtype
      dtype={'names':['f{}'.format(i) for i in range(indA.num_syms)],'formats':indA.num_syms * [ele_type]}
      temp_qnums, indA_to_common, indB_to_common = np.intersect1d(np.ascontiguousarray(indA.unique_qnums.T).view(dtype),
                                                                   np.ascontiguousarray(indB.unique_qnums.T).view(dtype),
                                                                   return_indices=True)
      common_qnums = temp_qnums.view(ele_type).reshape(-1, indA.num_syms).T
      
    return SymIndex(common_qnums, np.arange(len(indA_to_common), dtype=np.int16), indA.syms), indA_to_common, indB_to_common
    
  def reduce(self, 
             ind_kept: "SymIndex", 
             return_locs: bool = False, 
             strides: int = 1) -> Tuple["SymIndex", np.ndarray]:
    """
    Reduce the dim of `self` to keep only the index values whose qnum 
    intersects with a qnum in `ind_kept`.
    Args:
      ind_kept: SymIndex describing the qnums to be kept in the fused index.
      return_locs: if True, also return the index locations of kept values.
      strides: index strides used to compute the `return_locs`.
    Returns:
      SymIndex: the SymIndex of reduced dimension.
      np.ndarray: locations of kept values in the output index.
    """
    # generate index of common qnums and the label map between self and kept
    ind_reduced, self_to_common = SymIndex.intersect_inds(self, ind_kept)[:2]
    
    # construct the map from common_qnums to self qnums
    label_map = -np.ones(self.num_unique, dtype=np.int16)
    for n in range(len(self_to_common)):
      label_map[self_to_common[n]] = n
    
    # re-label self index qnums in terms of kept qnums
    new_ind_labels = label_map[self.ind_labels]
    
    # truncate index values not contained in `ind_kept`
    reduced_locs = (new_ind_labels >= 0)
    ind_reduced.ind_labels = np.asarray(new_ind_labels[reduced_locs], dtype=np.int16)
    
    if return_locs:
      return ind_reduced, np.asarray(strides*np.flatnonzero(reduced_locs), dtype=np.uint32)
    else:
      return ind_reduced
  
  
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
    return np.bincount(self.ind_labels)
  
  
  
  
  