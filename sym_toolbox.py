import time
import numpy as np
from numpy import linalg as LA
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable
# requires the SymTensor class to be imported 
from sym_index import SymIndex
from sym_tensor import SymTensor
"""
-----------------------------------------------------------------------------
Glen's toolbox of functions for SymTensors
-----------------------------------------------------------------------------
Quick reference:
  tensordot - basic function for contracting a pair of SymTensor, which shares
    a common input with numpy tensordot.
  partialtrace - function for taking a partial trace of a SymTensor.
  ncon - contractor for networks of SymTensor.
  check_ncon_inputs - function for checking that a network is consistant. 
""" 

#########################################
def tensordot(A: SymTensor, B: SymTensor, axes: int=2, do_again=True) -> SymTensor:
  """
  Compute tensor dot product of two SymTensor along specified axes, using 
  equivalent input to the numpy tensordot function. Reverts to numpy tensordot
  if A and B are numpy arrays.
  Args:
    A: first SymTensor in contraction.
    B: second SymTensor in contraction.
    axes (int or array_like): if integer_like, sum over the last N axes of A 
      and the first N axes of B in order. If array_like, either a list of axes 
      to be summed over or a pair of lists with the first applying to A axes 
      and the second to B axes. 
  Returns:
    SymTensor: tensor dot product of the input tensors.
  """

  if (type(A) == np.ndarray) and (type(B) == np.ndarray):
    # using numpy tensordot for numpy arrays
    return np.tensordot(A, B, axes)
  
  if (type(A) != SymTensor) or (type(B) != SymTensor):
    # do scalar multiplication if one or both inputs are scalar
    return A*B
  
  # transform input `axes` to a the standard form
  if type(axes) == int:
    axes = [np.arange(A.ndim-axes,A.ndim,dtype=np.int16),np.arange(0,axes,dtype=np.int16)]
  elif type(axes[0]) == int:
    axes = [np.array(axes,dtype=np.int16),np.array(axes,dtype=np.int16)]  
  else:
    axes = [np.array(axes[0],dtype=np.int16),np.array(axes[1],dtype=np.int16)]
  
  # find free indices and index permutation orders in reshaped indices
  A_free_axes = np.array([np.arange(A.ndim)[n] for n in range(A.ndim) if (np.intersect1d(axes[0],n).size == 0)], dtype=np.int16)
  B_free_axes = np.array([np.arange(B.ndim)[n] for n in range(B.ndim) if (np.intersect1d(axes[1],n).size == 0)], dtype=np.int16)
  A_order = np.concatenate([A_free_axes,axes[0]])
  B_order = np.concatenate([axes[1],B_free_axes])
  A_new_parts = [A.partitions[A_order[n]] for n in range(A.ndim)]
  B_new_parts = [B.partitions[B_order[n]] for n in range(B.ndim)]
  
  # find free indices and index permutation orders in original indices
  A_num_cont_orig = sum(A.index_groups[axes[0]])
  B_num_cont_orig = sum(B.index_groups[axes[1]])
  A_free_orig = np.concatenate(A_new_parts)[:(A.ndim_orig-A_num_cont_orig)]
  B_free_orig = np.concatenate(B_new_parts)[B_num_cont_orig:]
  
  if (len(axes[0]) == A.ndim) and (len(axes[1]) == B.ndim): # special case: inner product
    if np.array_equal(A.divergence.dual().unique_qnums,B.divergence.unique_qnums):
      return np.dot(A.transpose(A_order).transpose_data().data, B.transpose(B_order).transpose_data().data)
    else:
      return 0
  
  # define output tensor properties and initialize
  C_indices = [A.indices_orig[n] for n in A_free_orig] + [B.indices_orig[n] for n in B_free_orig]
  C_arrows = np.concatenate([A.arrows_orig[A_free_orig], B.arrows_orig[B_free_orig]])
  C_divergence = A.divergence @ B.divergence
  C_index_groups = np.concatenate([A.index_groups[A_free_axes],B.index_groups[B_free_axes]])
  C_cumul_groups = np.insert(np.cumsum(C_index_groups),0,0)
  C_partitions = [np.arange(C_cumul_groups[n], C_cumul_groups[n+1], dtype=np.int16) for n in range(len(C_index_groups))]
  C_tensor = SymTensor(A.dtype, C_indices, C_arrows, partitions=C_partitions, divergence=C_divergence)
  
  if axes[0].size == 0: # special case: outer product
    if ((len(A.data) > 0) and (len(B.data) > 0)) and (len(C_tensor.data) > 0):
      # find the location of the zero block in the output
      C_block_maps, C_block_ind, C_block_dims = C_tensor.retrieve_blocks(len(A_free_orig), lagging_div=C_divergence)
      zero_block_label = SymIndex.intersect_inds(C_block_ind, A.divergence)[1].item()
    
      # store the result of the outer product in the output tensor data
      C_tensor.data[C_block_maps[zero_block_label].ravel()] = np.outer(A.transpose_data().data, B.transpose_data().data).ravel()  
    return C_tensor
  
  else: # general case: do matrix product
    if ((len(A.data) > 0) and (len(B.data) > 0)) and (len(C_tensor.data) > 0):
      A_block_maps, A_block_ind, A_block_dims = A.retrieve_blocks(len(A_free_orig), transpose_order=np.concatenate(A_new_parts), leading_div=A.divergence)
      B_block_maps, B_block_ind, B_block_dims = B.retrieve_blocks(B_num_cont_orig, transpose_order=np.concatenate(B_new_parts), lagging_div=B.divergence)
      C_block_maps, C_block_ind, C_block_dims = C_tensor.retrieve_blocks(len(A_free_orig), leading_div=A.divergence, lagging_div=B.divergence)
      
      # construct map between qnum labels for each tensor and the common qnums
      common_ind, A_to_common, B_to_common = SymIndex.intersect_inds(A_block_ind, B_block_ind)
      C_to_common = SymIndex.intersect_inds(C_block_ind, common_ind)[1]
      
      # perform tensor contraction oone block at a time
      for n in range(common_ind.dim):
        nA = A_to_common[n]
        nB = B_to_common[n] 
        nC = C_to_common[n] 
        
        C_tensor.data[C_block_maps[nC].ravel()] = (A.data[A_block_maps[nA].reshape(A_block_dims[:,nA])] @ 
                                                   B.data[B_block_maps[nB].reshape(B_block_dims[:,nB])]).ravel()  
  
    return C_tensor
  
#########################################
def partialtrace(tensor: SymTensor, labels: np.ndarray) -> Tuple[SymTensor, np.ndarray, np.ndarray]:
  """
  Compute the partial trace of a SymTensor. Works by generating an identity
  SymTensor to contract with the existing tensor (not the most 
  computationally efficient approach; could be improved later).
  Args:
    tensor: the SymTensor for the partial trace.
    labels_A: integer labels for indices of A. Pairs of matching positive 
      labels indicate indices to be contracted with each other.
  Returns:
    SymTensor: the tensor after the partial trace.
    new labels
    labels for contracted indices
  """
  # identify labels of contracted indices
  cont_inds = [ele for ele in np.unique(labels) if sum(labels == ele) == 2]
  
  # find positions of contracted indices
  ind_top = []
  ind_bott = []
  for n in range(len(cont_inds)):
    temp_loc = np.flatnonzero(labels == cont_inds[n])
    ind_top = ind_top + [temp_loc[0]]
    ind_bott = ind_bott + [temp_loc[1]]
    
  cont_labels = np.asarray(ind_top+ind_bott, dtype=np.int16)
  
  # find new labels for tensor after partial trace
  new_labels = [labels[n] for n in range(len(labels)) if len(np.intersect1d(cont_inds,labels[n]))==0]
  
  if type(tensor) == SymTensor:
    # generate the identity tensor
    idn_indices = [tensor.indices[cont_labels[n]] for n in range(len(cont_labels))]
    idn_arrows = np.logical_not(np.asarray([tensor.arrows[cont_labels[n]] for n in range(len(cont_labels))]))
    idn_tensor = SymTensor.eye(idn_indices,idn_arrows)
    
    return tensordot(tensor,idn_tensor,axes=(cont_labels,np.arange(len(cont_labels)))), new_labels, cont_inds
  
  elif type(tensor) == np.ndarray:
    # generate labels appropriate for einsum
    ein_labels = -np.ones(len(labels),dtype=np.int16)
    for k in range(len(ind_top)):
      ein_labels[ind_top[k]] = k
      ein_labels[ind_bott[k]] = k
      
    remain_labels = np.arange(len(ind_top), len(labels) - len(ind_top),dtype=np.int16)  
    ein_labels[ein_labels<0] = remain_labels
    
    # explicitly cast each element as integer (otherwise weird einsum bug occurs) 
    ein_labels = [int(ein_labels[n]) for n in range(len(ein_labels))]
    remain_labels = [int(remain_labels[n]) for n in range(len(remain_labels))]
    
    return np.einsum(tensor,ein_labels,remain_labels), new_labels, cont_inds

#########################################
def ncon(tensors: List[SymTensor],
         connects_in: List[np.ndarray], 
         cont_order: Optional[np.ndarray]=None,
         check_network: Optional[bool]=True,
         check_dense: Optional[bool]=False) -> SymTensor:
  """
  Network CONtractor based on that of https://arxiv.org/abs/1402.0939. 
  Evaluates a tensor network via a sequence of pairwise contractions using 
  tensordot. Can perform both partial traces and outer products. Valid both 
  for networks of SymTensor and for networks composed of numpy arrays.
  Args:
    tensors: list of tensors in the network (either of type SymTensor or of 
      type np.ndarray).
    connects_in: list of 1d arrays (or lists) specifying the index labels on 
      the corresponding tensor.
    cont_order: 1d array specifying the order to contract the internal indices 
      of the network. Defaults to ascending order.
    check_network: sets whether to check the consistancy of the input network. 
    check_dense: if True then ncon routine will evaluate the network twice, 
      once with SymTensor and once after exporting to tensors dense numpy 
      arrays. Useful for testing SymTensor routines.
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
  if cont_order is None:
    cont_order = np.unique(flat_connect[flat_connect >= 0])
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
      tensors[ele], connects[ele], cont_ind = partialtrace(tensors[ele], connects[ele])
      cont_order = np.delete(cont_order, np.intersect1d(cont_order,cont_ind,return_indices=True)[1])

  # c=0
  # do all binary contractions
  while len(cont_order) > 0:
    # print("hello",c)
    # c += 1
    
    # identify tensors to be contracted
    cont_ind = cont_order[0]
    locs = [ele for ele in range(len(connects)) if sum(connects[ele] == cont_ind) > 0]

    # do binary contraction using tensordot
    cont_many, A_cont, B_cont = np.intersect1d(connects[locs[0]], connects[locs[1]], assume_unique=True, return_indices=True)
    # print("L",locs[0],locs[1],A_cont,B_cont)
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
    final_tensor = tensors[0].transpose(np.argsort(-np.asarray(connects[0])))
  else:
    final_tensor = tensors[0]
    if not sym_in_use:
      # take 0-dim numpy array to scalar 
      final_tensor = final_tensor.item()
  # final_tensor = tensors[0]
    
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
def check_ncon_inputs(tensors: List[SymTensor], 
                      connects_in: List[np.ndarray], 
                      cont_order: Optional[np.ndarray]=None) -> bool:
  """
  Function for checking that a tensor network is defined consistently, taking
  the same inputs as the ncon routine. Can detect many common errors (e.g. 
  mis-matched tensor dimensions and mislabelled tensors) and for networks of 
  SymTensors also checks that quantum numbers and index arrows match. This 
  routine is automatically called by ncon if check_network is enabled.
  Args:
    tensors: list of SymTensor in the contraction.
    connects_in: list of arrays, each of which contains the index labels of 
      the corresponding tensor.
    cont_order: 1d array describing the order with which tensors are to be 
      contracted.
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
  if cont_order is None:
    cont_order = np.unique(flat_connect[flat_connect >= 0])
  else:
    cont_order = np.array(cont_order)
  
  # generate dimensions, find all positive and negative labels
  dims_list = [np.array(tensor.shape, dtype=int) for tensor in tensors]
  flat_connect = np.concatenate(connects)
  pos_ind = flat_connect[flat_connect >= 0]
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
      if not np.array_equal(tensors[locs[0]].indices[locs[1]].qnums,tensors[locs[2]].indices[locs[3]].qnums):
        raise ValueError(('Quantum numbers mismatch between index %i of tensor '
                          '%i and index %i of tensor %i')%(locs[1],locs[0],locs[3],locs[2]))
            
      # check arrows on joining indices match up (incoming to outgoing)
      if tensors[locs[0]].arrows[locs[1]] == tensors[locs[2]].arrows[locs[3]]:
        raise ValueError(('Arrow mismatch between index %i of tensor '
                          '%i and index %i of tensor %i')%(locs[1],locs[0],locs[3],locs[2]))
      
  # network is valid!
  return True

  
  
  
  