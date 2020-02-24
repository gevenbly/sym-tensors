from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import linalg as LA
import itertools
import time
from timeit import default_timer as timer
from typing import List, Union, Any, Tuple, Type, Optional, Dict, Iterable

# import the symmetric tensor toolbox
from sym_index import SymIndex
from sym_tensor import SymTensor
import sym_toolbox as ST

"""
Testing script for ncon routine. Generates a completely random tensor network 
(within the specified parameters) and populates the network with randomly 
initialized SymTensors. Evaluates the network using ncon and then compares 
against the result from exporting to dense tensors and evaluating using the 
ncon for dense networks. Asserts that the two results should be within the 
specified tolerance.
"""

tolerance = 1e-10 # tolerance for difference between symmetric and dense contractions
num_tests = 100 # total number of tests to run
chi = 6 # bond dimension of indices
max_neg = 4 # maximum number of open indices in network
max_pos = 8 # maximum number of internal indices in network 
max_rank = 6 # maximum rank of any single tensor
with_divergence = True # use tensors with non-trivial divergence

for test_iter in range(num_tests):
  all_syms = ['U1','Z2','Z3'] 
  num_syms = np.random.randint(1,4)
  syms = [all_syms[np.random.randint(0,3)] for n in range(num_syms)]
  
  num_neg = np.random.randint(max_neg+1)
  num_pos = np.random.randint(max_pos+1)
  
  if (num_neg==0) and (num_pos==0):
    num_pos = np.random.randint(max_pos) + 1
  
  neg_labels = np.arange(0,-num_neg,-1)-1
  pos_labels = np.arange(num_pos)+1
  
  pos_ind = [SymIndex.rand(chi, syms) for n in range(len(pos_labels))]
  neg_ind = [SymIndex.rand(chi, syms) for n in range(len(neg_labels))]
  
  neg_arrows = list(np.random.rand(num_neg)>0.5)
  pos_arrows = [True]*num_pos
  
  all_labels = list(neg_labels) + list(pos_labels) + list(pos_labels) 
  all_ind = neg_ind + pos_ind + pos_ind
  all_arrows = neg_arrows + pos_arrows + list(np.logical_not(np.asarray(pos_arrows,dtype=bool)))
  num_ind = len(all_labels)
  
  perm_ord = np.argsort(np.random.rand(num_ind))
  
  new_labels = [all_labels[perm_ord[n]] for n in range(num_ind)]
  new_ind = [all_ind[perm_ord[n]] for n in range(num_ind)]
  new_arrows = [all_arrows[perm_ord[n]] for n in range(num_ind)]
  
  tensor_list = []
  tensor_labels = []
  current_count = 0
  current_tensor = 0
  while current_count < num_ind:
    num_remain = num_ind - current_count
    num_current = min(np.random.randint(max_rank) + 1,num_remain)
    
    indices = [new_ind[n] for n in range(current_count,current_count+num_current)]
    arrows = [new_arrows[n] for n in range(current_count,current_count+num_current)]
    labels = [new_labels[n] for n in range(current_count,current_count+num_current)]
    if with_divergence:
      divergence = SymIndex.rand(1, syms)
    else:
      divergence = SymIndex.identity(1, syms)
    
    tensor_list = tensor_list + [SymTensor.rand(indices,arrows,divergence)]
    tensor_labels = tensor_labels + [labels]
    
    current_count += num_current
  
  num_tensors = len(tensor_list)
  dense_tensor_list = [tensor_list[n].toarray() for n in range(num_tensors)]
  dense_final_tensor = ST.ncon(dense_tensor_list, tensor_labels)
  
  tensor_list_orig = tensor_list.copy()
  final_tensor = ST.ncon(tensor_list, tensor_labels)
  
  if type(final_tensor) == SymTensor:
    err_temp = (LA.norm(dense_final_tensor-final_tensor.toarray()))
  else:
    err_temp = (LA.norm(dense_final_tensor-final_tensor))

  assert err_temp < tolerance
  print("tests : % 3d, error : % 5.2e, norm: %5.2e" %(test_iter, err_temp, LA.norm(dense_final_tensor)))


