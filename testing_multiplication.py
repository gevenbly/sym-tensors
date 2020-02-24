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
Set of tests for the multiplication of two SymTensors using tensordot. 
Includes all possibilities: (matrix @ matrix, vector @ matrix, matrix @ vector, 
vector @ vector (outer product), vector @ vector (inner product)). Compares
result against the numpy tensordot performed on the corresponding dense 
tensors, and produces and assertion error if the results differ by more than 
the tolerance.
"""

number_of_tests = 100
tolerance = 1e-10

for k in range(number_of_tests):
  # set index dimensions
  chi = 5
  
  # select symmetry (or symmetries) at random
  all_syms = ['U1','Z2','Z3'] 
  num_syms = np.random.randint(1,4)
  syms = [all_syms[np.random.randint(0,3)] for n in range(num_syms)]
  
  ##############################################
  # Test: matrix @ matrix
  A_ndim = 5 # number of indices on A
  B_ndim = 6 # number of indices on B
  cont_ndim = 3 # number of contracted indices
  ##############################################
  
  # create index of random qnums
  q_ind = SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = np.array([False]*A_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  A = SymTensor.rand(A_indices,A_arrows,A_divergence)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = np.array([True]*B_ndim, dtype=bool)
  B_divergence = SymIndex.rand(1, syms)
  B = SymTensor.rand(B_indices,B_arrows,B_divergence)
  
  # select random indices to contract together
  A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
  B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]
  
  # contract symmetric tensors using tensordot
  C = ST.tensordot(A, B, axes=[A_cont,B_cont])
  
  # contract corresponding dense numpy arrays using numpy tensordot
  C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])
  
  # compare results
  cont_error0 = LA.norm(C.toarray()-C_full)
  assert cont_error0 < tolerance
 
  ##############################################
  # Test: vector @ matrix
  A_ndim = 4 # number of indices on A
  B_ndim = 6 # number of indices on B
  cont_ndim = 4 # number of contracted indices
  ##############################################
  
  # create index of random qnums
  q_ind = ST.SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = np.array([False]*A_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  A = ST.SymTensor.rand(A_indices,A_arrows,A_divergence)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = np.array([True]*B_ndim, dtype=bool)
  B_divergence = SymIndex.rand(1, syms)
  B = ST.SymTensor.rand(B_indices,B_arrows,B_divergence)
  
  # select random indices to contract together
  A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
  B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]
  
  # contract symmetric tensors using tensordot
  C = ST.tensordot(A, B, axes=[A_cont,B_cont])
  
  # contract corresponding dense numpy arrays using numpy tensordot
  C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])
  
  # compare results
  cont_error1 = LA.norm(C.toarray()-C_full)
  assert cont_error1 < tolerance
  
  ##############################################
  # Test: matrix @ vector
  A_ndim = 8 # number of indices on A
  B_ndim = 6 # number of indices on B
  cont_ndim = 6 # number of contracted indices
  ##############################################
  
  # create index of random qnums
  q_ind = ST.SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = np.array([False]*A_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  A = ST.SymTensor.rand(A_indices,A_arrows,A_divergence)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = np.array([True]*B_ndim, dtype=bool)
  B_divergence = SymIndex.rand(1, syms)
  B = ST.SymTensor.rand(B_indices,B_arrows,B_divergence)
  
  # select random indices to contract together
  A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
  B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]
  
  # contract symmetric tensors using tensordot
  C = ST.tensordot(A, B, axes=[A_cont,B_cont])
  
  # contract corresponding dense numpy arrays using numpy tensordot
  C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])
  
  # compare results
  cont_error2 = LA.norm(C.toarray()-C_full)
  assert cont_error2 < tolerance
  
  ##############################################
  # Test: inner product
  A_ndim = 6 # number of indices on A
  B_ndim = 6 # number of indices on B
  cont_ndim = 6 # number of contracted indices
  ##############################################
  
  # create index of random qnums
  q_ind = ST.SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = np.array([False]*A_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  A = ST.SymTensor.rand(A_indices,A_arrows,A_divergence)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = np.array([True]*B_ndim, dtype=bool)
  B_divergence = SymIndex.rand(1, syms)
  B = ST.SymTensor.rand(B_indices,B_arrows,B_divergence)
  
  # select random indices to contract together
  A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
  B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]
  
  # contract symmetric tensors using tensordot
  C = ST.tensordot(A, B, axes=[A_cont,B_cont])
  
  # contract corresponding dense numpy arrays using numpy tensordot
  C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])
  
  # compare results
  cont_error3 = LA.norm(C-C_full.item())
  assert cont_error3 < tolerance
  
  ##############################################
  # Test: outer product
  A_ndim = 3 # number of indices on A
  B_ndim = 4 # number of indices on B
  cont_ndim = 0 # number of contracted indices
  ##############################################
  
  # create index of random qnums
  q_ind = ST.SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = np.array([False]*A_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  A = ST.SymTensor.rand(A_indices,A_arrows,A_divergence)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = np.array([True]*B_ndim, dtype=bool)
  B_divergence = SymIndex.rand(1, syms)
  B = ST.SymTensor.rand(B_indices,B_arrows,B_divergence)
  
  # select random indices to contract together
  A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
  B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]
  
  # contract symmetric tensors using tensordot
  C = ST.tensordot(A, B, axes=[A_cont,B_cont])
  
  # contract corresponding dense numpy arrays using numpy tensordot
  C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])
  
  # compare results
  cont_error4 = LA.norm(C.toarray()-C_full)
  assert cont_error4 < tolerance
  
  ##############################################
  # Test: all cases possible
  max_num_ind = 4
  A_ndim = np.random.randint(1,max_num_ind+1)
  B_ndim = np.random.randint(1,max_num_ind+1)
  cont_ndim = np.random.randint(0,min(A_ndim,B_ndim)+1)
  ##############################################
  
  # create index of random qnums
  q_ind = ST.SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = np.array([False]*A_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  A = ST.SymTensor.rand(A_indices,A_arrows,A_divergence)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = np.array([True]*B_ndim, dtype=bool)
  A_divergence = SymIndex.rand(1, syms)
  B = ST.SymTensor.rand(B_indices,B_arrows,B_divergence)
  
  # select random indices to contract together
  A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
  B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]
  
  # contract symmetric tensors using tensordot
  C = ST.tensordot(A, B, axes=[A_cont,B_cont])
  
  # contract corresponding dense numpy arrays using numpy tensordot
  C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])
  
  # compare results
  if C_full.size == 1:
    cont_error5 = LA.norm(C-C_full.item())
  else:
    cont_error5 = LA.norm(C.toarray()-C_full)
    
  assert cont_error5 < tolerance
  
  print("test: %3d, tests : %5.2e, %5.2e, %5.2e, %5.2e, %5.2e, %5.2e" 
        %(k,cont_error0,cont_error1,cont_error2,cont_error3,cont_error4,cont_error5))


