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
Set of tests for the transpose and reshape of tensors, and the interaction of
these with operations with multiplications and contractions. Asserts that the
difference between the symmetric and dense operations is less than the 
tolerance.
"""
number_of_tests = 100
tolerance = 1e-10

for t in range(number_of_tests):
  """
  Example 1: Generate a random tensor and do multiple reshapes and 
  transpositions.
  """
  ##############################################
  # Example problem parameters:
  chi = 4 # index dimensions
  A_ndim = 8 # number of indices on A
  cont_ndim = 3 # number of contracted indices
  ##############################################
  
  # choose symmetry (or set of symmetries) at random
  all_syms = ['U1','Z2','Z3'] 
  num_syms = np.random.randint(1,4)
  syms = [all_syms[np.random.randint(0,3)] for n in range(num_syms)]
  
  # generate some indices of random qnums
  A_indices = [SymIndex.rand(chi, syms) for n in range(A_ndim)]
  # generate some random arrows
  A_arrows = np.random.rand(A_ndim) > 0.5
  # generate random divergence
  A_divergence = SymIndex.rand(1, syms)
  A = SymTensor.rand(A_indices,A_arrows,A_divergence) 
  # export to dense tensor
  A_full = A.toarray()
  
  # do a bunch of reshapes and transpositions
  for k in range(10):
    # generate random reshape
    grouping = []
    remaining = A_ndim
    while remaining > 0:
      curr_group = np.random.randint(min(remaining,3))+1
      remaining = remaining - curr_group
      grouping.append(curr_group)
    
    cumul_grouping = np.insert(np.cumsum(grouping),0,0)
    new_shape = [np.prod(A.shape_trans[cumul_grouping[n]:cumul_grouping[n+1]]) for n in range(len(grouping))]
    
    # do reshape
    A = A.reshape(new_shape)
    A_full = A_full.reshape(new_shape)
    
    # generate random transposition
    new_ord = np.random.choice(A.ndim,A.ndim,replace=False)
    
    # do transposition
    A = A.transpose(new_ord)
    A_full = A_full.transpose(new_ord)
    
  # compare
  difference = LA.norm(A.toarray() - A_full)
  print("test-1.",t,", symmetric contraction error:", difference)
  assert difference < tolerance
  
  
  """
  Example 2: Generate a pair of random tensors, do multiple reshapes and 
  transpositions, then contract the tensors.
  """
  ##############################################
  # Example problem parameters:
  chi = 5 # index dimensions
  A_ndim = 8 # number of indices on A
  B_ndim = 5 # number of indices on B
  cont_ndim = 3 # number of contracted indices
  ##############################################
  
  # choose symmetry (or set of symmetries) at random
  all_syms = ['U1','Z2','Z3'] 
  num_syms = np.random.randint(1,4)
  syms = [all_syms[np.random.randint(0,3)] for n in range(num_syms)]
  
  # create index of random qnums
  q_ind = SymIndex.rand(chi, syms) 
  
  # create symmetric tensors with random elements
  A_indices = [q_ind]*A_ndim
  A_arrows = [False]*A_ndim # all arrows incoming
  A_divergence = SymIndex.rand(1, syms)
  A_perm = np.random.choice(A_ndim,A_ndim,replace=False)
  A = SymTensor.rand(A_indices,A_arrows,A_divergence).transpose(A_perm)
  
  B_indices = [q_ind]*B_ndim
  B_arrows = [True]*B_ndim # all arrows outgoing
  B_divergence = SymIndex.rand(1, syms)
  B_perm = np.random.choice(B_ndim,B_ndim,replace=False)
  B = SymTensor.rand(B_indices,B_arrows,A_divergence).transpose(B_perm)
  
  # generate random permutations
  A_perm_ord = np.random.choice(A_ndim,A_ndim,replace=False)
  B_perm_ord = np.random.choice(B_ndim,B_ndim,replace=False)
  
  # contract symmetric tensors (via permute, reshape, mat_mul)
  C = (A.transpose(A_perm_ord).reshape(chi**(A_ndim-cont_ndim),chi**cont_ndim) @ 
       B.transpose(B_perm_ord).reshape(chi**cont_ndim,chi**(B_ndim-cont_ndim))
       ).reshape([chi]*(A_ndim + B_ndim - 2*cont_ndim))
  
  # contract corresponding dense numpy arrays (via permute, reshape, mat_mul)
  C_full = (A.toarray().transpose(A_perm_ord).reshape(chi**(A_ndim-cont_ndim),chi**cont_ndim) @ 
            B.toarray().transpose(B_perm_ord).reshape(chi**cont_ndim,chi**(B_ndim-cont_ndim))
            ).reshape([chi]*(A_ndim + B_ndim - 2*cont_ndim))
  
  # compare results (SymTensor contraction versus dense tensor contraction)
  difference = LA.norm(C.toarray() - C_full)
  print("test-2.",t,", symmetric contraction error:", difference)
  assert difference < tolerance
  
  
  """
  Example 3: Contract a MERA tensor network using `ncon`, after doing 
  transpositions and rehsapes on the initial tensors.
  """
  
  ##############################################
  # Example problem parameters:
  chi0 = 3
  chi1 = 2
  chi00 = chi0**2 # bond dimension (outgoing from isometry)
  chi11 = chi1**2 # bond dimension (outgoing from disentangler)
  ##############################################
  
  # choose symmetry (or set of symmetries) at random
  all_syms = ['U1','Z2','Z3'] 
  num_syms = np.random.randint(1,4)
  syms = [all_syms[np.random.randint(0,3)] for n in range(num_syms)]
  
  # create index of random qnums
  q_ind0 = ST.SymIndex.rand(chi0, syms) 
  q_ind1 = ST.SymIndex.rand(chi1, syms) 
  q_ind00 = q_ind0 @ q_ind0 
  q_ind11 = q_ind1 @ q_ind1 
  
  # initialize tensors (`u` disentanglers, `w` isometries, ham, rho)
  u_indices = [q_ind00, q_ind0, q_ind0, q_ind11, q_ind1, q_ind1]
  u_arrows = [False, False, False, True, True, True]
  u = ST.SymTensor.rand(u_indices,u_arrows).transpose(2,1,0,3,5,4).reshape(chi00,chi00,chi11,chi11)
  
  w_indices = [q_ind11, q_ind1, q_ind1, q_ind00]
  w_arrows = [False, False, False, True]
  w = ST.SymTensor.rand(w_indices,w_arrows).transpose(0,2,1,3).reshape(chi11,chi11,chi00)
  
  ham_indices = [q_ind00, q_ind00, q_ind00, q_ind00, q_ind00, q_ind00]
  ham_arrows = [False, False, False, True, True, True]
  ham = ST.SymTensor.rand(ham_indices,ham_arrows)
  
  rho_indices = [q_ind00, q_ind00, q_ind00, q_ind00, q_ind00, q_ind00]
  rho_arrows = [True, True, True, False, False, False]
  rho = ST.SymTensor.rand(rho_indices,rho_arrows)
  
  # contract a network using the `ncon` routine (from a 1d MERA algorithm) 
  tensors = [u,w,w,w,ham,u.conj(),u.conj(),w.conj(),w.conj(),w.conj(),rho.conj()]
  connects = [[4,6,15,14],[-4,15,19],[8,-3,18],[14,13,20],[3,5,7,-2,4,6],[-1,3,9,10],
              [5,7,11,12],[10,11,22],[8,9,21],[12,13,23],[18,19,20,21,22,23]] 
  cont_order = [4, 6, 13, 5, 7, 8, 18, 21, 20, 23, 22, 9, 10, 14, 3, 11, 12, 15, 19] 
  u_env = ST.ncon(tensors, connects, cont_order)
  
  # contract corresponding network of dense np.ndarry using `ncon` routine
  uf = u.toarray()
  wf = w.toarray()
  hamf = ham.toarray()
  rhof = rho.toarray()
  tensors = [uf,wf,wf,wf,hamf,uf.conj(),uf.conj(),wf.conj(),wf.conj(),wf.conj(),rhof.conj()]
  connects = [[4,6,15,14],[-4,15,19],[8,-3,18],[14,13,20],[3,5,7,-2,4,6],[-1,3,9,10],
              [5,7,11,12],[10,11,22],[8,9,21],[12,13,23],[18,19,20,21,22,23]] 
  cont_order = [4, 6, 13, 5, 7, 8, 18, 21, 20, 23, 22, 9, 10, 14, 3, 11, 12, 15, 19] 
  u_envf = ST.ncon(tensors, connects, cont_order)
  
  # compare results (SymTensor contraction versus dense tensor contraction)
  difference = LA.norm(u_env.toarray()-u_envf) / max(1e-10, LA.norm(u_envf))
  print("test-3.",t,", symmetric contraction error:", difference)
  assert difference < tolerance

