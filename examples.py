
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
import sym_tensor_toolbox as ST

"""
Some examples demonstrating the use of the toolbox (in creating, manipulating 
and contracting SymTensors).
"""

"""
Example 1: Contract a pair of tensors using reshape, transpose, mat_mul. 
Tensors are assumed symmetric w.r.t a single 'U1' symmetry (e.g. particle 
conservation).
"""
##############################################
# Example problem parameters:
chi = 5 # index dimensions
syms = ['U1'] # symmetry in use
A_ndim = 8 # number of indices on A
B_ndim = 5 # number of indices on B
cont_ndim = 3 # number of contracted indices
##############################################

# create index of random qnums
q_ind = ST.SymIndex.rand(chi, syms) 

# create symmetric tensors with random elements
A_indices = [q_ind]*A_ndim
A_arrows = np.array([False]*A_ndim, dtype=bool)
A = ST.SymTensor.rand(A_indices,A_arrows)

B_indices = [q_ind]*B_ndim
B_arrows = np.array([True]*B_ndim, dtype=bool)
B = ST.SymTensor.rand(B_indices,B_arrows)

# generate random permutations
A_perm_ord = np.argsort(np.random.rand(A_ndim))
B_perm_ord = np.argsort(np.random.rand(B_ndim))

# contract symmetric tensors (via permute, reshape, mat_mul)
C = (A.transpose(A_perm_ord).reshape(chi**(A_ndim-cont_ndim),chi**cont_ndim) @ 
     B.transpose(B_perm_ord).reshape(chi**cont_ndim,chi**(B_ndim-cont_ndim))
     ).reshape([chi]*(A_ndim + B_ndim - 2*cont_ndim))

# contract corresponding dense numpy arrays (via permute, reshape, mat_mul)
C_full = (A.toarray().transpose(A_perm_ord).reshape(chi**(A_ndim-cont_ndim),chi**cont_ndim) @ 
          B.toarray().transpose(B_perm_ord).reshape(chi**cont_ndim,chi**(B_ndim-cont_ndim))
          ).reshape([chi]*(A_ndim + B_ndim - 2*cont_ndim))

# compare results
print("symmetric contraction error:", LA.norm(C.toarray()-C_full))


"""
Example 2: Contract a pair of symmetric tensors using tensordot. Tensors are 
assumed symmetric w.r.t two 'U1' symmetries (e.g. particle and spin 
conservation).
"""
##############################################
# Example problem parameters:
syms = ['U1','U1'] # symmetries in use
A_ndim = 6 # number of indices on A
B_ndim = 8 # number of indices on B
cont_ndim = 4 # number of contracted indices
##############################################

# create index of qnums (here corresponding to a spin-1/2 fermion site)
q_spin = [0,1,-1,0] # spin quantum numbers
q_part = [-1,0,0,1] # particle quantum numbers
q_ind = ST.SymIndex.create([q_spin,q_part], syms) 

# create symmetric tensors with random elements
A_indices = [q_ind]*A_ndim
A_arrows = np.array([False]*A_ndim, dtype=bool)
A = ST.SymTensor.rand(A_indices,A_arrows)

B_indices = [q_ind]*B_ndim
B_arrows = np.array([True]*B_ndim, dtype=bool)
B = ST.SymTensor.rand(B_indices,B_arrows)

# select random indices to contract together
A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]

# contract symmetric tensors using tensordot
C = ST.tensordot(A, B, axes=[A_cont,B_cont])

# contract corresponding dense numpy arrays using numpy tensordot
C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])

# compare results
print("symmetric contraction error:", LA.norm(C.toarray()-C_full))














