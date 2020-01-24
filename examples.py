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
A_arrows = [False]*A_ndim # all arrows incoming
A = ST.SymTensor.rand(A_indices,A_arrows) 

B_indices = [q_ind]*B_ndim
B_arrows = [True]*B_ndim # all arrows outgoing
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

# compare results (SymTensor contraction versus dense tensor contraction)
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
A_arrows = [False]*A_ndim # all arrows incoming
A = ST.SymTensor.rand(A_indices,A_arrows)

B_indices = [q_ind]*B_ndim
B_arrows = [True]*B_ndim # all arrows outgoing
B = ST.SymTensor.rand(B_indices,B_arrows)

# select random indices to contract together
A_cont = np.argsort(np.random.rand(A_ndim))[:cont_ndim]
B_cont = np.argsort(np.random.rand(B_ndim))[:cont_ndim]

# contract symmetric tensors using tensordot
C = ST.tensordot(A, B, axes=[A_cont,B_cont])

# contract corresponding dense numpy arrays using numpy tensordot
C_full = np.tensordot(A.toarray(), B.toarray(), axes=[A_cont,B_cont])

# compare results (SymTensor contraction versus dense tensor contraction)
print("symmetric contraction error:", LA.norm(C.toarray()-C_full))


"""
Example 3: Contract a tensor network using `ncon`. Tensors are assumed 
symmetric w.r.t to a `U1` symmetry and a `Z2` symmetry. The network 
corresponds to a tensor environment from a MERA.
"""
##############################################
# Example problem parameters:
syms = ['U1','Z2'] # symmetries in use
chi0 = 8 # bond dimension (outgoing from isometry)
chi1 = 6 # bond dimension (outgoing from disentangler)
##############################################

# create index of random qnums
q_ind0 = ST.SymIndex.rand(chi0, syms) 
q_ind1 = ST.SymIndex.rand(chi1, syms) 

# initialize tensors (`u` disentanglers, `w` isometries, ham, rho)
u_indices = [q_ind0, q_ind0, q_ind1, q_ind1]
u_arrows = [False, False, True, True]
u = ST.SymTensor.rand(u_indices,u_arrows)

w_indices = [q_ind1, q_ind1, q_ind0]
w_arrows = [False, False, True]
w = ST.SymTensor.rand(w_indices,w_arrows)

ham_indices = [q_ind0, q_ind0, q_ind0, q_ind0, q_ind0, q_ind0]
ham_arrows = [False, False, False, True, True, True]
ham = ST.SymTensor.rand(ham_indices,ham_arrows)

rho_indices = [q_ind0, q_ind0, q_ind0, q_ind0, q_ind0, q_ind0]
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
print("symmetric contraction error:", LA.norm(u_env.toarray()-u_envf))











