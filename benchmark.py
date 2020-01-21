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
Some benchmark problems for testing contraction times:
"""

"""
Example 1: U1xU1 with 6 indices of dim 16
"""
##############################################
# Example problem parameters:
syms = ['U1','U1'] # symmetries in use
A_ndim = 6 # number of indices on A
B_ndim = 6 # number of indices on B
cont_ndim = 3 # number of contracted indices
##############################################

# create index of qnums (here corresponding to a spin-1/2 fermion site)
q_spin = [0,1,-1,0] # spin quantum numbers
q_part = [-1,0,0,1] # particle quantum numbers
q_ind0 = ST.SymIndex.create([q_spin,q_part], syms) 
q_ind = q_ind0 @ q_ind0

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
t0 = time.time()
C = ST.tensordot(A, B, axes=[A_cont,B_cont])
t_U1U1_6 = time.time() - t0

"""
Example 2: U1 with 6 indices of dim 16
"""
##############################################
# Example problem parameters:
syms = ['U1'] # symmetries in use
A_ndim = 6 # number of indices on A
B_ndim = 6 # number of indices on B
cont_ndim = 3 # number of contracted indices
##############################################

# create index of qnums (here corresponding to a spin-1/2 fermion site)
q_part = [-1,0,0,1] # particle quantum numbers
q_ind0 = ST.SymIndex.create([q_part], syms) 
q_ind = q_ind0 @ q_ind0

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
t0 = time.time()
C = ST.tensordot(A, B, axes=[A_cont,B_cont])
t_U1_6 = time.time() - t0

"""
Example 3: U1xU1 with 12 indices of dim 4
"""
##############################################
# Example problem parameters:
syms = ['U1','U1'] # symmetries in use
A_ndim = 12 # number of indices on A
B_ndim = 12 # number of indices on B
cont_ndim = 6 # number of contracted indices
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
t0 = time.time()
C = ST.tensordot(A, B, axes=[A_cont,B_cont])
t_U1U1_12 = time.time() - t0

"""
Example 4: U1 with 12 indices of dim 4
"""
##############################################
# Example problem parameters:
syms = ['U1'] # symmetries in use
A_ndim = 12 # number of indices on A
B_ndim = 12 # number of indices on B
cont_ndim = 6 # number of contracted indices
##############################################

# create index of qnums (here corresponding to a spin-1/2 fermion site)
q_part = [-1,0,0,1] # particle quantum numbers
q_ind = ST.SymIndex.create([q_part], syms) 

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
t0 = time.time()
C = ST.tensordot(A, B, axes=[A_cont,B_cont])
t_U1_12 = time.time() - t0

"""
Example 5: U1xU1 with 14 indices of dim 4
"""
##############################################
# Example problem parameters:
syms = ['U1','U1'] # symmetries in use
A_ndim = 14 # number of indices on A
B_ndim = 14 # number of indices on B
cont_ndim = 7 # number of contracted indices
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
t0 = time.time()
C = ST.tensordot(A, B, axes=[A_cont,B_cont])
t_U1U1_14 = time.time() - t0

"""
Example 6: U1 with 12 indices of dim 4
"""
##############################################
# Example problem parameters:
syms = ['U1'] # symmetries in use
A_ndim = 14 # number of indices on A
B_ndim = 14 # number of indices on B
cont_ndim = 7 # number of contracted indices
##############################################

# create index of qnums (here corresponding to a spin-1/2 fermion site)
q_part = [-1,0,0,1] # particle quantum numbers
q_ind = ST.SymIndex.create([q_part], syms) 

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
t0 = time.time()
C = ST.tensordot(A, B, axes=[A_cont,B_cont])
t_U1_14 = time.time() - t0

##############################################


all_times = np.vstack([np.array([t_U1U1_6,t_U1U1_12,t_U1U1_14]), 
                       np.array([t_U1_6,t_U1_12,t_U1_14])]).T
print("Contraction times:")
print(all_times)










