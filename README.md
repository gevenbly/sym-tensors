-----------------------------------------------------------------------------
Glen's symmetric tensor toolbox: a toolbox for the efficient representation and manipulation of tensors that are invariant under the action of an abelian group.
-----------------------------------------------------------------------------

Core design features:
* designed to be fast even for tensors with many indices and/or many different qnum sectors per index.
* designed such that SymTensor objects behave in a similar manner to numpy ndarrays (under transpose, rehape, matmul, tensordot etc), such that tensor network codes require minimal modifications when switching between the two.
  
Techinical Notes:
* uses element-wise encoding strategy (with row-major order) to store only the structually non-zero tensor elements in a dense array (or datavector).
* built in support for common symmetries ('U1', 'Z2', 'Z3'), although the toolbox is designed to allow the user to add any custom symmetry (or custom representation). To add a custom symmetry the user must supply the fusion rule in 'fuse_qnums' and the duality transform in 'SymIndex.dual'.
* supports tensors which are simultaneously symmetric under multiple groups (e.g. tensors symmetric under both 'U1' and 'Z2' with distinct index qnums for each symmetry group).
* uses np.int16 to represent both quantum numbers and index labels (which assumes that no index will have more than 32767 distinct symmetry sectors).
* uses np.uint32 to represent the positions that non-zero tensor elements would occupy within the dense array corresponding to a SymTensor (which assumes the size of the dense array would not exceed 2^32 elements)
