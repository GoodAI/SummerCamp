#ifndef CTW_CONTEXT_TREE_NODE_H
#define CTW_CONTEXT_TREE_NODE_H

//#include "ctw_context_tree.h"


typedef struct {
	float LogKt;
	float LogProbability;
	//indicies of childs corresponding to left (1) and right (0) child
	//(these  indices and not pointers because NODES array can be reallocated)
	int Children[2];
	int SymbolCount[2];
} ct_node;

__device__ int IsLeaf(int index);
__device__ float LogKtMultiplier(int index, int symbol);
__device__ int Visits(int index);

__device__ int FreeIfUnvisited(int XXX);

__device__ void updateLogProbability(int index);

__device__ void revert_node(int index, int symbol);

__device__ void update_node(int index, int symbol);

__device__ int subtree_size(int index);

__device__ int get_free_index();

__device__ int create_new_node();

__device__ void free_node(int index);

__device__ int rec(int num);


#endif