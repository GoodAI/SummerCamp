#ifndef CTW_CONTEXT_TREE_H
#define CTW_CONTEXT_TREE_H

#include "cuda_runtime.h"

#include "ctw_context_tree_node.h"

typedef struct {
	int age;//? age is not part of ct_tree, (it is in agent)
	float total_reward; //same here
	int history_count;
	//history?
	int last_update;
} ct_tree_undo;

typedef struct {
	int root_i;
	//context;
	int depth;

	int* history;//todo: change type to some space-efficient
	int history_array_size;
	//number of bits pushed into history so far. usually is bigger than array size
	// therefore we will be cyclically overwriting old history entries
	int history_count;
	//position of last entry in history is history_count%history_array_size
	

	//size of context is tree_depth
	// (what about playout?)
	//it contains indicies of nodes that got used
	int* context;

	int size;

	int first_free_index;

	int nodes_array_size;
	ct_node* nodes;

	int free_array_array_size;
	int free_array_first_free;
	int* free_array;

} ct_tree;	//refact: rename: ct_tree is abbr from "context tree tree"

__device__ ct_tree* create_ct_tree(int depth);

__device__ void update_tree_history_symbols(int symbols, int symbol_count);

__device__ void update_tree_history(int symbol);

__device__ void revert_tree_history(int symbolCount);

__device__ void update_context();

__device__ void update_tree(int symbols, int symbol_count);

__device__ float predict_symbol(int symbol);

__device__ void revert_tree(int symbol_count);

__device__ int get_model_size();

__device__ int generate_random_symbols_and_update(int symbol_count);

__device__ ct_tree_undo* backup_tree();

__device__ int history_position();


#endif