
#include "ctw_context_tree.h"
#include "cuda_runtime.h"
#include "vars.h"
#include "utils.h"
#include "ctw_context_tree_node.h"
#include <stdlib.h>
#include <assert.h>
#include <list>
#include <math.h>


__device__ ct_tree* create_ct_tree(int depth){
	//make context
	//make history

	TREE = (ct_tree*)malloc(1 * sizeof(ct_tree));


	TREE->nodes_array_size = 4;//todo: set to bigger, reasonable size
	TREE->nodes = (ct_node*)malloc(sizeof(ct_node) * TREE->nodes_array_size);
	TREE->first_free_index = 0;
	TREE->depth = depth;
	TREE->size = 0;

	TREE->context = (int*)malloc(sizeof(int)*(TREE->depth+1));

	TREE->free_array_array_size = 4;//todo:set to bigger
	TREE->free_array_first_free = 0;

	//TODO! Compute how much of history do we need (we do not need to remember everything, just enough for backtracking).
	TREE->history_array_size = depth+10;
	//we need depth+how much into future we are looking
	TREE->history_count = 0;
	TREE->history = (int*)malloc(sizeof(int) * TREE->history_array_size);
	
	//init is not needed.
	TREE->free_array = (int*)malloc(sizeof(int) * TREE->free_array_array_size);

	TREE->root_i = create_new_node();
	
	return TREE;
}

__device__ ct_tree_undo* backup_tree(){
	ct_tree_undo* undo_tree = (ct_tree_undo*)malloc(sizeof(ct_tree_undo));
	undo_tree->age = AGENT->age;
	undo_tree->total_reward = AGENT->total_reward;
	undo_tree->last_update = AGENT->last_update;
	undo_tree->history_count = TREE->history_count;
	return undo_tree;
}

__device__ int history_position(){
	return (TREE->history_count-1) % TREE->history_array_size;
}


__device__ void update_tree_history_symbols(int symbols, int symbol_count){
	//copy last /symbol_count/ bits of symbols into history. (first bit copied is on the left)
	for (int i = symbol_count-1; i >=0; i--){
		update_tree_history((symbols & (1<<i)) ? 1:0);
	}
}

__device__ void update_tree_history(int symbol){
	TREE->history_count++;
	TREE->history[history_position()] = symbol;
}

__device__ void revert_tree_history(int symbolCount){
	//there are no checks here: ie: you can only revert only until you start getting non-sense data.
	TREE->history_count -= symbolCount;
}

__device__ void update_context(){
	assert(TREE->history_count >= TREE->depth);
	TREE->context[0] = TREE->root_i;

	int node_i = TREE->root_i;
	int history_position = TREE->history_count-1;
	for (int i = 1; i <= TREE->depth; i++){
		// are we sure this is working?
		int symbol = TREE->history[history_position%TREE->history_array_size];
		history_position--;
		if (TREE->nodes[node_i].Children[symbol] != -1){
			node_i = TREE->nodes[node_i].Children[symbol];
		}
		else{
			int new_node_i = create_new_node();
			TREE->nodes[node_i].Children[symbol] = new_node_i;
			node_i = new_node_i;
		}
		TREE->context[i] = node_i;
	}
}

//__device__ void update_tree(int* symbol_list, int symbol_count) {
__device__ void update_tree(int symbols, int symbol_count) {
	//todo: add some kind of assert, like in update_context that will assert that we have enough of history - without history you cannot call update_tree
	for (int i = symbol_count-1; i >=0; i--){
		int symbol = (symbols & (1<<i)) ? 1:0;
		update_context();
		for (int j = TREE->depth - 1; j >= 0; j--){
			update_node(TREE->context[j], symbol);
		}
		update_tree_history(symbol);
	}
}


__device__ float predict_symbol(int symbol) {
	// -> predict_symbol?
	
	if (TREE->history_count + 1 <= TREE->depth) {
		return 0.5; 
	}

	float prob_history = TREE->nodes[TREE->root_i].LogProbability;

	update_tree(symbol, 1);

	float prob_sequence = TREE->nodes[TREE->root_i].LogProbability;
	revert_tree(1);
	return exp(prob_sequence - prob_history);
}


__device__ void revert_tree(int symbol_count)
{
	//change: symbol_count is no longer 1 by default
	for (int i = 0; i < symbol_count; i++) {
		if (TREE->history_count <= 0) {
			return;
		}
		int position = TREE->history_count - 1;
		int pos_norm = position % TREE->history_array_size;
		int symbol = TREE->history[pos_norm];
		TREE->history_count--;

		if (TREE->history_count >= TREE->depth) {
			update_context();

			for (int j = TREE->depth - 1; j >= 0; j--) {
				revert_node(TREE->context[j], symbol);
			}
		}
	}
}

__device__ int get_model_size() {
	return TREE->size;
}

__device__ int generate_random_symbols_and_update(int symbol_count) {

	int symbol_list = 0;

	for (int i = symbol_count; i >=0; i--) {
		int symbol;
		int symbol_to_predict = 1;
		
		if (rand_float() < predict_symbol(symbol_to_predict)){
			symbol = 1;
		}
		else{
			symbol = 0;
		}

		symbol_list = symbol_list | (symbol << i);

//		update_tree(symbol, 1);

		// here is change from original - I am not sure why it is needed now it may not be needed since gen rand symbol is not called when we do not have enough of history
		
		if (TREE->history_count >= TREE->depth){
			update_tree(symbol, 1);
		}
		else{
			update_tree_history(symbol);
		}
	}
	return symbol_list;
}