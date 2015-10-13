#include "ctw_context_tree.h"
#include "cuda_runtime.h"
#include <math.h>
#include "vars.h"
#include <stdlib.h>
#include <string.h>

__device__ int IsLeaf(int index){
	ct_node* me = &(TREE->nodes[index]);
	return (me->Children[0] == -1) && (me->Children[1] == -1);
}

//SetSymbolCount : not really needed

__device__ float LogKtMultiplier(int index, int symbol)
{
	ct_node* node = &(TREE->nodes[index]);

	float denominator = node->SymbolCount[0] + node->SymbolCount[1]+ 1;
	float numerator = node->SymbolCount[symbol] + 0.5;

	return log(numerator / denominator);
}

__device__ int Visits(int index)
{
	return TREE->nodes[index].SymbolCount[0] + TREE->nodes[index].SymbolCount[1];
}


__device__ int FreeIfUnvisited(int XXX); //TODO

__device__ int rec(int num){
	if (num == 1){
		return 1;
	}
	else{
		return 1 + rec(num - 1);
	}
}



__device__ void updateLogProbability(int index)
{
	ct_node* me = &(TREE->nodes[index]);
	if (IsLeaf(index))
	{
		me->LogProbability = me->LogKt;
	}
	else
	{
		float logChildProbability = 0;
		
		for (int i = 0; i < 2; i++){
			int child_i = me->Children[i];
			if (child_i != -1){
				logChildProbability += TREE->nodes[child_i].LogProbability;//HERE it do an error!
			}
		}
		
		//for better numerical results
		// (is this really helpful?)
		float a = fmax(me->LogKt, logChildProbability);
		float b = fmin(me->LogKt, logChildProbability);

		//todo: is it fast enough to compute Math.Log(0.5) every time. Joel has it cached.
		me->LogProbability = log(0.5) + a + log1p(exp(b - a));
	}
}

__device__ void revert_node(int index, int symbol)
{
	ct_node * me = &(TREE->nodes[index]);
	if (me->SymbolCount[symbol] > 1)
	{
		me->SymbolCount[symbol]--;
	}
	else
	{//note: this branch will (I think) never be used
		me->SymbolCount[symbol] = 0;
	}

	//todo: test removal
	//potentially redundant child for removal
	int redundant_child_i = me->Children[symbol];
	if ((redundant_child_i != -1) && Visits(redundant_child_i) == 0){
		//TODO: look at pycode
		//TODO: self.tree.tree_size -= redundant_child.size()
		me->Children[symbol] = -1;
	}
	me->LogKt -= LogKtMultiplier(index, symbol);
	updateLogProbability(index);
}


__device__ void update_node(int index, int symbol)
{
	ct_node* me = &(TREE->nodes[index]);
	me->LogKt += LogKtMultiplier(index, symbol);
	updateLogProbability(index);
	me->SymbolCount[symbol]++;
}


__device__ int subtree_size(int index) {
	int count = 1;
	for (int i = 0; i < 2; i++){
		if (TREE->nodes[index].Children[i] != -1){
			count += subtree_size(TREE->nodes[index].Children[i]);//TODO: can we do reasonable recursion on cuda?
		}
	}
	return count;
}


__device__ int get_free_index() {

	if (TREE->free_array_first_free > 0){
		//if there are free indices in queue, use last one
		return TREE->free_array_first_free--;
	}

	if (TREE->first_free_index >= TREE->nodes_array_size) {
		//resize array for nodes
		int old_size = TREE->nodes_array_size;
		TREE->nodes_array_size *= 2;
		ct_node* new_nodes = (ct_node*)malloc(sizeof(ct_node)*TREE->nodes_array_size);
		memcpy(new_nodes, TREE->nodes, sizeof(ct_node)*old_size);
		free(TREE->nodes);
		TREE->nodes = new_nodes;
	}

	return TREE->first_free_index++;
}


__device__ int create_new_node(){
	int index = get_free_index();
		
	ct_node* node = &(TREE->nodes[index]);
	node->LogProbability = 0.0;
	node->LogKt = 0.0;
	node->SymbolCount[0] = 0;
	node->SymbolCount[1] = 0;
	node->Children[0] = -1;
	node->Children[1] = -1;

	TREE->size++;

	return index;
}

__device__ void free_node(int index){
	//nodes[index] has to be used - you cannot free already freed node
	//low-todo: resize nodes array(?)
	
	if (TREE->first_free_index >= TREE->free_array_array_size){
		//resize
		int old_size = TREE->free_array_array_size;
		TREE->free_array_array_size *= 2;
		int* new_array = (int*)malloc(sizeof(int)*TREE->free_array_array_size);
		memcpy(new_array, TREE->free_array, sizeof(int)*old_size);
		free(TREE->free_array);
		TREE->free_array = new_array;
	}

	TREE->free_array[TREE->free_array_first_free] = index;

	TREE->size--;//not sure here: in fast version I have not it (maybe just forgotten?)
	
	TREE->free_array_first_free++;
}


