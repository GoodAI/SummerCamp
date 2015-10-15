#ifndef VARS_H
#define VARS_H

extern "C"
{


	#include "cuda_runtime.h"
	#include "ctw_context_tree.h"
	#include "ctw_context_tree_node.h"
	#include "CTW_agent.h"
	//Note: some vars will be shared across threads and some will be not. Beware of this distinction.


	/*extern __device__ int TREE_SIZE; //number of nodes in the tree


	extern __device__ int DEPTH; //depth of CT tree

	extern __device__ int ROOT_I; //index of root*/

	//extern __device__ ct_node* NODES;
	extern __device__ ct_tree* TREE;
	extern __device__ ctw_agent* AGENT;

	//n: to which memory put these values? todo: shared memory?
	extern __device__   unsigned int X;
	extern __device__   unsigned int A;
	extern __device__   unsigned int C;

	/*
	extern __device__ int FIRST_FREE_INDEX; // index of NODES that points to first element that is not in use.
	*/

	// FREE INDICES
	// HISTORY: not needed really. You only need some max. len back
	//CONTEXT

}

#endif