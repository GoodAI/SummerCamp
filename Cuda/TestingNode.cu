
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "ctw_context_tree.h"
#include "ctw_context_tree_node.h"
#include "vars.h"
#include "CTW_agent.h"

#include "utils.cu"
#include "ctw_context_tree.cu"
#include "ctw_context_tree_node.cu"
#include "vars.cu"
#include "CTW_agent.cu"

//#include "vars.h"
//#include "ctw_context_tree.h"


extern "C"
{

	__device__ ct_tree* TREE = NULL;
	__device__ ctw_agent* AGENT = NULL;


	__global__ void TestAll(int* testInts, 
		float* testFloats, 
		int intsNum, 
		int floatNum,
		
		int depth,
		int mc_simulations,
		int horizon,
		int maximum_action,
		int maximum_reward,
		int maximum_observation
		)
	{
		int threadId = blockIdx.y*blockDim.x*gridDim.x
			+ blockIdx.x*blockDim.x
			+ threadIdx.x;
		
		create_agent(depth, mc_simulations, horizon, maximum_action, maximum_reward, maximum_observation);


		int ints2[] = { 0, 1, 0, 1, 0, 0, 1, 1, 1 };

		for (int i = 0; i < 9; i++){
			update_tree_history(ints2[i]);
		}
		
		update_tree(9885, 5);

		ct_tree_undo* backup = backup_tree();
		
		update_tree(321567, 9);
		
		model_revert(backup);

		return;
		//global var tree is set inside:
		create_ct_tree(3);
		//TREE = tree;

		ct_tree* tree = TREE;
		/*
		ct_node* root = &(TREE->nodes[TREE->root_i]);
		
		update_node(tree->root_i, 0);
		update_node(tree->root_i, 1);
		update_node(tree->root_i, 0);
		update_node(tree->root_i, 0);
		update_node(tree->root_i, 0);

		
		testFloats[0] = root->LogKt;
		testFloats[1] = root->LogProbability;
		
		testInts[1] = root->SymbolCount[0];
		testInts[2] = root->SymbolCount[1];
		
		testFloats[2] = LogKtMultiplier(tree->root_i, 0);
		
		update_node(tree->root_i, 1);
		revert_node(tree->root_i, 1);

		testFloats[3] = root->LogKt;
		testFloats[4] = root->LogProbability;
		
		int node_i = create_new_node();
		root->Children[1] = node_i;

		int c = Visits(tree->root_i);
		int d = subtree_size(tree->root_i);
		int e = IsLeaf(tree->root_i);
		int f = IsLeaf(node_i);
		int g = rec(100);*/


		int n1 = create_new_node();
		int n2 = create_new_node();
		int n3 = create_new_node();
		int n4 = create_new_node();
		

		TREE->nodes[TREE->root_i].Children[1] = n1;
		TREE->nodes[TREE->root_i].Children[0] = n2;
		TREE->nodes[n2].Children[0] = n4;
		TREE->nodes[n2].Children[1] = n3;

		
		int ints[] = { 0, 1,0, 1, 0, 0, 1, 1,1 };
		
		for (int i = 0; i < sizeof(ints) / sizeof(int); i++){
			update_tree_history(ints[i]);
		}

		//update_tree(ints, sizeof(ints)/sizeof(int));

		revert_tree(2);

		for (int i = 0; i < 20; i++){
			create_new_node();
		}

		free_node(2);
		free_node(4);
		int a = create_new_node();
		int b = a + 100;

	}
}
