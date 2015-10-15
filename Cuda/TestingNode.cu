
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "ctw_context_tree.h"
#include "ctw_context_tree_node.h"
#include "vars.h"
#include "CTW_agent.h"
#include "mc_tree.h"

#include "utils.cu"
#include "ctw_context_tree.cu"
#include "ctw_context_tree_node.cu"
#include "vars.cu"
#include "CTW_agent.cu"
#include "mc_tree.cu"


//#include "vars.h"
//#include "ctw_context_tree.h"


extern "C"
{

	__device__ ct_tree* TREE = NULL;
	__device__ ctw_agent* AGENT = NULL;

	__device__  unsigned int X = 123456;//random generator: todo: put to better (shared?) memory
	__device__  unsigned int A = 1664525;
	__device__  unsigned int C = 1013904223;



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



		for (int i = 0; i < 20; i++){
			model_update_percept(rand_int_range(0, 2), rand_int_range(0, 10));
			model_update_action(rand_int_range(0, 2));
		}
		model_update_percept(rand_int_range(0, 2), rand_int_range(0, 10));

		int action = search();
		
	}
}
