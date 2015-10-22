

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


extern "C"
{
	__device__ ct_tree* TREE = NULL;
	__device__ ctw_agent* AGENT = NULL;

	__device__  unsigned int X = 123456;//random generator: todo: put to better (shared?) memory & to better place in source codes
	__device__  unsigned int A = 1664525;
	__device__  unsigned int C = 1013904223;

	__global__ void AixiInitKernel(
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
		int q = 1 + 1;
	}

	__global__ void AixiPlayKernel(int reward, int observation, int* actionOutput)
	{
		int threadId = blockIdx.y*blockDim.x*gridDim.x
			+ blockIdx.x*blockDim.x
			+ threadIdx.x;

		model_update_percept(observation, reward);

//		int action = search();
		int action = 1;
		model_update_action(action);
		actionOutput[0] = action;

	}

	__global__ void AixiTestKernel()
	{
		int threadId = blockIdx.y*blockDim.x*gridDim.x
			+ blockIdx.x*blockDim.x
			+ threadIdx.x;

		model_update_percept(0, 0);
		model_update_action(1);
		model_update_percept(0, 1);
		model_update_action(0);
		model_update_percept(1, 1);

		ct_tree_undo* backup = backup_tree();
		model_update_action(0);
		model_update_percept(1, 0);
		model_update_action(1);
		model_update_percept(0, 1);

		model_revert(backup);

		//		int action = search();
		int action = 1+2;
//		model_update_action(action);

	}
}
