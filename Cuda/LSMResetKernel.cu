#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	// Kernel for reseting the network to init state
	// Inner states of neurons and both external and internal inputs needs to be reseted

	__global__ void LSMResetKernel(
		int initState, // value of init state
		float* innerStates, // inner states of neurons
		float* edgeInputs, // edge inputs
		float* imageInput, // image inputs
		int count // number of neurons
		)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			innerStates[id] = initState;

			imageInput[id] = 0;

			for (int i = 0; i < count; i++){
				edgeInputs[id * count + i] = 0;
			}
		}

	}

}