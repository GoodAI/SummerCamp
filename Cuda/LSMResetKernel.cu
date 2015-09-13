#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	// Kernel for computing the inner state of neurons
	// The main equation for inner state of neurons X in time T is:
	// innerState[X, T] = (innerState[X, T-1] + A * imageInput + B * 1/N * sum(all edge inputs for X) / (A + B + Threshold),
	// where N is number of input neurons for neuron X and A/B are constants changeable in BrainSimulator

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