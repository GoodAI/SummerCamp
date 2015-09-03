//Includes for IntelliSense (WHAT is this for???)
#define _SIZE_T_DEFINED


#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

extern "C"{

	__global__ void LSMParseInputKernel(float* probabilities, float* input, int* imageOutput, float* imageInput, int spikes, float spikeSize, int count){

		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			int target = imageOutput[id];
			imageInput[target] = spikes * (input[id] > probabilities[id]) * spikeSize + (!spikes) * input[id];
		}
	}

}