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

	// Kernel to parse the input of LSM
	// In spiking environment the pixel will spike if its value is bigger than the random generated value
	// In non-spiking environment the pixel spikes its value

	__global__ void LSMParseInputKernel(
		float* input, // output of an image
		int* imageTargets, // target neurons of an image input
		float* imageInput, // image input for neurons
		int spikes, // spiking with probability/non-spiking of float input value
		float spikeSize, // size of a spike
		int count // size of an input
		)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			int target = imageTargets[id];
			imageInput[target] = spikes * spikeSize * input[id] + (!spikes) * input[id];
		}
	}

}