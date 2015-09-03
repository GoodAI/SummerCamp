#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	// Kernel for calculation of external output of neurons of LSM

	__global__ void LSMOutputKernel(
		float* states, // inner states of neurons
		float* outputs, // outputs of neurons
		float* nodeOutput, // output of LSM
		int count // number of neurons
		)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			float innerState = states[id];
			float output = outputs[id];
			nodeOutput[id] = fmaxf(innerState, output);
		}

	}

}