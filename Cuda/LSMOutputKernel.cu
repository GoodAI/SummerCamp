#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	__global__ void LSMOutputKernel(float* states, float* outputs, float* nodeOutput, float threshold, int count){

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