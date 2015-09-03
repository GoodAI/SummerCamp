#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	__global__ void LSMComputeEdgesKernel(float* edgeInputs, float* weights, float* neuronOutputs, int spikes, float spikeSize, int neurons, int count){
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			int i = (id - id%neurons) / neurons;
			edgeInputs[id] = weights[id] * (spikes * (neuronOutputs[i] > 0) * spikeSize + (!spikes) * neuronOutputs[i]);
		}

	}

}