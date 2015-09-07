#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	// Kernel for calculation the I/O of edges between neurons

	__global__ void LSMComputeEdgesKernel(
		float* edgeInputs, // edge I/O
		float* weights, // weights of edges
		float* neuronOutputs, // output of neuron
		int spikes, // boolean, whether there are spikes or not
		float spikeSize, // size of a spike
		int neurons, // number of neurons
		int count // number of edges
		)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			int i = (id - id%neurons) / neurons;
			edgeInputs[id] = weights[id] * (spikes * (neuronOutputs[i] > 0) * spikeSize + (!spikes) * neuronOutputs[i]);
		}

	}

}