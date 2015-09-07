#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	// Kernel for computing the inner state of neurons
	// The main equation for neurons X in time T is:
	// innerState[X, T] = (innerState[X, T-1] + A * imageInput + B * 1/N * sum(all edge inputs for X) / (A + B + Threshold),
	// where N is number of input neurons for neuron X and A/B are constants changeable in BrainSimulator

	__global__ void LSMComputeStateKernel(
		float a, // A constant of the main equation
		float b, // B constant of the main equation
		float* edgeInputs, // edge inputs
		float* imageInput, // image inputs
		float* neuronOutputs, // output of neurons
		float* innerStates, // inner states of neurons
		float threshold, // threshold for sending of output
		float connectivity, // connectivity of the network
		int count // number of neurons
		)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			float totalInput = 0;

			for (int i = 0; i < count; i++) {
				totalInput += edgeInputs[i * count + id];
			}

			totalInput = (totalInput * b) / (connectivity * count);

			totalInput += imageInput[id] * a;
			imageInput[id] = 0;

			neuronOutputs[id] = 0;

			innerStates[id] += totalInput;

			innerStates[id] /= (a + b + threshold);

			int c1 = (innerStates[id] >= threshold);
			int c2 = (innerStates[id] < threshold);

			neuronOutputs[id] = innerStates[id] * c1;
			innerStates[id] = innerStates[id] * c2;
		}

	}

}