#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	// Kernel for computing the inner state of neurons
	// The main equation for inner state of neurons X in time T is:
	// innerState[X, T] = (innerState[X, T-1] + A * imageInput + B * 1/N * sum(all edge inputs for X) / (A + B + Threshold),
	// where N is number of input neurons for neuron X and A/B are constants changeable in BrainSimulator

	__global__ void IF2ComputeStateKernel(
		int initState, // value of init state
		int refractoryState, // value of refractory state
		float refractory, // refractory
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

			neuronOutputs[id] = 0;

			if (innerStates[id] >= threshold){
				neuronOutputs[id] = innerStates[id];
				innerStates[id] = -65;
			}
			else {
				float totalInput = 0;

				for (int i = 0; i < count; i++) {
					totalInput += edgeInputs[i * count + id];
				}

				totalInput += imageInput[id];
				if (totalInput > 0){
					innerStates[id] += totalInput;
				}
			}

			/*int c1 = (innerStates[id] >= threshold);
			int c2 = (innerStates[id] < threshold);

			neuronOutputs[id] = c1 * innerStates[id];
			innerStates[id] = c2 * innerStates[id] + c1 * -130;

			int c3 = (innerStates[id] >= -65);
			int c4 = (innerStates[id] < -65);

			innerStates[id] = c3 * innerStates[id] + c4 * (innerStates[id] / 1.3f);

			if (c2 && c3){
			float totalInput = 0;

			for (int i = 0; i < count; i++) {
			totalInput += edgeInputs[i * count + id];
			}

			totalInput += imageInput[id];

			int c5 = (totalInput > 0);
			innerStates[id] += c5 * totalInput;
			}*/

			imageInput[id] = 0;
		}

	}

}