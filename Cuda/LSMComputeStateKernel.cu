#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

extern "C"{

	__global__ void LSMComputeStateKernel(float a, float b, float* edgeInputs, float* imageInput, float* neuronOutputs, float* innerStates, float threshold, float connectivity, int count){

		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < count){
			float totalInput = 0;
			int totalCount = 0;

			for (int i = 0; i < count; i++) {
				float temp = edgeInputs[i * count + id];
				totalInput += temp;

				int a1 = (temp > 0);
				totalCount = totalCount + a1;
			}

			totalInput = (totalInput * b) / (connectivity * count);

			totalInput += imageInput[id] * a;
			imageInput[id] = 0;

			neuronOutputs[id] = 0;

			innerStates[id] += totalInput;

			innerStates[id] /= (a + b + threshold);

			int a3 = (innerStates[id] >= threshold);
			int b3 = (innerStates[id] < threshold);

			neuronOutputs[id] = innerStates[id] * a3;
			innerStates[id] = innerStates[id] * b3;
		}

	}

}