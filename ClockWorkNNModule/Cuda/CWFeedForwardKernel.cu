//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

#include "../NeuralNetwork/Activation/ActivationFunction.cu"


extern "C"
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_HIDDEN_UNITS;
	__constant__ int D_OUTPUT_UNITS;
	__constant__ int D_NEURONS_PER_GROUP;
	__constant__ int D_NEURON_GROUPS;
	__constant__ ActivationFunctionEnum D_ACTIVATION_FUNCTION;

	__global__ void CWFeedforwardHiddenKernel(
		float *input, 
		float *hiddenActivations,
		float *hiddenActivationDerivatives, 
		float *inputWeights, 
		float *recurrentWeights,
		int *activeGroups
		)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		extern __shared__ float activeGroupsShared[];

		if (unitId < D_NEURON_GROUPS)
		{
			activeGroupsShared[unitId] = activeGroups[unitId];
		}
		__syncthreads();

		int groupID = unitId / D_NEURONS_PER_GROUP;

		if (unitId < D_HIDDEN_UNITS && activeGroupsShared[groupID] == 1)
		{
			int weightId = unitId * D_INPUT_UNITS;

			float weightedSum = 0;
			for (int i = 0; i < D_INPUT_UNITS; i++)
			{
				weightedSum += inputWeights[weightId] * input[i];
				weightId++;
			}

			weightId = unitId * D_HIDDEN_UNITS;

			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				weightedSum += recurrentWeights[weightId] * hiddenActivations[i];
				weightId++;
			}
			
			__syncthreads();

			hiddenActivations[unitId] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
			hiddenActivationDerivatives[unitId] = EvaluateDerivative(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}

	__global__ void CWFeedforwardOutputKernel(float *hiddenActivations, float *outputActivations, float *outputActivationDerivatives, float *outputWeights)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (unitId < D_OUTPUT_UNITS)
		{
			
			int weightId = unitId * D_HIDDEN_UNITS;

			float weightedSum = 0;
			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				weightedSum += outputWeights[weightId] * hiddenActivations[i];
				weightId++;
			}

			outputActivations[unitId] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
			
			outputActivationDerivatives[unitId] = EvaluateDerivative(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}
}
