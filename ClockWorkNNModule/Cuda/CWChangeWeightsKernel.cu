//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>



extern "C"
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_HIDDEN_UNITS;
	__constant__ int D_OUTPUT_UNITS;
	__constant__ int D_NEURONS_PER_GROUP;
	__constant__ int D_NEURON_GROUPS;

	__global__ void CWChangeInputWeightsKernel(
		float *inputWeights,
		float *inputWeightDeltas,
		float *outputWeights,
		float *outputDeltas,
		float *inputWeightRTRLDerivatives,

		float trainingRate,
		float momentum,

		int *activeGroups,
		int contextByActivations
		)
	{
		int weightId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int unitID = weightId / D_INPUT_UNITS;
		int groupID = unitID / D_NEURONS_PER_GROUP;

		extern __shared__ float activeGroupsShared[];

		if (weightId < D_NEURON_GROUPS)
		{
			activeGroupsShared[weightId] = activeGroups[weightId];
		}
		__syncthreads();

		if (weightId < D_HIDDEN_UNITS * D_INPUT_UNITS 
			&& (contextByActivations || activeGroupsShared[groupID]))
		{
			float gradient = 0;
			for (int i = 0; i < D_OUTPUT_UNITS; i++)
			{
				float sum = 0;
				for (int j = 0; j < D_HIDDEN_UNITS; j++)
				{
					sum += outputWeights[i * D_HIDDEN_UNITS + j] * inputWeightRTRLDerivatives[j * D_HIDDEN_UNITS * D_INPUT_UNITS + weightId];
				}

				gradient += outputDeltas[i] * sum;
			}

			__syncthreads();

			float weightDelta = trainingRate * gradient + momentum * inputWeightDeltas[weightId];
			inputWeightDeltas[weightId] = weightDelta;
			inputWeights[weightId] += weightDelta;
		}
	}

	__global__ void CWChangeRecurrentWeightsKernel(
		float *recurrentWeights,
		float *recurrentWeightDeltas,
		float *outputWeights,
		float *outputDeltas,
		float *recurrentWeightRTRLDerivatives,

		float trainingRate,
		float momentum,

		int *activeGroups,
		int contextByActivations
		)
	{
		int weightId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int unitID = weightId / D_HIDDEN_UNITS;
		int groupID = unitID / D_NEURONS_PER_GROUP;
		int x = weightId - (unitID * D_HIDDEN_UNITS);

		extern __shared__ float activeGroupsShared[];

		if (weightId < D_NEURON_GROUPS)
		{
			activeGroupsShared[weightId] = activeGroups[weightId];
		}
		__syncthreads();

		if (weightId < D_HIDDEN_UNITS * D_HIDDEN_UNITS 
			&& (contextByActivations || activeGroupsShared[groupID])
			&& (contextByActivations || (x >= (groupID * D_NEURONS_PER_GROUP))))		
		{
			float gradient = 0;

			for (int i = 0; i < D_OUTPUT_UNITS; i++)
			{
				float sum = 0;
				for (int j = 0; j < D_HIDDEN_UNITS; j++)
				{
					sum += outputWeights[i * D_HIDDEN_UNITS + j] * recurrentWeightRTRLDerivatives[j * D_HIDDEN_UNITS * D_HIDDEN_UNITS + weightId];
				}

				gradient += outputDeltas[i] * sum;
			}

			float weightDelta = trainingRate * gradient + momentum * recurrentWeightDeltas[weightId];
			recurrentWeightDeltas[weightId] = weightDelta;
			recurrentWeights[weightId] += weightDelta;
		}
	}

	__global__ void CWChangeOutputWeightsKernel(
		float *outputWeights,
		float *outputWeightDeltas,
		float *outputDeltas,
		float *hiddenActivations,

		float trainingRate,
		float momentum
		)
	{
		int weightId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int to = weightId / D_HIDDEN_UNITS;
		int from = weightId % D_HIDDEN_UNITS;

		if (weightId < D_OUTPUT_UNITS * D_HIDDEN_UNITS)
		{
			float gradient = outputDeltas[to] * hiddenActivations[from];
			float weightDelta = trainingRate * gradient + momentum * outputWeightDeltas[weightId];
			outputWeightDeltas[weightId] = weightDelta;
			outputWeights[weightId] += weightDelta;
		}
	}
}
