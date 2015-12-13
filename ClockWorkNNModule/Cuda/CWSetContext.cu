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
	__constant__ int D_NEURONS_PER_GROUP;
	__constant__ int D_NEURON_GROUPS;
	__constant__ int D_HIDDEN_UNITS;

	// kernel code
	// mean of activations of all activated units over the period
	// is adding to the activation of the before turned off unit.

	__global__ void CWSetContext(float *activations, float *contextActivations, int *activeGroups)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int unitID = id % D_HIDDEN_UNITS;
		int groupID = unitID / D_NEURONS_PER_GROUP;

		if (id < D_HIDDEN_UNITS * D_NEURON_GROUPS && activeGroups[groupID] != 0)
		{
			int unitPeriod = unitID / D_NEURONS_PER_GROUP + 1;
			int groupPeriod = id / D_HIDDEN_UNITS;
			contextActivations[id] += activations[unitID] * (unitPeriod / groupPeriod);
		}
	}
}