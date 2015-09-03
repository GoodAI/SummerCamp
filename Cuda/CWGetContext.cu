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
	__constant__ int D_NEURONS_GROUPS;
	__constant__ int D_HIDDEN_UNITS;

	//kernel code
	__global__ void CWGetContext(float *activations, float *contextActivations)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		
		if (id < D_NEURONS_PER_GROUP*D_NEURONS_GROUPS)
		{
			int unitID = id % D_HIDDEN_UNITS;
			int unitPeriod = unitID / D_NEURONS_PER_GROUP + 1;
			int groupPeriod = id / D_HIDDEN_UNITS;
			contextActivations[id] += activations[unitID] * (unitPeriod / groupPeriod);
		}
	}
}