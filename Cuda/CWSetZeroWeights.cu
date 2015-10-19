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

	//kernel code
	__global__ void CWSetZeroWeights(float *recurrentWeights, int neuronGroup)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		if (unitId < neuronGroup*D_NEURONS_PER_GROUP)
		{
			recurrentWeights[unitId] = 0;
		}
	}
}