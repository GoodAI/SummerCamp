
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

// use existing
extern "C"
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_HIDDEN_UNITS;
	__constant__ int D_OUTPUT_UNITS;

	__global__ void partialDerivatives(
		float* activation,
		float* target,
		float* outputDelta,
		int* periods
		)
	{
		int threadId = blockIdx.y*blockDim.x*gridDim.x
			+ blockIdx.x*blockDim.x
			+ threadIdx.x;

		if (threadId < D_OUTPUT_UNITS)
		{
		}
	}
}
