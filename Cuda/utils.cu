
#include "utils.h"

#include "cuda_runtime.h"

#include "vars.h"

__device__ unsigned int LogBase2(int v){//beware behavior at 0?
	//from https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious
	int r = 0;

	while (v >>= 1)
	{
		r++;
	}
	return r;
}

__device__ int BitsNeeded(int value) {
	if (value == 0) return 0;
	return LogBase2(value) + 1;
}


//__device__ float RandomFloat(float min, float max)

//__device__ bool ProbabilisticDecision(float limit) 

//__device__ int RandomElement(int[] a) 

//Log1P is math.h log1p 

// Encode & Decode: Not needed (maybe?)

// IntArrayToString: not needed so far