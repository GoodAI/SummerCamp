
#include "utils.h"
#include "cuda_runtime.h"
#include "vars.h"
#include <limits.h>

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

__device__ unsigned int rand_uint(){
	//linear congruence. Values from Numerical Recipes
	
	X = (A*X + C);
	return X;
}

__device__ int rand_int(){
	return (int)rand_uint();
}

__device__ float rand_float(){
	return ((float)rand_uint()) / UINT_MAX;
}

__device__ float rand_range(float from, float to){
	float range = to - from;
	return range*rand_float() + from;
}

__device__ bool rand_decision(float probability){
	return rand_float() < probability;
}

__device__ int rand_int_range(int min, int max){
	//both min and max are possible values
	int range = max - min;
	return (rand_uint() % (range+1)) + min;
}


//__device__ float RandomFloat(float min, float max)

//__device__ bool ProbabilisticDecision(float limit) 

//__device__ int RandomElement(int[] a) 

//Log1P is math.h log1p 

// Encode & Decode: Not needed (maybe?)

// IntArrayToString: not needed so far