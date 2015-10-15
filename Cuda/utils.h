#ifndef GRANDFATHER_H
#define GRANDFATHER_H

#include "cuda_runtime.h"


__device__ unsigned int LogBase2(int v);

__device__ int BitsNeeded(int value);

__device__ unsigned int rand_uint();

__device__ int rand_int();

__device__ float rand_float();

__device__ float rand_range(float from, float to);

__device__ bool rand_decision(float probability);

__device__ int rand_int_range(int min, int max);


#endif