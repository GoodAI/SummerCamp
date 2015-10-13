#ifndef GRANDFATHER_H
#define GRANDFATHER_H

#include "cuda_runtime.h"


__device__ unsigned int LogBase2(int v);

__device__ int BitsNeeded(int value);

#endif