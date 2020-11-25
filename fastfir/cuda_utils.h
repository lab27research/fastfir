#pragma once

////Standard CUDA includes (really just to make IDEs happy)
#include "cuda.h"
#include "cuda_runtime_api.h"

////CUFFT includes
#include "cufft.h"
#include "cufftXt.h"

////Helper functions from samples
#include "helper_cuda.h"

////Memory allocation defines
#define HOST_MALLOC(ptr,size) checkCudaErrors(cudaMallocHost(ptr,size))
#define HOST_FREE(ptr) checkCudaErrors(cudaFreeHost(ptr));
#define DEVICE_MALLOC(ptr,size) checkCudaErrors(cudaMalloc(ptr,size))
#define DEVICE_FREE(ptr) checkCudaErrors(cudaFree(ptr));

////Utilities for block/thread management

//Returns the max threads per block for the GPU at index gpu_index
int getMaxThreadsPerBlock(int gpu_index);
//Returns the number of blocks necessary a ensure a minimum of total_threads
int getNumBlocks(int tpb, int total_threads);
