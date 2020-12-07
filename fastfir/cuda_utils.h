/*
* Copyright 2020 Curtis Rhodes
*
* This file is part of Fastfir.
*
* Fastfir is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Fastfir is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with Fastfir.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

////Standard CUDA includes (really just to make IDEs happy)
#include "cuda.h"
#include "cuda_runtime_api.h"

////CUFFT includes
#include "cufft.h"
#include "cufftXt.h"

////Helper functions from samples
#include "helper_cuda.h"

////NVTX include
#include "nvtx3/nvToolsExt.h"

////Half type includes
#include "cuda_fp16.h"
#include "cuda_bf16.h"

////Memory allocation defines
#define HOST_MALLOC(ptr,size) checkCudaErrors(cudaMallocHost(ptr,size))
#define HOST_FREE(ptr) checkCudaErrors(cudaFreeHost(ptr));
#define DEVICE_MALLOC(ptr,size) checkCudaErrors(cudaMalloc(ptr,size))
#define DEVICE_FREE(ptr) checkCudaErrors(cudaFree(ptr));

////Utilities for block/thread management

//Returns the max threads per block for the GPU at index gpu_index
int getMaxThreadsPerBlock(int gpu_index);
//Returns the max threads that can run concurrently on each SM
int getMaxThreadsPerSM(int gpu_index);
//Returns the highest number of threads per block that can still achieve 100% occupancy
// (assuming only contraint is tpb)
int getBestTPB(int gpu_index);
//Returns the number of blocks necessary a ensure a minimum of total_threads
int getNumBlocks(int tpb, int total_threads);
