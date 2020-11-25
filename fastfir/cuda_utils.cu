#include "cuda_utils.h"

int getMaxThreadsPerBlock(int gpu_index) {
    int tpb = 0;
    cudaDeviceGetAttribute(&tpb, cudaDevAttrMaxThreadsPerBlock, gpu_index);
    return tpb;
}

int getNumBlocks(int tpb, int total_threads) {
    return total_threads / tpb;
}