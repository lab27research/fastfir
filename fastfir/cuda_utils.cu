#include "cuda_utils.h"

int getMaxThreadsPerBlock(int gpu_index) {
    int tpb = 0;
    cudaDeviceGetAttribute(&tpb, cudaDevAttrMaxThreadsPerBlock, gpu_index);
    return tpb;
}

int getNumBlocks(int tpb, int total_threads) {
    int num_blocks = total_threads / tpb;
    if (num_blocks * tpb < total_threads) {
        num_blocks += 1;
    }
    return num_blocks;
}