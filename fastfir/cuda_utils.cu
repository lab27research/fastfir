#include "cuda_utils.h"

int getMaxThreadsPerBlock(int gpu_index) {
    int tpb = 0;
    cudaDeviceGetAttribute(&tpb, cudaDevAttrMaxThreadsPerBlock, gpu_index);
    return tpb;
}

int getMaxThreadsPerSM(int gpu_index) {
    int tps = 0;
    cudaDeviceGetAttribute(&tps, cudaDevAttrMaxThreadsPerMultiProcessor, gpu_index);
    return tps;
}

int getBestTPB(int gpu_index) {
    int tpb = getMaxThreadsPerBlock(gpu_index);
    int tps = getMaxThreadsPerSM(gpu_index);

    int best_tpb = tps;
    while (best_tpb > tpb) {
        best_tpb /= 2;
    }
    return best_tpb;
}

int getNumBlocks(int tpb, int total_threads) {
    int num_blocks = total_threads / tpb;
    if (num_blocks * tpb < total_threads) {
        num_blocks += 1;
    }
    return num_blocks;
}
