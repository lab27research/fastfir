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
