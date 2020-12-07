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

#include "cuda_utils.h"

#include "FastFir.h"

#include <vector>

//Freq domain GPU implementation of the FastFir (overlap and add implementation)
//
//Utilizes one stream for H->D transfers and several streams for processing/D->H transfers
//
//Processing chain:
// [Stream 1] H->D Transfer (sequentially for each input buffer)
// [Stream 2] cufft Forward FFT, custom cpx mpy + scaling, cufft Inverse FFT, D->H Transfer
// [Stream 3] cufft Forward FFT, custom cpx mpy + scaling, cufft Inverse FFT, D->H Transfer
// [Stream 4] cufft Forward FFT, custom cpx mpy + scaling, cufft Inverse FFT, D->H Transfer
// [Stream 5] cufft Forward FFT, custom cpx mpy + scaling, cufft Inverse FFT, D->H Transfer
// ...
//
//All processing complex float
//
//Each processing stream synchronized to input transfer through "transfer1_done_events"
//
//For contiguous outputs, processing streams are synchronized with previous processing stream
// so they can add in the results from the previous stream (using "kernels_done_events")

class FastFirGPU1 : public FastFir
{
public:
    FastFirGPU1(float* mask, int mask_samps, int input_samps,
                int buffers_per_call = 1, bool contiguous = false);
    ~FastFirGPU1();
    void run(float* input, float* output);

    //Allows override of number of streams
    void setNumProcStreams(int num_streams);

private:
    void initProcStreams(int num_streams);

    int fft_size_;

    //Device buffers
    float* d_io_buffer_;
    float* d_mask_buffer_;

    //Streams/plan variables (one for each processing stream)
    std::vector<cudaStream_t> proc_streams_;//Runs all processing (H->D, kernels, and D->H)
    std::vector<cufftHandle> cufft_plans_;

    //Stream synchronization events (one for each input buffer)
    std::vector<cudaEvent_t> kernels_done_events_;

};
