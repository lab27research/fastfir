#pragma once

#include "cuda_utils.h"

#include "FastFir.h"

#include <vector>

//Freq domain GPU implementation of the FastFir (overlap and add implementation)
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

    //Transfer streams
    cudaStream_t transfer1_stream_;//For transfers H->D

    //Streams/plan variables, one for each processing stream
    std::vector<cudaStream_t> proc_streams_;//For all processing and D->H transfers
    std::vector<cufftHandle> cufft_plans_;

    //Stream synchronization events (one for each buffer)
    std::vector<cudaEvent_t> transfer1_done_events_;
    std::vector<cudaEvent_t> kernels_done_events_;

};

__global__ void vectorCpxMpy(float* input1, float* input2, float* output, int NN);
__global__ void vectorCpxScale(float* input1, float* output, float scale, int NN);
__global__ void vectorCpxAdd(float* input1, float* input2, float* output, int NN);