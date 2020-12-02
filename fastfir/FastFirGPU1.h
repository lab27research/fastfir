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

    //Transfer streams
    cudaStream_t transfer1_stream_;//For transfers H->D

    //Streams/plan variables (one for each processing stream)
    std::vector<cudaStream_t> proc_streams_;//For all processing and D->H transfers
    std::vector<cufftHandle> cufft_plans_;

    //Stream synchronization events (one for each buffer)
    std::vector<cudaEvent_t> transfer1_done_events_;
    std::vector<cudaEvent_t> kernels_done_events_;

};

__global__ void vectorCpxMpy(float* input1, float* input2, float* output, int NN);
__global__ void vectorCpxScale(float* input1, float* output, float scale, int NN);
__global__ void vectorCpxAdd(float* input1, float* input2, float* output, int NN);