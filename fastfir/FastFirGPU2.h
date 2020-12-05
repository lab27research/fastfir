#pragma once

#include "cuda_utils.h"

#include "FastFir.h"

#include <vector>

//Freq domain GPU implementation of the FastFir (overlap and add implementation)
//
//Same as FastFirGPU1, but added conversion to half precision types before each
// transfer (loss of precision on input and loss of precision on output)

class FastFirGPU2 : public FastFir
{
public:
    FastFirGPU2(float* mask, int mask_samps, int input_samps,
                int buffers_per_call = 1, bool contiguous = false);
    ~FastFirGPU2();
    void run(float* input, float* output);

    //Allows override of number of streams
    void setNumProcStreams(int num_streams);

private:
    void initProcStreams(int num_streams);

    int fft_size_;

    //Device buffers
    nv_bfloat16* h_io_nv_bfloat16_buffer_;
    nv_bfloat16* d_io_nv_bfloat16_buffer_;
    float* d_io_buffer_;
    float* d_mask_buffer_;

    //Streams/plan variables (one for each processing stream)
    std::vector<cudaStream_t> proc_streams_;//Runs all processing (H->D, kernels, and D->H)
    std::vector<cufftHandle> cufft_plans_;

    //Stream synchronization events (one for each input buffer)
    std::vector<cudaEvent_t> kernels_done_events_;

};
