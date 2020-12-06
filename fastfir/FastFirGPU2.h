#pragma once

#include "cuda_utils.h"

#include "FastFir.h"

#include <vector>

//Freq domain GPU implementation of the FastFir (overlap and add implementation)
//
//Same as FastFirGPU1, using 16-bit transfers/processing
//
//Note:
// All GPU work (FFTs and kernels) uses FP16 (half) processing
//    ...my original goal here was to use bfloat16 processing, but it appears
//    that I cannot get cufft to perform complex-to-complex bfloat16 FFTs
//    on the RTX 3090 (I receive CUFFT_INVALID_TYPE when making the plan using
//    CUDA_C_16BF as the passed type)
// As I wanted to keep the run(float* input, float* output) interface the same,
//    I needed to perform 32-bit -> 16-bit conversion prior to the transfers using
//    the CPU.  Using SIMD instructions I can create a reasonably fast conversion from
//    float to nv_bfloat16, so that is why the numbers are transfered in this way.
//
// So in summary:
//  [on CPU] convert float -> nv_bfloat16
//    H->D transfer
//  [on GPU] nv_bfloat16 -> half
//  [on GPU] cufft FWD call, scaling/mpy kernels, cufft INV call
//  [on GPU] half -> nv_bfloat16
//    D->H transfer
//  [on CPU] nv_bfloat16 -> float
//
// Note that if I can figure out why cufft will not let me using bfloat16 this will
// eliminate several steps.

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

    //Buffers for several conversion layers
    nv_bfloat16* h_bfloat16_buffer_;//size: 2*bpc*fft_size
    nv_bfloat16* d_bfloat16_buffer_;//size: 2*bpc*fft_size
    half* d_half_buffer_;//size: 2*bpc*fft_size

    //Mask buffer
    half* d_mask_buffer_;

    //Streams/plan variables (one for each processing stream)
    std::vector<cudaStream_t> proc_streams_;//Runs all processing (H->D, kernels, and D->H)
    std::vector<cufftHandle> cufft_plans_;

    //Stream synchronization events (one for each input buffer)
    std::vector<cudaEvent_t> kernels_done_events_;

};
