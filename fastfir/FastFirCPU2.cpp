#include "FastFirCPU2.h"
#include "math_utils.h"
#include "cuda_utils.h"

#include <string.h>
#include <stdio.h>

FastFirCPU2::FastFirCPU2(float* mask, int mask_samps, int input_samps,
                         int buffers_per_call, bool contiguous)
    :FastFir(mask, mask_samps, input_samps, buffers_per_call, contiguous)
{

    //Choose an FFT Size equal to next power of 2
    fft_size_ = FastFir::getFFTSize(mask_samps, input_samps);

    //Allocate input/output buffer and mask fft buffer
    size_t io_buffer_bytes = sizeof(float) * 2 * fft_size_;
    size_t mask_buffer_bytes = sizeof(float) * 2 * fft_size_;
    HOST_MALLOC(&io_buffer_, io_buffer_bytes);
    HOST_MALLOC(&mask_buffer_, mask_buffer_bytes);

    //Generate mask FFT buffer (only need to do this once)
    fwd_plan_ = fftwf_plan_dft_1d(fft_size_,
                                  (fftwf_complex*)mask_buffer_,
                                  (fftwf_complex*)mask_buffer_,
                                  FFTW_FORWARD,
                                  FFTW_ESTIMATE);
    size_t mask_bytes = sizeof(float) * 2 * mask_samps_;
    size_t non_mask_bytes = sizeof(float) * 2 * (fft_size_ - mask_samps_);
    memcpy(mask_buffer_, mask, mask_bytes);
    memset(&mask_buffer_[2 * mask_samps_], 0, non_mask_bytes);
    fftwf_execute(fwd_plan_);
    fftwf_destroy_plan(fwd_plan_);

    //Create 1D FFTW plans
    fwd_plan_ = fftwf_plan_dft_1d(fft_size_,
                                  (fftwf_complex*)io_buffer_,
                                  (fftwf_complex*)io_buffer_,
                                  FFTW_FORWARD,
                                  FFTW_ESTIMATE);
    rev_plan_ = fftwf_plan_dft_1d(fft_size_,
                                  (fftwf_complex*)io_buffer_,
                                  (fftwf_complex*)io_buffer_,
                                  FFTW_BACKWARD,
                                  FFTW_ESTIMATE);

}

FastFirCPU2::~FastFirCPU2()
{

    //Destroy all FFTW plans
    fftwf_destroy_plan(fwd_plan_);
    fftwf_destroy_plan(rev_plan_);

    //Free all memory
    HOST_FREE(io_buffer_);
    HOST_FREE(mask_buffer_);

}

void FastFirCPU2::run(float* input, float* output)
{
    size_t input_bytes = sizeof(float) * 2 * input_samps_;
    size_t non_input_bytes = sizeof(float) * 2 * (fft_size_ - input_samps_);
    int output_samps_0sided = FastFir::getOutputSampsNoTransient(mask_samps_, input_samps_);
    int output_samps_1sided = FastFir::getOutputSamps1Sided(mask_samps_, input_samps_);
    int output_samps_2sided = FastFir::getOutputSamps2Sided(mask_samps_, input_samps_);
    size_t output_bytes_0sided = sizeof(float) * 2 * output_samps_0sided;
    size_t output_bytes_1sided = sizeof(float) * 2 * output_samps_1sided;
    size_t output_bytes_2sided = sizeof(float) * 2 * output_samps_2sided;
    int left_transient_samps = output_samps_1sided - output_samps_0sided;
    int non_left_transient_samps = output_samps_2sided - left_transient_samps;
    size_t non_left_transient_bytes = sizeof(float) * 2 * non_left_transient_samps;

    float* in_ptr = input;
    float* out_ptr = output;
    for (int ii = 0; ii < buffers_per_call_; ii++) {

        //Copy in and zero pad
        memcpy(io_buffer_, in_ptr, input_bytes);
        memset(&io_buffer_[2 * input_samps_], 0, non_input_bytes);

        //Run fwd fft
        fftwf_execute(fwd_plan_);

        //Perform complex multiply and scaling
        float scale = (float)(1.0 / fft_size_);
        for (int jj = 0; jj < fft_size_; jj++) {
            float aa = io_buffer_[2 * jj];
            float bb = io_buffer_[2 * jj + 1];
            float cc = mask_buffer_[2 * jj];
            float dd = mask_buffer_[2 * jj + 1];
            //(a+bj)*(c+dj) = ac-bd + j(ad+bc)

            io_buffer_[2 * jj] = (aa * cc - bb * dd) * scale;
            io_buffer_[2 * jj + 1] = (aa * dd + bb * cc) * scale;

            //complex_madd(aa, bb, cc, dd, &io_buffer_[2 * jj]);
        }

        //Run rev fft
        fftwf_execute(rev_plan_);

        //Copy to output
        if (!contiguous_) {
            //Simply set output directly from io_buffer_
            memcpy(out_ptr, io_buffer_, output_bytes_2sided);

            //Move by full 2-sided transient
            out_ptr += 2 * output_samps_2sided;

        }
        else {
            if (ii == 0) {
                //For first buffer simply write entire result to memory
                memcpy(out_ptr, io_buffer_, output_bytes_2sided);

                //Move by full 1-sided samples
                out_ptr += 2 * output_samps_1sided;

            }
            else {
                //For all other buffers

                //Add in left transient
                for (int ii = 0; ii < left_transient_samps; ii++) {
                    out_ptr[2 * ii] += io_buffer_[2 * ii];
                    out_ptr[2 * ii + 1] += io_buffer_[2 * ii + 1];
                }

                //Set the rest
                memcpy(&out_ptr[2 * left_transient_samps],
                       &io_buffer_[2 * left_transient_samps],
                       non_left_transient_bytes);

                //Move by right transient of previous buffer and full
                // overlap samps for current buffer
                out_ptr += 2 * output_samps_1sided;
            }
        }

        in_ptr += 2 * input_samps_;
    }

}
