#include "FastFirCPU2.h"
#include "sse_utils.h"

#include <string.h>
#include <stdio.h>

FastFirCPU2::FastFirCPU2(float* mask, int mask_samps, int input_samps, int buffers_per_call, bool contiguous)
	:FastFir(mask, mask_samps, input_samps, buffers_per_call, contiguous)
{

	//Allocate input/output buffer and mask fft buffer
	ALIGNED_MALLOC(io_buffer_, 2 * fft_size_ * sizeof(float));
	ALIGNED_MALLOC(mask_buffer_, 2 * fft_size_ * sizeof(float));

	//Generate mask FFT buffer (only need to do this once)
	fwd_plan_ = fftwf_plan_dft_1d(fft_size_, (fftwf_complex*)mask_buffer_, (fftwf_complex*)mask_buffer_, FFTW_FORWARD, FFTW_ESTIMATE);
	memcpy(mask_buffer_, mask, 2 * mask_samps_ * sizeof(float));
	memset(&mask_buffer_[2 * mask_samps_], 0, 2 * (fft_size_ - mask_samps_) * sizeof(float));
	fftwf_execute(fwd_plan_);
	fftwf_destroy_plan(fwd_plan_);

	//Create 1D FFTW plans
	fwd_plan_ = fftwf_plan_dft_1d(fft_size_, (fftwf_complex*)io_buffer_, (fftwf_complex*)io_buffer_, FFTW_FORWARD, FFTW_ESTIMATE);
	rev_plan_ = fftwf_plan_dft_1d(fft_size_, (fftwf_complex*)io_buffer_, (fftwf_complex*)io_buffer_, FFTW_BACKWARD, FFTW_ESTIMATE);

}

FastFirCPU2::~FastFirCPU2()
{

	//Destroy all FFTW plans
	fftwf_destroy_plan(fwd_plan_);
	fftwf_destroy_plan(rev_plan_);

	//Free all memory
	ALIGNED_FREE(io_buffer_);
	ALIGNED_FREE(mask_buffer_);

}

void FastFirCPU2::run(float* input, float* output)
{

	for (int ii = 0; ii < buffers_per_call_; ii++) {

		float* in_ptr = &input[2 * ii * input_samps_];
		float* out_ptr = &output[2 * ii * output_samps_];

		//Copy in and zero pad
		memcpy(io_buffer_, in_ptr, 2 * fft_size_ * sizeof(float));
		memset(&io_buffer_[2 * input_samps_], 0, 2 * (fft_size_ - input_samps_) * sizeof(float));

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
		}

		//Run rev fft
		fftwf_execute(rev_plan_);

		//Copy to output
		memcpy(out_ptr, io_buffer_, 2 * sizeof(output_samps_));
	}

}
