#include "FastFirCPU1.h"
#include "sse_utils.h"

#include <string.h>
#include <stdio.h>

FastFirCPU1::FastFirCPU1(float* mask, int mask_samps, int input_samps, int buffers_per_call, bool contiguous)
	:FastFir(mask, mask_samps, input_samps, buffers_per_call, contiguous)
{
	//Allocate mask buffer and copy in mask data
	ALIGNED_MALLOC(mask_buffer_, 2 * mask_samps_ * sizeof(float));

	//Reverse mask in memory (we are performing convolution)
	for (int ii = 0; ii < mask_samps; ii++) {
		mask_buffer_[2 * ii] = mask[2 * (mask_samps - 1 - ii)];
		mask_buffer_[2 * ii + 1] = mask[2 * (mask_samps - 1 - ii) + 1];
	}

}

FastFirCPU1::~FastFirCPU1()
{
	//Free mask memory
	ALIGNED_FREE(mask_buffer_);

}

void FastFirCPU1::run(float* input, float* output)
{

	int iterations;
	int NN;
	int MM = mask_samps_;
	if (contiguous_) {
		//Treat entire input as though it is one buffer
		// (only one double sided transient for entire input)
		iterations = 1;
		NN = input_samps_ * buffers_per_call_;
	}
	else {
		//Run separate calls for each buffer
		// (output will have double sided transient for each input)
		iterations = buffers_per_call_;
		NN = input_samps_;
	}

	float* out_ptr = output;

	for (int ii = 0; ii < iterations; ii++) {

		//Establish pointers for each processing buffer
		float* start_in_ptr = &input[2 * ii * input_samps_];

		//Process left transient
		for (int jj = 0; jj < MM - 1; jj++) {
			float accum[2] = { 0.0,0.0 };
			float* mask_ptr = &mask_buffer_[2 * (mask_samps_ - 1)];//Start at end of mask
			float* in_ptr = start_in_ptr + 2 * jj;//Start at end of input to be included

			for (int kk = 0; kk < jj + 1; kk++) {
				complex_madd(mask_ptr[0], mask_ptr[1], in_ptr[0], in_ptr[1], accum);

				mask_ptr -= 2;//Walk backwards
				in_ptr -= 2;//Walk backwards
			}

			out_ptr[0] = accum[0];
			out_ptr[1] = accum[1];
			out_ptr += 2;

		}

		//Process full overlap section
		for (int jj = 0; jj < NN - MM + 1; jj++) {
			float accum[2] = { 0.0,0.0 };
			float* mask_ptr = mask_buffer_;//Start at beginning of mask
			float* in_ptr = start_in_ptr + 2 * jj;//Start at beginning of input to be include

			for (int kk = 0; kk < mask_samps_; kk++) {
				complex_madd(mask_ptr[0], mask_ptr[1], in_ptr[0], in_ptr[1], accum);

				mask_ptr += 2;//Walk forwards
				in_ptr += 2;//Walk forwards
			}

			out_ptr[0] = accum[0];
			out_ptr[1] = accum[1];
			out_ptr += 2;

		}

		//Process right transient
		for (int jj = 0; jj < MM - 1; jj++) {
			float accum[2] = { 0.0,0.0 };
			float* mask_ptr = mask_buffer_;//Start at beginning of mask
			float* in_ptr = start_in_ptr + 2 * (NN - MM + 1 + jj);//Start at beginning of input to be include

			for (int kk = 0; kk < MM - 1 - jj; kk++) {
				complex_madd(mask_ptr[0], mask_ptr[1], in_ptr[0], in_ptr[1], accum);

				mask_ptr += 2;//Walk forwards
				in_ptr += 2;//Walk forwards
			}

			out_ptr[0] = accum[0];
			out_ptr[1] = accum[1];
			out_ptr += 2;

		}
	}

}


