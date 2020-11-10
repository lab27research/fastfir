#pragma once

#include "fftw3.h"

#include "FastFir.h"

//Freq domain CPU implementation of the FastFir (overlap and save implementation)
class FastFirCPU2: public FastFir
{
public:
	FastFirCPU2(float* mask, int mask_samps, int input_samps, int buffers_per_call=1, bool contiguous=false);
	~FastFirCPU2();
	void run(float* input, float* output);

private:
	float* io_buffer_;
	float* mask_buffer_;
	fftwf_plan fwd_plan_;
	fftwf_plan rev_plan_;
	
};

