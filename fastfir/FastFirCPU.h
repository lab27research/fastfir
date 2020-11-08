#pragma once

#include "fftw3.h"

#include "FastFir.h"

class FastFirCPU: public FastFir
{
public:
	FastFirCPU(float* mask, int mask_samps, int input_samps, int buffers_per_call);
	~FastFirCPU();
	void run(float* input, float* output);

private:
	float* io_buffer_;
	float* mask_buffer_;
	fftwf_plan fwd_plan_;
	fftwf_plan rev_plan_;
	
};

