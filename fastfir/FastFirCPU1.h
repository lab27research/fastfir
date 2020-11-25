#pragma once

#include "fftw3.h"

#include "FastFir.h"

//Time domain CPU implementation of the FastFir (reference design)
class FastFirCPU1 : public FastFir
{
public:
	//Specify filter mask to be used for convolution, how many input samps per buffer, and
	// the number of buffers per call
	FastFirCPU1(float* mask, int mask_samps, int input_samps, int buffers_per_call=1, bool contiguous=false);
	~FastFirCPU1();
	void run(float* input, float* output);

private:
	float* mask_buffer_;

};