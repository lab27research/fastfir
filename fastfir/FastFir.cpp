#include "FastFir.h"

#include <cmath>
#include <stdio.h>

FastFir::FastFir(float* mask, int mask_samps, int input_samps, int buffers_per_call)
{

	//Store off input/mask/output parameters
	mask_samps_ = mask_samps;
	input_samps_ = input_samps;
	output_samps_ = FastFir::getOutputSamps(mask_samps, input_samps);
	buffers_per_call_ = buffers_per_call;

	//Choose an FFT Size equal to next power of 2
	fft_size_ = input_samps + mask_samps - 1;
}

FastFir::~FastFir()
{
	//Nothing for now
}

void FastFir::run(float* input, float* output)
{
	printf("WARNING :: Executing FastFir.run() base class... You probably shouldn't be doing this!");
}

int FastFir::getFFTSize(int mask_samps, int input_samps)
{
	//Code grabbed from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	unsigned int vv = input_samps + mask_samps - 1;
	vv--;
	vv |= vv >> 1;
	vv |= vv >> 2;
	vv |= vv >> 4;
	vv |= vv >> 8;
	vv |= vv >> 16;
	vv++;
	return (int)vv;
}

int FastFir::getOutputSamps(int mask_samps, int input_samps)
{
	return input_samps - mask_samps + 1;
}

double FastFir::getTimeDomainFLOPs(int mask_samps, int input_samps)
{
	double MM = mask_samps;
	double NN = input_samps;
	return (8 * MM - 1) * (NN - MM + 1) + ((8 * MM * (MM - 1)) / 2) - (MM - 1);

}

double FastFir::getFreqDomainFLOPs(int mask_samps, int input_samps)
{
	double WW = FastFir::getFFTSize(mask_samps, input_samps);
	return (10 * WW * log2(WW)) + (6 * WW);
}
