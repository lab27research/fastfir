#include "FastFir.h"

#include <cmath>
#include <stdio.h>

FastFir::FastFir(float* mask, int mask_samps, int input_samps, int buffers_per_call, bool contiguous)
{
    //Store off input parameters
    mask_samps_ = mask_samps;
    input_samps_ = input_samps;
    buffers_per_call_ = buffers_per_call;
    contiguous_ = contiguous;
}

FastFir::~FastFir()
{
    //Nothing for now
}

int FastFir::getTotalOutputSamps()
{
    if (!contiguous_) {
        return buffers_per_call_ * FastFir::getOutputSamps2Sided(input_samps_, mask_samps_);
    }
    else {
        return FastFir::getOutputSamps2Sided(buffers_per_call_ * input_samps_, mask_samps_);
    }
}

void FastFir::run(float* input, float* output)
{
    printf("WARNING :: Executing FastFir.run() base class... You probably shouldn't be doing this!");
}

int FastFir::getFFTSize(int mask_samps, int input_samps)
{
    //Code grabbed from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    unsigned int vv = FastFir::getOutputSamps2Sided(mask_samps, input_samps);

    if (vv == 0) {
        //Technically next power of two would be 1 (would return 0 unless we intervene)
        return 1;
    }

    vv--;
    vv |= vv >> 1;
    vv |= vv >> 2;
    vv |= vv >> 4;
    vv |= vv >> 8;
    vv |= vv >> 16;
    vv++;
    return (int)vv;
}

int FastFir::getOutputSampsNoTransient(int mask_samps, int input_samps)
{
    return input_samps - mask_samps + 1;
}

int FastFir::getOutputSamps1Sided(int mask_samps, int input_samps)
{
    return input_samps;
}

int FastFir::getOutputSamps2Sided(int mask_samps, int input_samps)
{
    return input_samps + mask_samps - 1;
}

double FastFir::getTimeDomainFLOPs(int mask_samps, int input_samps)
{
    double MM = mask_samps;
    double NN = input_samps;
    return (8 * MM - 1) * (NN - MM + 1) + ((8 * MM * (MM - 1)) / 2) - (MM - 1);

}

double FastFir::getFreqDomainFLOPs(int mask_samps, int input_samps)
{
    //Assume non-contiguous
    double fft_size = FastFir::getFFTSize(mask_samps, input_samps);

    //Note: two transforms (forward and reverse), one complex mpy per fft point
    double cpx_mpys = (2*(fft_size / 2) * log2(fft_size) + fft_size);
    double cpx_adds = (2*(fft_size)*log2(fft_size));

    //Also include scaling operation (to accomodate for non-scaled transforms)
    cpx_mpys += 2 * fft_size;

    return 6 * cpx_mpys + 2 * cpx_adds;
}
