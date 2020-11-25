#include "ImpulseSource.h"

ImpulseSource::ImpulseSource(double snr, unsigned long long samp0)
{
    snr_ = snr;
    samp0_ = samp0;
    current_samp_ = 0;
}

void ImpulseSource::getBuffer(float* buffer, int buffer_samps)
{
    //Generate buffer of noise
    for (int ii = 0; ii < buffer_samps; ii++) {
        buffer[2 * ii] = 0;
        buffer[2 * ii + 1] = 0;
    }

    //Add in impulse if we have hit sample samp0
    if (current_samp_ < samp0_ && samp0_ < current_samp_ + buffer_samps) {
        int ii = (int)(samp0_ - current_samp_);
        buffer[2 * ii] += 1;
        //Note: Adding entire amplitude to real portion (phase=0)
    }

    current_samp_ += buffer_samps;
}

ImpulseSource::~ImpulseSource()
{
    //Nothing to do here for now
}
