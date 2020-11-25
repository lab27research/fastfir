#pragma once
class ImpulseSource
{
public:
    ImpulseSource(double snr, unsigned long long samp0);
    void getBuffer(float* buffer, int buffer_samps);
    ~ImpulseSource();
private:
    double snr_;
    unsigned long long samp0_;
    unsigned long long current_samp_;
};

