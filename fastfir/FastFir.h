#pragma once
//Create a virtual class that all implementations can derive from
class FastFir
{
public:
	//Constructs a Fast Fir implementation that will process "buffers_per_call" sets of
	// size "input_samps" using specified input mask
	FastFir(float* mask, int mask_samps, int input_samps, int buffers_per_call);
	~FastFir();

	//Implementation must process "buffers_per_call" sets of size "input_samps" that are
	// assumed to be contiguous at the pointer "input"
	//Results are placed in "buffers_per_call" sets of size getOutputSamps()
	virtual void run(float* input, float* output);

	static int getFFTSize(int mask_samps, int input_samps);
	static int getOutputSamps(int mask_samps, int input_samps);
	static double getTimeDomainFLOPs(int mask_samps, int input_samps);
	static double getFreqDomainFLOPs(int mask_samps, int input_samps);

protected:
	int mask_samps_;
	int input_samps_;
	int buffers_per_call_;
	int fft_size_;
	int output_samps_;
};


