# Fastfir

Fastfir is a library that implements fast 1D complex convolution using both CPU and GPU implementations.


Several implementations are derived from a common abtract class (FastFir):
* **FastFirCPU1** - CPU time domain reference implementation
* **FastFirCPU2** - CPU frequency domain implementation that uses FFTW (single threaded)
* **FastFirGPU1** - GPU frequency domain implementation that uses CUFFT
* **FastFirGPU2** - GPU frequency domain implementation that uses half-precision CUFFT (work in progress)


The interface is the same regardless.  All input/mask/output data is assumed to be interleaved 32-bit float values.

```C++
FastFirGPU1(float* mask, int mask_samps, int input_samps,
            int buffers_per_call = 1, bool contiguous = false);
void run(float* input, float* output);

//Set up input buffers
int mask_samps = 1024;
int input_samps = 4096;
float* mask = (float*) malloc(sizeof(float) * 2 * mask_samps);
float* input = (float*) malloc(sizeof(float) * 2 * input_samps);

//Create FastFir instantiation
FastFirGPU1 ff(mask, mask_samps, input_samps);

//Set up output buffer
int output_samps = ff.getTotalOutputSamps();
float* output = (float*) malloc(sizeof(float) * 2 * output_samps);

//Execute convolution
ff.run(input,output);
```

Note: this library will likely be renamed after finding similarly named repositories on github
