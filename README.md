# Fastfir

Fastfir is a library that implements fast 1D complex convolution using both CPU and GPU implementations.


Several implementations are derived from a common abtract class (FastFir):
* **FastFirCPU1** - CPU time domain reference implementation
* **FastFirCPU2** - CPU frequency domain implementation that uses FFTW (single threaded)
* **FastFirGPU1** - GPU frequency domain implementation that uses CUFFT
* **FastFirGPU2** - GPU frequency domain implementation that uses half-precision CUFFT (work in progress)


The interface is the same regardless of implementation:

```C++
//Base class interface definitions
FastFir::FastFir(float* mask, int mask_samps, int input_samps,
                 int buffers_per_call = 1, bool contiguous = false);
void FastFir::run(float* input, float* output);
```

All input/mask/output data is assumed to be interleaved 32-bit float values:

```C++
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


Performance numbers on i5-7600 (single-threaded):
```
Running explore for class FastFirCPU2
(mask size / input size / contiguous / proc_iterations ... GFLOPS)
config: 64 256 131072 0 4...21.462639 GFLOPs/sec
config: 128 512 65536 0 4...23.890910 GFLOPs/sec
config: 256 1024 32768 0 4...25.527554 GFLOPs/sec
config: 512 2048 16384 0 4...22.183814 GFLOPs/sec
config: 1024 4096 8192 0 4...19.528164 GFLOPs/sec
config: 2048 8192 4096 0 4...20.824465 GFLOPs/sec
config: 4096 16384 2048 0 4...20.037187 GFLOPs/sec
config: 8192 32768 1024 0 4...17.111707 GFLOPs/sec
config: 16384 65536 512 0 4...12.828131 GFLOPs/sec
config: 32768 131072 256 0 4...12.241786 GFLOPs/sec
config: 65536 262144 128 0 4...7.041684 GFLOPs/sec
config: 131072 524288 64 0 4...4.585242 GFLOPs/sec
config: 262144 1048576 32 0 4...3.403907 GFLOPs/sec
config: 524288 2097152 16 0 4...4.068113 GFLOPs/sec
config: 1048576 4194304 8 0 4...4.111998 GFLOPs/sec
config: 2097152 8388608 4 0 4...4.308033 GFLOPs/sec
config: 4194304 16777216 2 0 4...4.332632 GFLOPs/sec
config: 8388608 33554432 1 0 4...4.316943 GFLOPs/sec
config: 16777216 67108864 1 0 4...4.101851 GFLOPs/sec
```

Performance numbers on RTX-3090 (PCIe Gen3):
```
Running explore for class FastFirGPU1
(mask size / input size / contiguous / proc_iterations ... GFLOPS)
config: 64 256 131072 0 4...3.026979 GFLOPs/sec
config: 128 512 65536 0 4...6.604298 GFLOPs/sec
config: 256 1024 32768 0 4...13.681897 GFLOPs/sec
config: 512 2048 16384 0 4...30.443688 GFLOPs/sec
config: 1024 4096 8192 0 4...60.401805 GFLOPs/sec
config: 2048 8192 4096 0 4...135.219077 GFLOPs/sec
config: 4096 16384 2048 0 4...235.284316 GFLOPs/sec
config: 8192 32768 1024 0 4...316.503228 GFLOPs/sec
config: 16384 65536 512 0 4...352.383519 GFLOPs/sec
config: 32768 131072 256 0 4...366.766980 GFLOPs/sec
config: 65536 262144 128 0 4...413.205010 GFLOPs/sec
config: 131072 524288 64 0 4...446.584273 GFLOPs/sec
config: 262144 1048576 32 0 4...466.655972 GFLOPs/sec
config: 524288 2097152 16 0 4...476.422860 GFLOPs/sec
config: 1048576 4194304 8 0 4...469.793494 GFLOPs/sec
config: 2097152 8388608 4 0 4...432.012449 GFLOPs/sec
config: 4194304 16777216 2 0 4...362.628655 GFLOPs/sec
config: 8388608 33554432 1 0 4...275.207135 GFLOPs/sec
config: 16777216 67108864 1 0 4...286.948517 GFLOPs/sec
```

Note: this library will likely be renamed after finding similarly named repositories on github
