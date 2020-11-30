#include "test_benches.h"
#include "math_utils.h"
#include "cuda_utils.h"

//Tests White Gaussian Noise generation
void test_generate_wgn_cf() {
    const int output_samps = 1000;
    float* output;
    HOST_MALLOC(&output, 2 * output_samps * sizeof(float));
    generate_wgn_cf(0.5, 0.1, output, output_samps);
    datplot_write_cf("noise.csv", output, output_samps, 0, 1);
}

//Tests CUFFT (without memory transfers to/from GPU)
void test_cufft() {
    printf("Running test_cufft...\n");
    int iterations = 100;
    size_t min_pow = 4;
    size_t max_pow = 25;
    size_t max_fft_size = round(pow(2, max_pow));

    //Allocate device memory for largest tranform
    float* buffer;
    printf("Running test_cufft, min_pow=%i, max_pow=%i\n", min_pow, max_pow);
    DEVICE_MALLOC((void**)&buffer, 2 * max_fft_size * sizeof(float));
    cudaMemset(buffer, 1, 2 * max_fft_size * sizeof(float));

    for (int ii = min_pow; ii <= max_pow; ii++) {
        int fft_size = round(pow(2, ii));

        //Create plan
        cufftHandle temp_plan;
        checkCudaErrors(cufftCreate(&temp_plan));
        size_t workSize;
        checkCudaErrors(cufftMakePlan1d(temp_plan, fft_size, CUFFT_C2C, 1, &workSize));

        //Run plan several times and time
        Stopwatch sw;
        for (int jj = 0; jj < iterations; jj++) {
            checkCudaErrors(cufftExecC2C(temp_plan, (cufftComplex*)buffer, (cufftComplex*)buffer, CUFFT_FORWARD));
        }
        cudaDeviceSynchronize();
        double runtime = sw.getElapsed();
        double time_per_call = runtime / iterations;

        //Print out duration and GFLOPs estimation
        double num_cpx_mpys = (fft_size / 2.0) * log2(fft_size);
        double num_cpx_adds = fft_size * log2(fft_size);
        printf("fft_size/time_per_call/GFLOPs: %i %f %f\n", fft_size, time_per_call, (6*num_cpx_mpys + 2*num_cpx_adds) / time_per_call / 1e9);

        //Destroy plan
        checkCudaErrors(cufftDestroy(temp_plan));

    }
}
