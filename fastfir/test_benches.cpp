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

//Test for comparing float-> half/half2/bfloat conversions
// Note: Run in NSIGHT Systems to compare timings
//
//... it would appear we need a faster conversion for any of this
// to be fast enough to integrate.
void test_conversion_performance() {

    printf("Running float-> half/half2/bfloat16 conversion...\n");

    //Input buffer
    float* buffer1;

    //Output buffers for explicit versions
    half* buffer2;
    half2* buffer3;
    nv_bfloat16* buffer4;
    nv_bfloat162* buffer5;

    //Output buffers for vectorized functions
    nv_bfloat16* buffer6;
    nv_bfloat16* buffer7;

    int iterations = 1000;
    int buffer_samps = 128*8192;

    double estimated_mem_bw = 28.8e9;//Bytes/sec
    double buffer_bytes = sizeof(float) * 2 * buffer_samps;
    double estimated_transfer_time = buffer_bytes / estimated_mem_bw;
    printf("Estimated mem time = %fus\n", estimated_transfer_time * 1e6);

    //Create input buffer
    HOST_MALLOC(&buffer1, 2 * buffer_samps * sizeof(float));

    //Create output buffers
    HOST_MALLOC(&buffer2, 2 * buffer_samps * sizeof(half));
    HOST_MALLOC(&buffer3, buffer_samps * sizeof(half2));
    HOST_MALLOC(&buffer4, 2 * buffer_samps * sizeof(nv_bfloat16));
    HOST_MALLOC(&buffer5, buffer_samps * sizeof(nv_bfloat162));

    //Vectorized versions
    HOST_MALLOC(&buffer6, 2*buffer_samps * sizeof(nv_bfloat16));
    HOST_MALLOC(&buffer7, 2*buffer_samps * sizeof(nv_bfloat16));

    //Convert float -> half
    memset(buffer1, 0, 2 * buffer_samps * sizeof(float));

    nvtxRangePushA("float -> half");
    float* ptr1 = buffer1;
    half* ptr2 = buffer2;
    for (int ii = 0; ii < buffer_samps; ii++) {
        *ptr2 = __float2half(*ptr1);
        *(ptr2+1) = __float2half(*(ptr1+1));
        ptr2 += 2;
        ptr1 += 2;
    }
    nvtxRangePop();


    //Convert float -> half2
    memset(buffer1, 0, 2 * buffer_samps * sizeof(float));

    nvtxRangePushA("float -> half");
    ptr1 = buffer1;
    half2* ptr3 = buffer3;
    for (int ii = 0; ii < buffer_samps; ii++) {
        *ptr3 = __floats2half2_rn(*ptr1, *(ptr1 + 1));
        ptr1 += 2;
        ptr3 += 1;
    }
    nvtxRangePop();

    //Convert float -> nv_bfloat16
    memset(buffer1, 0, 2 * buffer_samps * sizeof(float));

    nvtxRangePushA("float -> nv_bfloat16");
    ptr1 = buffer1;
    nv_bfloat16* ptr4 = buffer4;
    for (int ii = 0; ii < buffer_samps; ii++) {
        /*
        *(ptr4++) = __float2bfloat16_rz(*(ptr1++));
        *(ptr4++) = __float2bfloat16_rz(*(ptr1++));
        * */
        
        *ptr4 = __float2bfloat16_rz(*ptr1);
        *(ptr4 + 1) = __float2bfloat16_rz(*(ptr1 + 1));
        ptr4 += 2;
        ptr1 += 2;
        
    }
    nvtxRangePop();

    //Convert float -> nv_bfloat162
    memset(buffer1, 0, 2 * buffer_samps * sizeof(float));

    nvtxRangePushA("float -> nv_bfloat162");
    ptr1 = buffer1;
    nv_bfloat162* ptr5 = buffer5;
    for (int ii = 0; ii < buffer_samps; ii++) {
        *ptr5 = __floats2bfloat162_rn(*ptr1, *(ptr1 + 1));
        ptr1 += 2;
        ptr5 += 1;
    }
    nvtxRangePop();


    //Run scalar bfloat conversion
    memset(buffer1, 0, 2 * buffer_samps * sizeof(float));
    nvtxRangePushA("float2bfloat16_scalar");
    cpxvec_float2bfloat16_scalar(buffer1, buffer6, buffer_samps);
    nvtxRangePop();

    //Run avx bfloat conversion
    memset(buffer1, 0, 2 * buffer_samps * sizeof(float));
    nvtxRangePushA("float2bfloat16_avx");
    cpxvec_float2bfloat16_avx(buffer1, buffer7, buffer_samps);
    nvtxRangePop();

}

//Tests CUFFT (without memory transfers to/from GPU)
void test_cufft() {
    printf("Running test_cufft...\n");
    int iterations = 100;

    //Determine fft size parameters
    int min_pow = 8;
    int max_pow = 18;
    size_t max_fft_size = round(pow(2, max_pow));

    //Determine target memory size parameters
    size_t current_mem_size = 64 * 1024 * 1024;//In units of bytes
    int mem_doublings = 6;
    size_t max_mem_size = current_mem_size;
    for (int jj = 1; jj < mem_doublings; jj++) {
        max_mem_size *= 2;
    }

    //Allocate device memory for largest tranform
    float* buffer;
    printf("Running test_cufft, min_pow=%i, max_pow=%i, max_mem_size=%iMB\n", min_pow, max_pow, max_mem_size / (1024 * 1024));
    size_t max_buffer_bytes = sizeof(float) * 2 * max_fft_size;
    int max_num_ffts = (std::max)((size_t)1, max_mem_size / max_buffer_bytes);
    DEVICE_MALLOC((void**)&buffer, max_num_ffts * max_buffer_bytes);
    cudaMemset(buffer, 1, 2 * max_fft_size * sizeof(float));


    for (int jj = 0; jj < mem_doublings; jj++) {
        printf("Running for target mem size=%iMB\n", (int)(current_mem_size / (1024 * 1024)));
        for (int ii = min_pow; ii <= max_pow; ii++) {
            //Determine cufft parameters
            int fft_size = (int)round(pow(2, ii));
            size_t buffer_bytes = sizeof(float) * 2 * fft_size;
            int num_ffts = (std::max)((size_t)1, current_mem_size / buffer_bytes);

            //Create plan
            cufftHandle temp_plan;
            checkCudaErrors(cufftCreate(&temp_plan));
            size_t workSize;
            //checkCudaErrors(cufftMakePlan1d(temp_plan, fft_size, CUFFT_C2C, num_ffts, &workSize));
            checkCudaErrors(cufftMakePlanMany(temp_plan, 1, &fft_size, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, num_ffts, &workSize));

            //Run plan several times and clock total run time
            Stopwatch sw;
            for (int jj = 0; jj < iterations; jj++) {
                checkCudaErrors(cufftExecC2C(temp_plan, (cufftComplex*)buffer, (cufftComplex*)buffer, CUFFT_FORWARD));
            }
            cudaDeviceSynchronize();
            double runtime = sw.getElapsed();
            double time_per_call = runtime / iterations;

            //Print out duration and GFLOPs estimation
            double num_cpx_mpys = num_ffts * (fft_size / 2.0) * log2(fft_size);
            double num_cpx_adds = num_ffts * fft_size * log2(fft_size);
            printf("    fft_size/num_ffts/time_per_call/GFLOPs: %i %i %f %f\n", fft_size, num_ffts, time_per_call, (6 * num_cpx_mpys + 2 * num_cpx_adds) / time_per_call / 1e9);

            //Destroy plan
            checkCudaErrors(cufftDestroy(temp_plan));

        }
        current_mem_size *= 2;
    }
}
