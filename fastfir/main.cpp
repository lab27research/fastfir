
#include "FastFirCPU1.h"
#include "FastFirCPU2.h"
#include "FastFirGPU1.h"
#include "FastFirGPU2.h"
#include "math_utils.h"
#include "Stopwatch.h"
#include "datplot_utils.h"
#include "test_benches.h"

void nsight_systems_test();
void nsight_compute_test();

//FastFir is a set of CPU and GPU classes used to implement floating point complex convolution
//
//FastFirCPU1 - CPU time domain reference implementation
//FastFirCPU2 - CPU frequency domain implementation that uses FFTW (single threaded)
//FastFirGPU1 - GPU frequency domain implementation that uses CUFFT
//  -->All implementations use the Abstract Base Class: FastFir, which defines the
//     interferface for the constructor and run functions
//
//Several tests defined in test_benches.h/.cpp:
//
//unit_test1 - Writes a csv after processing short known set of sequences (you can verify manually)
//validate - Compares the results feeding random inputs to each implementation, prints info
//            quantifying differences
//get_time_per_call - Runs implementation several times using random input, estimates time per call
//                     of the run() function
//explore - Estimates GFLOPs achieved for each provided runtime config

int main() {

    /*
    unit_test2<FastFirCPU1>("input1.csv", "mask1.csv", "output1.csv");
    unit_test2<FastFirCPU2>("input2.csv", "mask2.csv", "output2.csv");
    unit_test2<FastFirGPU1>("input3.csv", "mask3.csv", "output3.csv");
    unit_test2<FastFirGPU2>("input4.csv", "mask4.csv", "output4.csv");
    return 1;
    */

    /*
    test_bfloat16_conversions();
    return 1;
    */

    /*
    test_conversion_performance();
    return 1;
    */

    /*
    for (int ii = 4; ii < 20; ii++) {
        int mask_samps = pow(2, ii);
        int input_samps = mask_samps * 4;
        printf("Running for %i/%i\n", mask_samps, input_samps);
        validate<FastFirGPU1, FastFirGPU2>(mask_samps,input_samps,10);
    }
    return 1;
    */



    //For debugging with NSight Systems
    /*
    nsight_systems_test();
    return 1;
    */

    //For debugging with NSight Compute
    /*
    nsight_compute_test();
    return 1;
    */

    ///All unit, validation, and performance tests

    //Unit test to determine cufft flops if not bound by H->D and D->H transfers
    //test_cufft();

    /*
    //Run unit tests that can be verified externally
    unit_test2<FastFirCPU1>("input1.csv", "mask1.csv", "output1.csv");
    unit_test2<FastFirCPU2>("input2.csv", "mask2.csv", "output2.csv");
    unit_test2<FastFirGPU1>("input3.csv", "mask3.csv", "output3.csv");
    unit_test2<FastFirGPU2>("input4.csv", "mask4.csv", "output4.csv");

    //Compare implementations and understand output difference
    validate<FastFirCPU1, FastFirCPU2>(256, 1024, 9);
    validate<FastFirCPU2, FastFirGPU1>(256, 1024, 9);
    validate<FastFirGPU1, FastFirGPU2>(256, 1024, 9);

    //Tests per-call performance (small workload)
    int mask_samps = 256;
    int input_samps = 1024;
    int buffers_per_call = 10;
    int iterations = 10;
    double pc1 = get_time_per_call<FastFirCPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc2 = get_time_per_call<FastFirCPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc3 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc4 = get_time_per_call<FastFirGPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    printf("Small buffer timings (us): %f/%f/%f/%f\n", pc1 * 1e6, pc2 * 1e6, pc3 * 1e6, pc4 * 1e6);

    //Test per-call performance (large workload)
    mask_samps = 128 * 1024;
    input_samps = 512 * 1024;
    buffers_per_call = 10;
    iterations = 10;
    double pc5 = get_time_per_call<FastFirCPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc6 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc7 = get_time_per_call<FastFirGPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    printf("Large buffer timings (us): %f/%f/%f\n", pc5 * 1e6, pc6 * 1e6, pc7 * 1e6);
    */


    //Run "explore" command to test a variety of input sizes
    std::vector<FFConfig> configs;
    size_t target_memsize = round(0.25 * 1024 * 1024 * 1024);//Target around a gig total buffer size
    int min_pow = 12;
    int max_pow = 27;
    int explore_iterations = 4;
    //Note: Use for full results (ran overnight)
    //int min_pow = 8;
    //int max_pow = 27;
    //int iterations = 10;
    for (int ii = min_pow; ii <= max_pow; ii++) {
        FFConfig cc;
        cc.input_samps = (int)pow(2, ii);
        cc.mask_samps = cc.input_samps / 4;
        cc.buffer_per_call = (std::max)(1, (int)round(target_memsize / (sizeof(float) * 2 * cc.input_samps)));

        cc.contiguous = true;
        cc.iterations = explore_iterations;
        configs.push_back(cc);
    }
    explore<FastFirGPU2>("explore_gpu2.csv", configs);
    explore<FastFirGPU1>("explore_gpu1.csv", configs);
    explore<FastFirCPU2>("explore_cpu2.csv", configs);

}

void nsight_systems_test() {
    std::vector<FFConfig> configs;
    FFConfig cc;

    //Run one small sized, non-contiguous
    cc.input_samps = 8192;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = false;
    cc.iterations = 4;
    configs.push_back(cc);

    //Run one small sized, contiguous
    cc.input_samps = 8192;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = true;
    cc.iterations = 4;
    configs.push_back(cc);

    //Run one large sized, non-contiguous
    cc.input_samps = 1 * 1024 * 1024;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = false;
    cc.iterations = 4;
    configs.push_back(cc);

    //Run one large sized, contiguous
    cc.input_samps = 1 * 1024 * 1024;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = true;
    cc.iterations = 4;
    configs.push_back(cc);

    explore<FastFirGPU2>("nsight_compute_test.csv", configs);
}

void nsight_compute_test() {
    int mask_samps = 256;
    int input_samps = 1024;
    int buffers_per_call = 10;
    int iterations = 10;
    double pc3 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
}