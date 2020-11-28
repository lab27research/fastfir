#include "FastFirCPU1.h"
#include "FastFirCPU2.h"
#include "FastFirGPU1.h"
#include "ImpulseSource.h"
#include "math_utils.h"
#include "Stopwatch.h"
#include "datplot_utils.h"
#include "test_benches.h"

//Simulation to compare CPU vs GPU implementations for FFT-Based FIR filtering
//
//Assumptions: 1.) Input buffers cannot guarantee alignment or to be page locked
//                 (copies are necessary for alignment guarantees)
//
//

#include "add_kernel.h"
int main() {

    //Run unit tests that can be verified externally
    unit_test1<FastFirCPU1>("input1.csv", "mask1.csv", "output1.csv");
    unit_test2<FastFirCPU1>("input2.csv", "mask2.csv", "output2.csv");

    //Compare implementations and understand output difference
    validate<FastFirCPU1, FastFirCPU2>(256, 1024, 9);
    validate<FastFirCPU2, FastFirGPU1>(256, 1024, 9);

    //Tests per-call performance (small workload)
    int mask_samps = 256;
    int input_samps = 1024;
    int buffers_per_call = 10;
    int iterations = 100;
    double pc1 = get_time_per_call<FastFirCPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc2 = get_time_per_call<FastFirCPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc3 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    printf("pc1=%f, pc2=%f, pc3=%f\n", pc1, pc2, pc3);

    //Test per-call performance (large workload)
    mask_samps = 128 * 1024;
    input_samps = 512 * 1024;
    buffers_per_call = 10;
    iterations = 100;
    double pc4 = get_time_per_call<FastFirCPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc5 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    printf("pc4=%f, pc5=%f\n", pc4, pc5);

}
