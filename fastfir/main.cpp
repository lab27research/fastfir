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

    //Write output files to verify test case
    unit_test<FastFirCPU1>("input1.csv", "mask1.csv", "output1.csv");
    unit_test<FastFirCPU2>("input2.csv", "mask2.csv", "output2.csv");

    //Generates random inputs, tests each implementation,
    // quantifies differences
    validate<FastFirCPU1, FastFirCPU2>(256, 1024, 9);
    validate<FastFirCPU2, FastFirGPU1>(256, 1024, 9);

}

void test() {
    int buffers_per_call = 10;
    int input_samps = 1024;
    int mask_samps = 256;
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* input;
    float* mask;
    float* output;
    ALIGNED_MALLOC(mask, 2 * mask_samps * buffers_per_call * sizeof(float));
    ALIGNED_MALLOC(input, 2 * input_samps * buffers_per_call * sizeof(float));
    ALIGNED_MALLOC(output, 2 * output_samps * buffers_per_call * sizeof(float));

    //Create CPU-Based FIR Filter
    FastFirCPU2 ff1(mask, mask_samps, input_samps, buffers_per_call, false);

    //Create input source
    double snr = 10;
    unsigned int samp0 = 5;
    ImpulseSource is(snr, samp0);

    //This is where we need to add test bench
    Stopwatch sw;
    int total_runs = 10;
    for (int ii = 0; ii < total_runs / buffers_per_call; ii++) {

        //Fill input buffers
        for (int jj = 0; jj < buffers_per_call; jj++) {
            is.getBuffer(&input[2 * jj * input_samps], input_samps);
        }

        //Run algorithm
        ff1.run(input, output);
    }
    double runtime = sw.getElapsed();
    printf("Completed in %.9f seconds\n", runtime);
    printf("Average time per run: %.9f\n", runtime / total_runs);

    datplot_write_cf("input.csv", input, input_samps);
    datplot_write_cf("output.csv", output, output_samps);

}