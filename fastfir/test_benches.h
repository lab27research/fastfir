#pragma once

#include "FastFirCPU1.h"
#include "FastFir.h"
#include "datplot_utils.h"
#include <vector>

//Unit tests
void test_generate_wgn_cf();
void test_reference_design();

#include <string>
using namespace std;

template <class ff_type>
void unit_test(string input_csv, string mask_csv, string output_csv)
{
    //Create mask/input/output buffers
    const int mask_samps = 2;
    const int input_samps = 4;
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* mask;
    float* input;
    float* output;
    ALIGNED_MALLOC(mask, 2 * mask_samps * sizeof(float));
    ALIGNED_MALLOC(input, 2 * input_samps * sizeof(float));
    ALIGNED_MALLOC(output, 2 * output_samps * sizeof(float));
    float fill1[2 * mask_samps] = { 1,0.5,1,0.5 };
    float fill2[2 * input_samps] = { 1,0,2,0,3,0,4,0 };
    memcpy(mask, fill1, 2 * mask_samps * sizeof(float));
    memcpy(input, fill2, 2 * input_samps * sizeof(float));

    //Create FastFir and run on the single buffer
    ff_type ff(mask, mask_samps, input_samps, 1, false);
    ff.run(input, output);

    //Write to datplot files
    datplot_write_cf((char*)input_csv.c_str(), input, input_samps, 0, 1);
    datplot_write_cf((char*)mask_csv.c_str(), mask, mask_samps, 0, 1);
    datplot_write_cf((char*)output_csv.c_str(), output, output_samps, 0, 1);

}

//Validation test that compares two FastFilt implementations
//
//Prints the SNR of the result 10*log10( pow(res1) / pow(res1-res2) )
template <class ff_type1, class ff_type2>
void validate(int mask_samps, int input_samps, int buffers_per_call) {

    //Setup all input/output buffers
    //Note: allocate large enough for both contiguous and non-contiguous
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* input;
    float* mask;
    float* output1;
    float* output2;
    float* output3;
    float* output4;
    ALIGNED_MALLOC(mask, 2 * mask_samps * sizeof(float));
    ALIGNED_MALLOC(input, 2 * input_samps * buffers_per_call * sizeof(float));
    ALIGNED_MALLOC(output1, 2 * output_samps * buffers_per_call * sizeof(float));
    ALIGNED_MALLOC(output2, 2 * output_samps * buffers_per_call * sizeof(float));
    ALIGNED_MALLOC(output3, 2 * output_samps * buffers_per_call * sizeof(float));
    ALIGNED_MALLOC(output4, 2 * output_samps * buffers_per_call * sizeof(float));

    //Generate input and mask, create filters for both contiguous and non-contiguous
    generate_wgn_cf(0.5, 0.1, mask, mask_samps);
    generate_wgn_cf(0.5, 0.1, input, input_samps * buffers_per_call);

    ff_type1 ff1(mask, mask_samps, input_samps, buffers_per_call, false);
    ff_type2 ff2(mask, mask_samps, input_samps, buffers_per_call, false);
    ff_type1 ff3(mask, mask_samps, input_samps, buffers_per_call, true);
    ff_type2 ff4(mask, mask_samps, input_samps, buffers_per_call, true);

    //Run both contiguous
    ff1.run(input, output1);
    ff2.run(input, output2);
    ff3.run(input, output3);
    ff4.run(input, output4);

    //DEBUG
    datplot_write_cf("temp1.csv", output1, ff1.getTotalOutputSamps(), 0, 1);
    datplot_write_cf("temp2.csv", output2, ff2.getTotalOutputSamps(), 0, 1);
    datplot_write_cf("temp3.csv", output3, ff3.getTotalOutputSamps(), 0, 1);
    datplot_write_cf("temp4.csv", output4, ff4.getTotalOutputSamps(), 0, 1);

    //Compute and print SNR
    double accum1 = 0.0;
    double accum2 = 0.0;
    double accum3 = 0.0;
    double accum4 = 0.0;
    for (int ii = 0; ii < ff1.getTotalOutputSamps(); ii++) {
        float aa = output1[2 * ii];
        float bb = output1[2 * ii + 1];
        float cc = aa - output2[2 * ii];
        float dd = bb - output2[2 * ii + 1];
        accum1 += sqrt(aa * aa + bb * bb);//Sum power of first input
        accum2 += sqrt(cc * cc + dd * dd);//Sum power of difference
    }
    for (int ii = 0; ii < ff3.getTotalOutputSamps(); ii++) {
        float aa = output3[2 * ii];
        float bb = output3[2 * ii + 1];
        float cc = aa - output4[2 * ii];
        float dd = bb - output4[2 * ii + 1];
        accum3 += sqrt(aa * aa + bb * bb);//Sum power of first input
        accum4 += sqrt(cc * cc + dd * dd);//Sum power of difference
    }
    printf("Non-contiguous SNR: %f\n", 10.0 * log10(accum1 / accum2));
    printf("Contiguous SNR: %f\n", 10.0 * log10(accum3 / accum4));
}

//Test varies processing parameters for passed FastFir and writes a summary output csv outlining
// performance metrics
template <class ff_type1>
void explore(char* output_csv,
             std::vector<int> mask_sizes,
             std::vector<int> input_sizes,
             std::vector<int> buffers_per_call) {
}