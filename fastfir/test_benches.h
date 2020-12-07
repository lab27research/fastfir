/*
* Copyright 2020 Curtis Rhodes
*
* This file is part of Fastfir.
*
* Fastfir is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Fastfir is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with Fastfir.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "FastFir.h"
#include "cuda_utils.h"
#include "Stopwatch.h"

#include <vector>
#include <string>
using namespace std;

//Unit tests
void test_generate_wgn_cf();
void test_conversion_performance();
void test_cufft();

//A unit test that processes a short sequence (that can be verified by hand)
template <class ff_type>
void unit_test1(string input_csv, string mask_csv, string output_csv)
{
    printf("Running unit_test1 for %s, outputs at %s/%s/%s\n", typeid(ff_type).name(), input_csv.c_str(), mask_csv.c_str(), output_csv.c_str());

    //Create mask/input/output buffers
    const int mask_samps = 2;
    const int input_samps = 4;
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* mask;
    float* input;
    float* output;
    HOST_MALLOC(&mask, 2 * mask_samps * sizeof(float));
    HOST_MALLOC(&input, 2 * input_samps * sizeof(float));
    HOST_MALLOC(&output, 2 * output_samps * sizeof(float));
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

    //Free memory
    HOST_FREE(mask);
    HOST_FREE(input);
    HOST_FREE(output);

}

//A unit test that uses correlation of known random sequence to verify FIR implementation
// (can verified expected peak by hand and see that peak in each buffer)
//Note: resulting correlation peaks should have magnitude equal to mask_samps
template<class ff_type>
void unit_test2(string input_csv, string mask_csv, string output_csv) {

    printf("Running unit_test2 for %s, outputs at %s/%s/%s\n", typeid(ff_type).name(), input_csv.c_str(), mask_csv.c_str(), output_csv.c_str());

    int buffers_per_call = 10;
    int input_samps = 1024;
    int mask_samps = 256;
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* input;
    float* flipped_mask;
    float* output;
    HOST_MALLOC(&flipped_mask, 2 * mask_samps * sizeof(float));
    HOST_MALLOC(&input, 2 * input_samps * buffers_per_call * sizeof(float));
    HOST_MALLOC(&output, 2 * output_samps * buffers_per_call * sizeof(float));

    //Populate every inpt buffer with the same values
    generate_wgn_cf(0.0, 0.001*sqrt(2.0) / 2.0, input, input_samps);
    for (int ii = 1; ii < buffers_per_call; ii++) {
        memcpy(&input[2 * ii * input_samps], input, 2 * input_samps * sizeof(float));
    }
    //Flipping and conjugating mask turns convolution into correlation
    for (int ii = 0; ii < mask_samps; ii++) {
        flipped_mask[2 * ii] = input[2 * (mask_samps - 1 - ii)];
        flipped_mask[2 * ii + 1] = -input[2 * (mask_samps - 1 - ii) + 1];
    }

    //Create FIR Filter and run algorithm
    ff_type ff1(flipped_mask, mask_samps, input_samps, buffers_per_call, false);
    ff1.run(input, output);

    //Write output files (should contain periodic correlation peaks)
    datplot_write_cf((char*)mask_csv.c_str(), flipped_mask, mask_samps, 0, 1);
    datplot_write_cf((char*)input_csv.c_str(), input, input_samps, 0, 1);
    datplot_write_cf((char*)output_csv.c_str(), output, ff1.getTotalOutputSamps(), 0, 1);

    //Free memory
    HOST_FREE(flipped_mask);
    HOST_FREE(input);
    HOST_FREE(output);
}

//Validation test that compares two FastFilt implementations
//
//Prints the SNR of the result 10*log10( pow(res1) / pow(res1-res2) )
template <class ff_type1, class ff_type2>
void validate(int mask_samps, int input_samps, int buffers_per_call) {
    printf("Running validation test %s vs %s...\n", typeid(ff_type1).name(), typeid(ff_type2).name());

    //Setup all input/output buffers
    //Note: allocate large enough for both contiguous and non-contiguous
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* input;
    float* mask;
    float* output1;
    float* output2;
    float* output3;
    float* output4;
    HOST_MALLOC(&mask, 2 * mask_samps * sizeof(float));
    HOST_MALLOC(&input, 2 * input_samps * buffers_per_call * sizeof(float));
    HOST_MALLOC(&output1, 2 * output_samps * buffers_per_call * sizeof(float));
    HOST_MALLOC(&output2, 2 * output_samps * buffers_per_call * sizeof(float));
    HOST_MALLOC(&output3, 2 * output_samps * buffers_per_call * sizeof(float));
    HOST_MALLOC(&output4, 2 * output_samps * buffers_per_call * sizeof(float));

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
    //datplot_write_cf("temp1.csv", output1, ff1.getTotalOutputSamps(), 0, 1);
    //datplot_write_cf("temp2.csv", output2, ff2.getTotalOutputSamps(), 0, 1);
    //datplot_write_cf("temp3.csv", output3, ff3.getTotalOutputSamps(), 0, 1);
    //datplot_write_cf("temp4.csv", output4, ff4.getTotalOutputSamps(), 0, 1);

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
        accum1 += aa * aa + bb * bb;//Sum power of first input
        accum2 += cc * cc + dd * dd;//Sum power of difference
    }
    for (int ii = 0; ii < ff3.getTotalOutputSamps(); ii++) {
        float aa = output3[2 * ii];
        float bb = output3[2 * ii + 1];
        float cc = aa - output4[2 * ii];
        float dd = bb - output4[2 * ii + 1];
        accum3 += aa * aa + bb * bb;//Sum power of first input
        accum4 += cc * cc + dd * dd;//Sum power of difference
    }
    printf("Non-contiguous SNR: %f\n", 10.0 * log10(accum1 / accum2));
    printf("Contiguous SNR: %f\n", 10.0 * log10(accum3 / accum4));

    //Free memory
    HOST_FREE(mask);
    HOST_FREE(input);
    HOST_FREE(output1);
    HOST_FREE(output2);
    HOST_FREE(output3);
    HOST_FREE(output4);
}

//Test varies processing parameters for passed FastFir and writes a summary output csv outlining
// performance metrics
struct FFConfig {
    int mask_samps;
    int input_samps;
    int buffer_per_call;
    bool contiguous;
    int iterations;
};

struct FFResult {
    //Input parameters
    FFConfig config;

    //Output parameters
    double time_per_buffer;
    double time_flops_per_buffer;
    double freq_flops_per_buffer;
    double time_fps;
    double freq_fps;
};


#include "datplot_utils.h"

template <class ff_type1>
void explore(char* output_csv,
             vector<FFConfig>& config_list) {

    printf("Running explore for %s\n", typeid(ff_type1).name());

    //Run all specified processing
    vector<FFResult> results;
    for (int ii = 0; ii < config_list.size(); ii++) {
        int mask_samps = config_list[ii].mask_samps;
        int input_samps = config_list[ii].input_samps;
        int buffers_per_call = config_list[ii].buffer_per_call;
        bool contiguous = config_list[ii].contiguous;
        int iterations = config_list[ii].iterations;
        printf("config: %i %i %i %i %i...", mask_samps, input_samps, buffers_per_call, contiguous, iterations);

        //Run processing and store results
        FFResult res;
        res.config = config_list[ii];
        res.time_per_buffer = get_time_per_call<ff_type1>(mask_samps, input_samps, buffers_per_call, contiguous, iterations) / buffers_per_call;
        res.time_flops_per_buffer = FastFir::getTimeDomainFLOPs(mask_samps, input_samps);
        res.freq_flops_per_buffer = FastFir::getFreqDomainFLOPs(mask_samps, input_samps);
        res.time_fps = res.time_flops_per_buffer / res.time_per_buffer;
        res.freq_fps = res.freq_flops_per_buffer / res.time_per_buffer;
        results.push_back(res);

        printf("%f GFLOPs/sec\n", res.freq_fps/1e9);
    }

    //Write output data
    dataplot_write_ffresults(output_csv, results);
}

template<class ff_type>
double get_time_per_call(int mask_samps, int input_samps, int buffers_per_call, bool contiguous, int iterations) {
    int output_samps = FastFir::getOutputSamps2Sided(mask_samps, input_samps);
    float* input;
    float* mask;
    float* output;
    size_t mask_bytes = sizeof(float) * 2 * mask_samps;
    size_t input_bytes = sizeof(float) * 2 * input_samps * buffers_per_call;
    size_t output_bytes = sizeof(float) * 2 * output_samps * buffers_per_call;
    HOST_MALLOC(&mask, mask_bytes);
    HOST_MALLOC(&input, input_bytes);
    HOST_MALLOC(&output, output_bytes);

    //Populate bogus mask and input
    memset(mask, 0, 2 * mask_samps);
    memset(input, 0, 2 * input_samps * buffers_per_call * sizeof(float));

    //Create FIR Filter
    ff_type ff1(mask, mask_samps, input_samps, buffers_per_call, contiguous);

    //This is where we need to add test bench
    Stopwatch sw;
    for (int ii = 0; ii < iterations; ii++) {

        //Run algorithm
        ff1.run(input, output);
    }
    double runtime = sw.getElapsed();

    //Free memory
    HOST_FREE(mask);
    HOST_FREE(input);
    HOST_FREE(output);

    return runtime / iterations;
}
