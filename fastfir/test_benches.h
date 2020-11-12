#pragma once

#include "FastFirCPU1.h"
#include "FastFir.h"
#include "datplot_utils.h"
#include <vector>

//Unit tests
void test_generate_wgn_cf();
void test_reference_design();

//Validation test that compares two FastFilt implementations
//
//Prints the SNR of the result 10*log10( pow(res1) / pow(res1-res2) )
template <class ff_type1, class ff_type2>
void validate() {

	//Setup all input/output buffers
	//Note: allocate large enough for both contiguous and non-contiguous
	int buffers_per_call = 9;
	int input_samps = 1024;
	int mask_samps = 256;
	int output_samps = FastFir::getOutputSamps(mask_samps, input_samps);
	float* input;
	float* mask;
	float* output1;
	float* output2;
	ALIGNED_MALLOC(mask, 2 * mask_samps * buffers_per_call * sizeof(float));
	ALIGNED_MALLOC(input, 2 * input_samps * buffers_per_call * sizeof(float));
	ALIGNED_MALLOC(output1, 2 * output_samps * buffers_per_call * sizeof(float));
	ALIGNED_MALLOC(output2, 2 * output_samps * buffers_per_call * sizeof(float));

	//Generate input and mask, create filters for both contiguous and non-contiguous
	generate_wgn_cf(0.5, 0.1, input, output_samps);
	ff_type1 ff1(mask, mask_samps, input_samps, buffers_per_call, false);
	ff_type2 ff2(mask, mask_samps, input_samps, buffers_per_call, false);
	ff_type1 ff3(mask, mask_samps, input_samps, buffers_per_call, true);
	ff_type2 ff4(mask, mask_samps, input_samps, buffers_per_call, true);

	//Run both contiguous
	ff1.run(input, output1);
	ff2.run(input, output2);

	//Compute and print SNR
	double accum1 = 0.0;
	double accum2 = 0.0;
	for (int ii = 0; ii < ff1.getTotalOutputSamps(); ii++) {
		float aa = output1[2 * ii];
		float bb = output1[2 * ii + 1];
		float cc = aa - output2[2 * ii];
		float dd = bb - output2[2 * ii + 1];
		//Accumulate energy from first output
		accum1 += sqrt(aa * aa + bb * bb);
		accum2 += sqrt(cc * cc + dd * dd);
	}
	printf("Non-contiguous SNR: %f\n", 10.0 * log10(accum1 / accum2));

	////Run for contiguous ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
}

//Test varies processing parameters for passed FastFir and writes a summary output csv outlining
// performance metrics
template <class ff_type1>
void explore(char* output_csv,
	std::vector<int> mask_sizes,
	std::vector<int> input_sizes,
	std::vector<int> buffers_per_call) {
}