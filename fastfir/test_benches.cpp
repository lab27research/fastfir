#include "test_benches.h"

void test_reference_design()
{
	float mask[2 * 2] = { 1,0.5,1,0.5 };
	float input[2 * 4] = { 1,0,2,0,3,0,4,0 };
	float output[2 * (4 + 2 - 1)];

	FastFirCPU1 ff(mask, 2, 4);
	int mask_samps = 2;
	int input_samps = 4;
	int output_samps = ff.getTotalOutputSamps();

	ff.run(input, output);

	datplot_write_cf("input.csv", input, input_samps, 0, 1);
	datplot_write_cf("mask.csv", mask, mask_samps, 0, 1);
	datplot_write_cf("output.csv", output, output_samps, 0, 1);


}

void validate_FastFir(FastFir* ff)
{
}

void explore_FastFir(FastFir* ff, std::vector<int> mask_sizes, std::vector<int> input_sizes, std::vector<int> buffers_per_call)
{

}