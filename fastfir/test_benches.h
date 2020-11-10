#pragma once

#include "FastFirCPU1.h"
#include "FastFir.h"
#include "datplot_utils.h"
#include <vector>

void test_reference_design();

//Test compares results from passed FastFir to reference FastFirCPU1 (time-domain CPU version)
void validate_FastFir(FastFir* ff);

//Test varies processing parameters for passed FastFir and writes a summary output csv outlining
// performance metrics
void explore_FastFir(FastFir* ff,
	char* output_csv,
	std::vector<int> mask_sizes,
	std::vector<int> input_sizes,
	std::vector<int> buffers_per_call);