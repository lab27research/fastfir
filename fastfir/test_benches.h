#pragma once

#include "FastFir.h"
#include <vector>

void explore_FastFir(FastFir* ff,
	std::vector<int> mask_sizes,
	std::vector<int> input_sizes,
	std::vector<int> buffers_per_call);