#pragma once

#include <cstdio>
#include <math.h>
#include <vector>

#include "test_benches.h"

//Write a Complex Float csv file that can be easily plotted with datplot
//Columns: index,time,real,imag,mag,phase
void datplot_write_cf(char* filename, float* data, int data_samps, double xstart = 0.0, double xdelta = 1.0);

//Write FastFir result data to output file
//Columns: index,mask_samps,input_samps,time_per_buffer,time_flops_per_buffer,freq_flops_per_buffer,
// estimated_time_fps,estimated_freq_fps
void dataplot_write_ffresults(char* filename, std::vector<FFResult>& result_list);