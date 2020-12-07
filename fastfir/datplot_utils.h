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

#include <cstdio>
#include <math.h>
#include <vector>

#include "test_benches.h"

//Datplot is free plotting software useful for quick debugging.  It ingests csv files and
// is able to perform basic 2D plotting
//The following functions generate files that can be easily ingested into Datplot

//Write a Complex Float csv file that can be easily plotted with datplot
//Columns: index,time,real,imag,mag,phase
void datplot_write_cf(char* filename, float* data, int data_samps, double xstart = 0.0, double xdelta = 1.0);

//Write FastFir result data to output file
//Columns: index,mask_samps,input_samps,time_per_buffer,time_flops_per_buffer,freq_flops_per_buffer,
// estimated_time_fps,estimated_freq_fps
void dataplot_write_ffresults(char* filename, std::vector<FFResult>& result_list);