#pragma once

#include <cstdio>
#include <math.h>

//Write a Complex Float csv file that can be easily plotted with datplot
//Columns: time,real,imag,mag,phase
void datplot_write_cf(char* filename, float* data, int data_samps, double xstart = 0.0, double xdelta = 1.0);