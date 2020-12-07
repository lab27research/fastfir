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

#include "datplot_utils.h"

void datplot_write_cf(char* filename, float* data, int data_samps, double xstart, double xdelta)
{
    FILE* fid = fopen(filename, "w");

    fprintf(fid, "index,time,real,imag,mag,phase\n");

    for (int ii = 0; ii < data_samps; ii++) {
        double time = xstart + ii * xdelta;
        double real = data[2 * ii];
        double imag = data[2 * ii + 1];
        double mag = sqrt(real * real + imag * imag);
        double phase;
        if (imag == 0.0 && real == 0.0) {
            phase = 0.0;
        }
        else {
            phase = atan2(imag, real);
        }
        fprintf(fid, "%i", ii);
        fprintf(fid, ",%.16e", time);
        fprintf(fid, ",%.16e", real);
        fprintf(fid, ",%.16e", imag);
        fprintf(fid, ",%.16e", mag);
        fprintf(fid, ",%.16e", phase);
        fprintf(fid, "\n");

    }

    fclose(fid);
}

void dataplot_write_ffresults(char* filename, std::vector<FFResult>& result_list) {
    FILE* fid = fopen(filename, "w");

    fprintf(fid, "index,mask_samps,input_samps,time_per_buffer,time_flops_per_buffer,freq_flops_per_buffer,estimated_time_fps,estimated_freq_fps\n");

    for (int ii = 0; ii < result_list.size(); ii++) {
        fprintf(fid, "%i", ii);
        fprintf(fid, ",%i", result_list[ii].config.mask_samps);
        fprintf(fid, ",%i", result_list[ii].config.input_samps);
        fprintf(fid, ",%.16e", result_list[ii].time_per_buffer);
        fprintf(fid, ",%.16e", result_list[ii].time_flops_per_buffer);
        fprintf(fid, ",%.16e", result_list[ii].freq_flops_per_buffer);
        fprintf(fid, ",%.16e", result_list[ii].time_fps);
        fprintf(fid, ",%.16e", result_list[ii].freq_fps);
        fprintf(fid, "\n");
    }

    fclose(fid);
}

