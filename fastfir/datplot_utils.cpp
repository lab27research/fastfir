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
}

