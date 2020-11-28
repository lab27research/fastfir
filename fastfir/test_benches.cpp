#include "test_benches.h"
#include "math_utils.h"

void test_generate_wgn_cf() {
    const int output_samps = 1000;
    float* output;
    HOST_MALLOC(&output, 2 * output_samps * sizeof(float));
    generate_wgn_cf(0.5, 0.1, output, output_samps);
    datplot_write_cf("noise.csv", output, output_samps, 0, 1);
}
