#include "FastFirCPU1.h"
#include "FastFirCPU2.h"
#include "FastFirGPU1.h"
#include "ImpulseSource.h"
#include "math_utils.h"
#include "Stopwatch.h"
#include "datplot_utils.h"
#include "test_benches.h"

void nsight_compute_test() {
    std::vector<FFConfig> configs;
    FFConfig cc;

    //Run one small sized, non-contiguous
    cc.input_samps = 8192;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = false;
    cc.iterations = 4;
    configs.push_back(cc);

    //Run one small sized, contiguous
    cc.input_samps = 8192;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = true;
    cc.iterations = 4;
    configs.push_back(cc);

    //Run one large sized, non-contiguous
    cc.input_samps = 1 * 1024 * 1024;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = false;
    cc.iterations = 4;
    configs.push_back(cc);

    //Run one large sized, contiguous
    cc.input_samps = 1 * 1024 * 1024;
    cc.mask_samps = cc.input_samps / 4;
    cc.buffer_per_call = 10;
    cc.contiguous = true;
    cc.iterations = 4;
    configs.push_back(cc);

    explore<FastFirGPU1>("nsight_compute_test.csv", configs);
}

#include "add_kernel.h"
int main() {

    nsight_compute_test();
    return 1;

    //Unit test to determine cufft flops if not bound by H->D and D-> transfers
    test_cufft();

    //Run unit tests that can be verified externally
    unit_test1<FastFirCPU1>("input1.csv", "mask1.csv", "output1.csv");
    unit_test2<FastFirCPU1>("input2.csv", "mask2.csv", "output2.csv");

    //Compare implementations and understand output difference
    validate<FastFirCPU1, FastFirCPU2>(256, 1024, 9);
    validate<FastFirCPU2, FastFirGPU1>(256, 1024, 9);

    //Tests per-call performance (small workload)
    int mask_samps = 256;
    int input_samps = 1024;
    int buffers_per_call = 10;
    int iterations = 10;
    double pc1 = get_time_per_call<FastFirCPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc2 = get_time_per_call<FastFirCPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc3 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    printf("pc1=%f, pc2=%f, pc3=%f\n", pc1, pc2, pc3);

    //Test per-call performance (large workload)
    mask_samps = 128 * 1024;
    input_samps = 512 * 1024;
    buffers_per_call = 10;
    iterations = 10;
    double pc4 = get_time_per_call<FastFirCPU2>(mask_samps, input_samps, buffers_per_call, true, iterations);
    double pc5 = get_time_per_call<FastFirGPU1>(mask_samps, input_samps, buffers_per_call, true, iterations);
    printf("pc4=%f, pc5=%f\n", pc4, pc5);


    //Run "explore" command to test a variety of input sizes
    std::vector<FFConfig> configs;
    size_t target_memsize = 0.25 * 1024 * 1024 * 1024;
    int min_pow = 22;
    int max_pow = 29;
    int explore_iterations = 4;
    //Note: Use for full results (ran overnight)
    //int min_pow = 8;
    //int max_pow = 27;
    //int iterations = 10;
    for (int ii = min_pow; ii <= max_pow; ii++) {
        FFConfig cc;
        cc.input_samps = pow(2, ii);
        cc.mask_samps = cc.input_samps / 4;
        cc.buffer_per_call = std::max(1, (int)round(target_memsize / (sizeof(float) * 2 * cc.input_samps)));

        cc.contiguous = false;
        cc.iterations = explore_iterations;
        configs.push_back(cc);
    }
    explore<FastFirGPU1>("explore2.csv", configs);
    explore<FastFirCPU2>("explore1.csv", configs);

}
