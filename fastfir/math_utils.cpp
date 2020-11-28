#include <random>

//Routine that uses built-in C++ guassian generator to generate White Gaussian Noise
//
//Note: STD is std for both real and complex
void generate_wgn_cf(double mean, double std, float* output, int output_samps) {
    std::default_random_engine gen;
    std::normal_distribution<float> dist(mean, std);

    for (int ii = 0; ii < output_samps; ii++) {
        output[2 * ii] = dist(gen);
        output[2 * ii + 1] = dist(gen);
    }
}