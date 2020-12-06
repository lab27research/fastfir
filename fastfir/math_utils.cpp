#include "math_utils.h"

#include "avx_utils.h"

#include <random>

#include <immintrin.h>

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

//CUDA Intrinsic Version
void cpxvec_float2bfloat16_scalar(float* input, nv_bfloat16* output, int num_samps) {
    float* ptr1 = input;
    nv_bfloat16* ptr2 = output;

    for (int ii = 0; ii < num_samps; ii++) {
        *(ptr2++) = __float2bfloat16_rz(*(ptr1++));
        *(ptr2++) = __float2bfloat16_rz(*(ptr1++));
    }
}

//Unit test to verify shuffle/permute commands function as expected
void test_shuffle() {

    float val1 = 1.123456789;
    nv_bfloat16 val2 = __float2bfloat16_rz(val1);
    printf("float/bfloat16: %f/%f\n", val1, __bfloat162float(val2));

    char* ptr1 = (char*)&val1;
    for (int ii = 0; ii < 4; ii++) {
        printf("%i/", *(ptr1++));
    }
    printf("\n");
    ptr1 = (char*)&val2;
    for (int ii = 0; ii < 2; ii++) {
        printf("%i/", *(ptr1++));
    }
    printf("\n");

    char* input_data;
    HOST_MALLOC(&input_data, 32 * sizeof(char));
    for (int ii = 0; ii < 32; ii++) {
        input_data[ii] = ii;
    }
    __m256i reg1 = _mm256_load_si256((__m256i*) input_data);
    print_packed_chars(reg1);
    __m256i reg2 = _mm256_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                    2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    print_packed_chars(reg2);
    __m256i reg3 = _mm256_shuffle_epi8(reg1, reg2);
    print_packed_chars(reg3);
    __m256i reg4 = _mm256_permute4x64_epi64(reg3, _MM_SHUFFLE(3, 3, 2, 0));
    print_packed_chars(reg4);

    HOST_FREE(input_data);
}

//AVX Version of vectorized float -> bfloat16 conversion
void cpxvec_float2bfloat16_avx(float* input, nv_bfloat16* output, int num_samps) {
    int iterations = (num_samps - 4) / 4;//Process 4 complex samples at a time (8 floats),
                                               //excluding last set of 4 samples
    int leftovers = num_samps - (iterations * 4);

    //Preload shuffle mask for initial 8-bit shuffle
    //This mask turns:
    // LSBs                                                                                       MSBs
    //[Float 0   ][Float 1   ][Float 2   ][Float 3   ][Float 4   ][Float 5   ][Float 6   ][Float 7   ]
    //[LSB ...MSB][LSB ...MSB][LSB ...MSB][LSB ...MSB][LSB ...MSB][LSB ...MSB][LSB ...MSB][LSB ...MSB]
    //  0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/
    //...into:
    //  2/ 3/ 6/ 7/10/11/14/15/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/18/19/22/23/26/27/30/31/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/
    //[Hlf0][Hlf1][Hlf2][Hlf3]                        [Hlf4][Hlf5][Hlf6][Hlf7]
    //[L /M]
    //(Grabbing MSB of each dataset assuming 32-bit float)
    __m256i shuffle_reg = _mm256_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                           2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);

    float* in_ptr = input;
    nv_bfloat16* out_ptr = output;
    for (int ii = 0; ii < iterations; ii++) {
        //Load 4 samples at a time, treating it as a packed integer
        //  0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/
        __m256i reg1 = _mm256_load_si256((__m256i*) in_ptr);

        //Shuffle MSBs of each float into LSBs of respective 128-bit lanes
        //  2/ 3/ 6/ 7/10/11/14/15/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/18/19/22/23/26/27/30/31/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/
        reg1 = _mm256_shuffle_epi8(reg1, shuffle_reg);

        //Pack all of these values into the first 128 bit lane
        //  2/ 3/ 6/ 7/10/11/14/15/18/19/22/23/26/27/30/31/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/ 0/
        reg1 = _mm256_permute4x64_epi64(reg1, _MM_SHUFFLE(3, 3, 2, 0));

        //Store entire 256bit result to output array
        //Note: must use unaligned store, as output is moving by 8x16 = 128bits
        _mm256_storeu_si256((__m256i*) out_ptr, reg1);

        //Move pointers by 4 complex samples
        in_ptr += 8;
        out_ptr += 8;
    }

    //Clean up the rest with the non-AVX function
    cpxvec_float2bfloat16_scalar(in_ptr, out_ptr, leftovers);
}

void cpxvec_bfloat162float_scalar(nv_bfloat16* input, float* output, int num_samps) {
    nv_bfloat16* ptr1 = input;
    float* ptr2 = output;

    for (int ii = 0; ii < num_samps; ii++) {
        *(ptr2++) = *(ptr1++);
        *(ptr2++) = *(ptr1++);
    }
}

void cpxvec_bfloat162float_avx(nv_bfloat16* input, float* output, int num_samps) {
    int iterations = num_samps / 8;//Process 8 complex samples at a time (16 nv_bfloat16 values = 256bits)
    int leftovers = num_samps - (iterations * 8);

    //Converts:
    //  0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/10/11/12/13/14/15/ 8/ 9/10/11/12/13/14/15/
    //To (X=zero valued):
    //  X/ X/ 0/ 1/ -1/ -1/ 2/ 3/ X/ X/ 4/ 5/ X/ X/ 6/ 7/ X/ X/ 8/ 9/ X/ X/10/11/ X/ X/12/13/ X/ X/14/15/
    __m256i shuffle_reg = _mm256_setr_epi8(-1, -1, 0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7,
                                           -1, -1, 0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 6, 7);

    nv_bfloat16* in_ptr = input;
    float* out_ptr = output;
    for (int ii = 0; ii < iterations; ii++) {
        //Load 4 samples at a time, treating it as a packed integer
        //  0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/
        __m256i reg1 = _mm256_load_si256((__m256i*) in_ptr);

        //Goal: achieve the follow two 256-bit register setups (X are 0-valued 8-bit entries)
        //  X/ X/ 0/ 1/ X/ X/ 2/ 3/ X/ X/ 4/ 5/ X/ X/ 6/ 7/ X/ X/ 8/ 9/ X/ X/10/11/ X/ X/12/13/ X/ X/14/15/
        //  X/ X/16/17/ X/ X/18/19/ X/ X/20/21/ X/ X/22/23/ X/ X/24/25/ X/ X/26/27/ X/ X/28/29/ X/ X/30/21/

        //Perform 64-bit shuffle to get appropriate values in the lanes
        //  0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/
        //  0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 0/ 1/ 2/ 3/ 4/ 5/ 6/ 7/ 8/ 9/10/11/12/13/14/15/ 8/ 9/10/11/12/13/14/15/
        // 16/17/18/19/20/21/22/23/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/24/25/26/27/28/29/30/31/
        __m256i reg2 = _mm256_permute4x64_epi64(reg1, _MM_SHUFFLE(1, 1, 0, 0));
        __m256i reg3 = _mm256_permute4x64_epi64(reg1, _MM_SHUFFLE(3, 3, 2, 2));

        //Now perform in-lane 8bit shuffle insert 0s and place bytes in correct places
        reg2 = _mm256_shuffle_epi8(reg2, shuffle_reg);
        reg3 = _mm256_shuffle_epi8(reg3, shuffle_reg);

        //Store a total of 512 bits into output array
        _mm256_store_si256((__m256i*) out_ptr, reg2);
        _mm256_store_si256((__m256i*) (out_ptr + 8), reg3);

        //Move pointers by 8 complex samples
        in_ptr += 16;
        out_ptr += 16;
    }

    //Clean up the rest with the non-AVX function
    cpxvec_bfloat162float_scalar(in_ptr, out_ptr, leftovers);
}

//Test that directly compares non-AVX vs AVX bfloat16 conversion functions
void test_bfloat16_conversions() {

    float* input1;
    float* input2;
    float* input3;
    nv_bfloat16* output1;
    nv_bfloat16* output2;
    int input_samps = 16;
    HOST_MALLOC(&input1, 2 * input_samps * sizeof(float));
    HOST_MALLOC(&input2, 2 * input_samps * sizeof(float));
    HOST_MALLOC(&input3, 2 * input_samps * sizeof(float));
    HOST_MALLOC(&output1, 2 * input_samps * sizeof(nv_bfloat16));
    HOST_MALLOC(&output2, 2 * input_samps * sizeof(nv_bfloat16));

    generate_wgn_cf(0.0, 0.1, input1, input_samps);

    cpxvec_float2bfloat16_scalar(input1, output1, input_samps);
    cpxvec_float2bfloat16_avx(input1, output2, input_samps);
    cpxvec_bfloat162float_scalar(output1, input2, input_samps);
    cpxvec_bfloat162float_avx(output1, input3, input_samps);

    printf("Comparing both version:\n");
    for (int ii = 0; ii < input_samps; ii++) {
        printf("float/bfloat16/bfloat16/float/float: %f/%f/%f/%f\n",
               input1[2 * ii],
               __bfloat162float(output1[2 * ii]),
               __bfloat162float(output2[2 * ii]),
               input2[2 * ii],
               input3[2 * ii]);
        printf("float/bfloat16/bfloat16/float/float: %f/%f/%f/%f\n",
               input1[2 * ii + 1],
               __bfloat162float(output1[2 * ii + 1]),
               __bfloat162float(output2[2 * ii + 1]),
               input2[2 * ii + 1],
               input3[2 * ii + 1]);
    }
}