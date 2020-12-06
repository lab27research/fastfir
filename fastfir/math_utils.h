#pragma once

#include "cuda_utils.h"

////VECTORIZED MATH routines

//Standard non-SSE complex multiply
//(aa+bbj)*(cc+ddj) = (aa*cc-bb*dd) + j(aa*dd + bb*cc)
inline void complex_madd(float aa, float bb, float cc, float dd, float* out) {
    out[0] += aa * cc - bb * dd;
    out[1] += aa * dd + bb * cc;
}

////NOISE GENERATION
void generate_wgn_cf(double mean, double std, float* output, int output_samps);


//half/bfloat16 type conversion
void test_bfloat16_conversions();
void cpxvec_float2bfloat16_scalar(float* input, nv_bfloat16* output, int num_samps);
void cpxvec_float2bfloat16_avx(float* input, nv_bfloat16* output, int num_samps);
//Note: input and output must be disjoint for bfloat16->float conversions
void cpxvec_bfloat162float_scalar(nv_bfloat16* input, float* output, int num_samps);
void cpxvec_bfloat162float_avx(nv_bfloat16* input, float* output, int num_samps);
