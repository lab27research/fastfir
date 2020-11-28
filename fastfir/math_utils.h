#pragma once

////VECTORIZED MATH routines

//Standard non-SSE complex multiply
//(aa+bbj)*(cc+ddj) = (aa*cc-bb*dd) + j(aa*dd + bb*cc)
inline void complex_madd(float aa, float bb, float cc, float dd, float* out) {
    out[0] += aa * cc - bb * dd;
    out[1] += aa * dd + bb * cc;
}

////NOISE GENERATION
void generate_wgn_cf(double mean, double std, float* output, int output_samps);
