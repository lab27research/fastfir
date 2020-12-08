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

#include "cuda_utils.h"

//Performs a complex multiply assuming a single
// interleaved complex sample in each array
template<class T>
inline void cpxmpy(T* input1, T* input2, T* output) {
    T aa = *(input1);
    T bb = *(input1 + 1);
    T cc = *(input2);
    T dd = *(input2 + 1);

    //(aa+bbj)*(cc+ddj) = (aa*cc - bb*dd) + (aa*dd + bb*cc)j
    *(output) = (aa * cc - bb * dd);
    *(output + 1) = (aa * dd + bb * cc);
}

//Performs a complex add assuming a single
// interleaved complex sample in each array
template<class T>
inline void cpxadd(T* input1, T* input2, T* output) {
    T aa = *(input1);
    T bb = *(input1 + 1);
    T cc = *(input2);
    T dd = *(input2 + 1);

    //(aa+bbj) + (cc+ddj) = (aa+cc) + (bb+dd)j
    *(output) = (aa + cc);
    *(output + 1) = (bb + dd);
}

//Standard non-SSE complex multiply
//(aa+bbj)*(cc+ddj) = (aa*cc-bb*dd) + j(aa*dd + bb*cc)
inline void complex_madd(float aa, float bb, float cc, float dd, float* out) {
    out[0] += aa * cc - bb * dd;
    out[1] += aa * dd + bb * cc;
}

//Noise generation
void generate_wgn_cf(double mean, double std, float* output, int output_samps);


//Routines for  float->bfloat16->float conversion
void test_bfloat16_conversions();
void cpxvec_float2bfloat16_scalar(float* input, nv_bfloat16* output, int num_samps);
void cpxvec_float2bfloat16_avx(float* input, nv_bfloat16* output, int num_samps);
//Note: input and output must be disjoint for bfloat16->float conversions
void cpxvec_bfloat162float_scalar(nv_bfloat16* input, float* output, int num_samps);
void cpxvec_bfloat162float_avx(nv_bfloat16* input, float* output, int num_samps);
