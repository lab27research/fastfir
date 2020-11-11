#pragma once

////MEMORY ALLOCATION ROUTINES

#include <malloc.h>

//Note: This file is WINDOWS only for now!

//Some day I'll have AVX-512
#define CPU_ALIGNMENT 64

//For Linux
//#define ALIGNED_MALLOC(ptr,size) posix_memalign((void**) &ptr,CPU_ALIGNMENT,size);
//#define ALIGNED_FREE(ptr) free(ptr);

//For Windows
#define ALIGNED_MALLOC(ptr,size) ptr = (float*) _aligned_malloc(size,CPU_ALIGNMENT)
#define ALIGNED_FREE(ptr) _aligned_free(ptr);


////VECTORIZED MATH routines

//Standard non-SSE complex multiply
//(aa+bbj)*(cc+ddj) = (aa*cc-bb*dd) + j(aa*dd + bb*cc)
inline void complex_madd(float aa, float bb, float cc, float dd, float* out) {
	out[0] += aa * cc - bb * dd;
	out[1] += aa * dd + bb * cc;
}



////NOISE GENERATION

void generate_wgn_cf(double mean, double std, float* output, int output_samps);
