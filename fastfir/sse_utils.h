#pragma once

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
