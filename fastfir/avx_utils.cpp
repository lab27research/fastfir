#include "avx_utils.h"

#include "cuda_utils.h"

//Prints the contents of a 256-bit integer register
// assuming it contains packed bfloat16s
void print_packed_bfloat16s(__m256i reg) {
    nv_bfloat16* buffer;
    HOST_MALLOC(&buffer, 16 * sizeof(nv_bfloat16));
    _mm256_store_si256((__m256i*) buffer, reg);
    for (int ii = 0; ii < 16; ii++) {
        printf("%f/", __bfloat162float(buffer[ii]));
    }
    printf("\n");
    HOST_FREE(buffer);
}

//Prints the contents of a 256-bit integer register
// assuming it contains packed chars
void print_packed_chars(__m256i reg) {
    char* buffer;
    HOST_MALLOC(&buffer, 32 * sizeof(char));
    _mm256_store_si256((__m256i*) buffer, reg);
    for (int ii = 0; ii < 32; ii++) {
        printf("%2i/", buffer[ii]);
    }
    printf("\n");
    HOST_FREE(buffer);
}