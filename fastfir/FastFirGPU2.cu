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

#include "FastFirGPU2.h"

#include "math_utils.h"

#include <algorithm>

namespace FFGPU2 {
    __global__ void cpxFloat2Half(float* input, half* output, int NN);
    __global__ void cpxBfloat162Float(nv_bfloat16* input, float* output, int NN);
    __global__ void cpxBfloat162Half(nv_bfloat16* input, half* output, int NN);
    __global__ void cpxHalf2Bfloat16(half* input, nv_bfloat16* output, int NN);
    __global__ void vectorCpxMpy(half* input1, half* input2, half* output, int NN);
    __global__ void vectorCpxScale(half* input1, half* output, half scale, int NN);
    __global__ void vectorCpxAdd(half* input1, half* input2, half* output, int NN);
}

FastFirGPU2::FastFirGPU2(float* mask, int mask_samps, int input_samps,
                         int buffers_per_call, bool contiguous)
    : FastFir(mask, mask_samps, input_samps, buffers_per_call, contiguous) {

    //Choose an FFT Size equal to next power of 2
    fft_size_ = FastFir::getFFTSize(mask_samps_, input_samps_);

    //Allocate device memory
    int total_output_samps = FastFir::getTotalOutputSamps();
    size_t h_bfloat16_buffer_bytes = sizeof(nv_bfloat16) * 2 * buffers_per_call * fft_size_;
    size_t d_bfloat16_buffer_bytes = sizeof(nv_bfloat16) * 2 * buffers_per_call_ * fft_size_;
    size_t d_io_buffer_bytes = sizeof(half) * 2 * buffers_per_call_ * fft_size_;
    size_t mask_buffer_bytes = sizeof(half) * 2 * fft_size_;
    HOST_MALLOC(&h_bfloat16_buffer_, h_bfloat16_buffer_bytes);
    DEVICE_MALLOC((void**)&d_bfloat16_buffer_, d_bfloat16_buffer_bytes);
    DEVICE_MALLOC((void**)&d_half_buffer_, d_io_buffer_bytes);
    DEVICE_MALLOC((void**)&d_mask_buffer_, mask_buffer_bytes);

    ////Initialize mask buffer using full precision, then convert to half

    //Copy input mask to temporary device buffer
    size_t temp_mask_bytes = sizeof(float) * 2 * fft_size_;
    size_t mask_bytes = sizeof(float) * 2 * mask_samps_;
    size_t non_mask_bytes = sizeof(float) * 2 * (fft_size_ - mask_samps_);
    float* temp_mask;
    DEVICE_MALLOC((void**)&temp_mask, temp_mask_bytes);
    checkCudaErrors(cudaMemcpy(temp_mask, mask, mask_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(&temp_mask[2 * mask_samps_], 0, non_mask_bytes));

    //Run full 32-bit precision fft
    cufftHandle temp_plan;
    checkCudaErrors(cufftCreate(&temp_plan));
    size_t workSize;
    long long workNN = fft_size_;
    checkCudaErrors(cufftXtMakePlanMany(temp_plan, 1, &workNN, NULL, 1, 1, CUDA_C_32F,
                                        NULL, 1, 1, CUDA_C_32F, 1, &workSize, CUDA_C_32F));
    checkCudaErrors(cufftXtExec(temp_plan, (void*)temp_mask, (void*)temp_mask, CUFFT_FORWARD));
    checkCudaErrors(cufftDestroy(temp_plan));

    //Copy into half buffer
    int tpb = getBestTPB(0);
    int num_blocks1 = getNumBlocks(tpb, fft_size_);
    FFGPU2::cpxFloat2Half << <num_blocks1, tpb >> > (temp_mask, d_mask_buffer_, fft_size_);
    cudaDeviceSynchronize();

    //Free temporary buffer
    DEVICE_FREE(temp_mask);

    //Default to buffers_per_call_, with a max of 4
    //Note: each one must allocate their own FFT working buffers!
    // todo: recommend checking GPU memory and warning/limiting here
    initProcStreams((std::min)(4, buffers_per_call_));

    //Create one event per processing buffer
    kernels_done_events_.resize(buffers_per_call_);
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        checkCudaErrors(cudaEventCreate(&kernels_done_events_[ii]));
    }

    //Execute plans at least once to ensure no first-call overhead
    checkCudaErrors(cudaMemset(d_half_buffer_, 0, d_io_buffer_bytes));
    for (int ii = 0; ii < (std::min)((int)cufft_plans_.size(), buffers_per_call_); ii++) {
        half* d_io_ptr = &d_half_buffer_[2 * ii * fft_size_];
        checkCudaErrors(cufftXtExec(cufft_plans_[ii], (void*)d_io_ptr, (void*)d_io_ptr, CUFFT_FORWARD));
        checkCudaErrors(cufftXtExec(cufft_plans_[ii], (void*)d_io_ptr, (void*)d_io_ptr, CUFFT_INVERSE));
    }
    cudaDeviceSynchronize();
}

FastFirGPU2::~FastFirGPU2() {
    //Destroy events
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        checkCudaErrors(cudaEventDestroy(kernels_done_events_[ii]));
    }

    //Destroy streams/plans
    initProcStreams(0);

    //Free device memory
    DEVICE_FREE(d_half_buffer_);
    DEVICE_FREE(d_mask_buffer_);

}

void FastFirGPU2::run(float* input, float* output) {
    nvtxRangePushA("FastFirGPU2::run");
    size_t input_bytes = sizeof(nv_bfloat16) * 2 * input_samps_;
    size_t non_input_bytes = sizeof(nv_bfloat16) * 2 * (fft_size_ - input_samps_);\
    int output_samps_0sided = FastFir::getOutputSampsNoTransient(mask_samps_, input_samps_);
    int output_samps_1sided = FastFir::getOutputSamps1Sided(mask_samps_, input_samps_);
    int output_samps_2sided = FastFir::getOutputSamps2Sided(mask_samps_, input_samps_);
    size_t output_bytes_0sided = sizeof(nv_bfloat16) * 2 * output_samps_0sided;
    size_t output_bytes_1sided = sizeof(nv_bfloat16) * 2 * output_samps_1sided;
    size_t output_bytes_2sided = sizeof(nv_bfloat16) * 2 * output_samps_2sided;
    int left_transient_samps = output_samps_1sided - output_samps_0sided;

    ////Determine kernal parameters
    int tpb = getBestTPB(0);
    //For kernels processing full fft size
    int num_blocks1 = getNumBlocks(tpb, fft_size_);
    //For kernels processing only transients
    int num_blocks2 = getNumBlocks(tpb, left_transient_samps);

    //Calculate fft scaling
    half scale = (((double)1.0) / fft_size_);

    //Output pointer movement depends on if we are using contiguous buffers
    //Note: with half types we are going to memcpy to second half of float output buffer
    int total_output_samps = FastFir::getTotalOutputSamps();
    nv_bfloat16* h_output_ptr = (nv_bfloat16*)h_bfloat16_buffer_;

    int num_proc_streams = (int)proc_streams_.size();
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        int proc_stream_index = ii % num_proc_streams;

        //Choose streams
        cudaStream_t proc_stream = proc_streams_[proc_stream_index];

        //Choose cufft plans
        cufftHandle proc_plan = cufft_plans_[proc_stream_index];

        //Note: we are going to transfer the nv_bfloat16 samples to the end of the float buffer
        nv_bfloat16* d_io_nv_bfloat16_ptr = &d_bfloat16_buffer_[2 * ii * fft_size_];
        half* d_io_ptr = &d_half_buffer_[2 * ii * fft_size_];

        //Convert input data from float to nv_bfloat16
        nv_bfloat16* h_io_nv_bfloat16_ptr = &h_bfloat16_buffer_[2 * ii * fft_size_];
        float* h_io_ptr = &input[2 * ii * input_samps_];
        nvtxRangePushA("cpu input conversion");
        cpxvec_float2bfloat16_avx(h_io_ptr, h_io_nv_bfloat16_ptr, input_samps_);
        nvtxRangePop();

        /*
        printf("float/nv_float16=%f/%f\n", h_io_ptr[0], (float)h_io_nv_bfloat16_ptr[0]);
        printf("float/nv_float16=%f/%f\n", h_io_ptr[1], (float)h_io_nv_bfloat16_ptr[1]);
        printf("float/nv_float16=%f/%f\n", h_io_ptr[2], (float)h_io_nv_bfloat16_ptr[2]);
        printf("float/nv_float16=%f/%f\n", h_io_ptr[3], (float)h_io_nv_bfloat16_ptr[3]);
        */


        //DEBUG
        /*
        nv_bfloat16* bfloat16_data;
        half* half_data;
        HOST_MALLOC(&bfloat16_data, 4 * sizeof(nv_bfloat16));
        HOST_MALLOC(&half_data, 4 * sizeof(half));
        */

        //Transfer1 : H->D : Move input samples to device and zero pad (transfers are nv_bfloat16 buffers)
        checkCudaErrors(cudaMemcpyAsync(d_io_nv_bfloat16_ptr, h_io_nv_bfloat16_ptr, input_bytes, cudaMemcpyHostToDevice, proc_stream));
        checkCudaErrors(cudaMemsetAsync(&d_io_nv_bfloat16_ptr[2 * input_samps_], 0, non_input_bytes, proc_stream));

        //Convert from bfloat16 to half
        FFGPU2::cpxBfloat162Half << <num_blocks1, tpb, 0, proc_stream >> > (d_io_nv_bfloat16_ptr, d_io_ptr, fft_size_);

        /*
        cudaMemcpy(bfloat16_data, d_io_nv_bfloat16_ptr, 4 * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaMemcpy(half_data, d_io_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("nv_float16/half=%f/%f\n", (float)bfloat16_data[0], (float)half_data[0]);
        printf("nv_float16/half=%f/%f\n", (float)bfloat16_data[1], (float)half_data[1]);
        printf("nv_float16/half=%f/%f\n", (float)bfloat16_data[2], (float)half_data[2]);
        printf("nv_float16/half=%f/%f\n", (float)bfloat16_data[3], (float)half_data[3]);
        */
        

        //Run fwd fft
        checkCudaErrors(cufftXtExec(proc_plan, (void*)d_io_ptr, (void*)d_io_ptr, CUFFT_FORWARD));

        /*
        cudaMemcpy(half_data, d_io_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("half0=%f\n", (float)half_data[0]);
        printf("half0=%f\n", (float)half_data[1]);
        printf("half0=%f\n", (float)half_data[2]);
        printf("half0=%f\n", (float)half_data[3]);
        */
        
        //Scale first so multiply does not overflow
        FFGPU2::vectorCpxScale << <num_blocks1, tpb, 0, proc_stream >> > (d_io_ptr, d_io_ptr, scale, fft_size_);
        checkCudaErrors(cudaPeekAtLastError());

        /*
        cudaMemcpy(half_data, d_io_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("half1_input=%f\n", (float)half_data[0]);
        printf("half1_input=%f\n", (float)half_data[1]);
        printf("half1_input=%f\n", (float)half_data[2]);
        printf("half1_input=%f\n", (float)half_data[3]);
        */

        /*
        cudaMemcpy(half_data, d_mask_buffer_, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("half1_mask=%f\n", (float)half_data[0]);
        printf("half1_mask=%f\n", (float)half_data[1]);
        printf("half1_mask=%f\n", (float)half_data[2]);
        printf("half1_mask=%f\n", (float)half_data[3]);
        */

        //Run cpx mpy/scaling kernel
        FFGPU2::vectorCpxMpy << <num_blocks1, tpb, 0, proc_stream >> > (d_io_ptr, d_mask_buffer_, d_io_ptr, fft_size_);
        checkCudaErrors(cudaPeekAtLastError());
        
        /*
        cudaMemcpy(half_data, d_io_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("half2=%f\n", (float)half_data[0]);
        printf("half2=%f\n", (float)half_data[1]);
        printf("half2=%f\n", (float)half_data[2]);
        printf("half2=%f\n", (float)half_data[3]);
        */
        

        //Run rev fft
        checkCudaErrors(cufftXtExec(proc_plan, (void*)d_io_ptr, (void*)d_io_ptr, CUFFT_INVERSE));

        /*
        cudaMemcpy(half_data, d_io_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("half3=%f\n", (float)half_data[0]);
        printf("half3=%f\n", (float)half_data[1]);
        printf("half3=%f\n", (float)half_data[2]);
        printf("half3=%f\n", (float)half_data[3]);
        */
        

        if (contiguous_) {
            //For contiguous, add in transient from previous kernels (need to wait until they are finished)
            // Note: for all buffers except the first
            if (ii != 0) {
                checkCudaErrors(cudaStreamWaitEvent(proc_stream, kernels_done_events_[ii - 1]));
                half* prev_d_io_ptr = &d_half_buffer_[2 * (ii - 1) * fft_size_];
                FFGPU2::vectorCpxAdd << <num_blocks2, tpb, 0, proc_stream >> > (d_io_ptr, &prev_d_io_ptr[2 * output_samps_1sided], d_io_ptr, left_transient_samps);
                checkCudaErrors(cudaPeekAtLastError());
            }
        }
        checkCudaErrors(cudaEventRecord(kernels_done_events_[ii], proc_stream));

        //Convert from half to bfloat16
        FFGPU2::cpxHalf2Bfloat16 << <num_blocks1, tpb, 0, proc_stream >> > (d_io_ptr, d_io_nv_bfloat16_ptr, fft_size_);

        /*
        cudaMemcpy(half_data, d_io_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        cudaMemcpy(bfloat16_data, d_io_nv_bfloat16_ptr, 4 * sizeof(half), cudaMemcpyDeviceToHost);
        printf("half/bfloat16=%f/%f\n", (float)half_data[0], (float)bfloat16_data[0]);
        printf("half/bfloat16=%f/%f\n", (float)half_data[1], (float)bfloat16_data[1]);
        printf("half/bfloat16=%f/%f\n", (float)half_data[2], (float)bfloat16_data[2]);
        printf("half/bfloat16=%f/%f\n", (float)half_data[3], (float)bfloat16_data[3]);
        */
        

        //Transfer2 : D->H : Move output samples to host
        if (!contiguous_) {
            //Simply move data to its respective output buffer
            checkCudaErrors(cudaMemcpyAsync(h_output_ptr, d_io_nv_bfloat16_ptr, output_bytes_2sided, cudaMemcpyDeviceToHost, proc_stream));

            h_output_ptr += 2 * output_samps_2sided;
        }
        else {
            //We need to add in overlaps
            if (ii != buffers_per_call_ - 1) {
                //Simply move the first data into output buffer
                checkCudaErrors(cudaMemcpyAsync(h_output_ptr, d_io_nv_bfloat16_ptr, output_bytes_1sided, cudaMemcpyDeviceToHost, proc_stream));
                h_output_ptr += 2 * output_samps_1sided;
            }
            else {
                //Copy full 2-sided result for last buffer
                checkCudaErrors(cudaMemcpyAsync(h_output_ptr, d_io_nv_bfloat16_ptr, output_bytes_2sided, cudaMemcpyDeviceToHost, proc_stream));
                h_output_ptr += 2 * output_samps_2sided;
            }

        }
    }

    //Synchronize all streams
    cudaDeviceSynchronize();

    //Convert all output data from bfloat16 to float
    nvtxRangePushA("cpu output conversion");
    h_output_ptr = (nv_bfloat16*)h_bfloat16_buffer_;
    cpxvec_bfloat162float_avx(h_output_ptr, output, total_output_samps);
    nvtxRangePop();

    /*
    printf("nv_float16/float=%f/%f\n", (float)h_output_ptr[0], output[0]);
    printf("nv_float16/float=%f/%f\n", (float)h_output_ptr[1], output[1]);
    printf("nv_float16/float=%f/%f\n", (float)h_output_ptr[2], output[2]);
    printf("nv_float16/float=%f/%f\n", (float)h_output_ptr[3], output[3]);
    */
    
    nvtxRangePop();
}


//Allows override of number of streams
void FastFirGPU2::setNumProcStreams(int num_streams) {
    initProcStreams(num_streams);
}

void FastFirGPU2::initProcStreams(int num_streams) {
    //De-allocate any currently created strams/plans
    if (proc_streams_.size() != 0) {
        for (int ii = 0; ii < proc_streams_.size(); ii++) {
            checkCudaErrors(cufftDestroy(cufft_plans_[ii]));
            checkCudaErrors(cudaStreamDestroy(proc_streams_[ii]));
        }
    }

    //Initialize new configuration
    cufft_plans_.resize(num_streams);
    proc_streams_.resize(num_streams);
    for (int ii = 0; ii < num_streams; ii++) {
        //Initialize streams
        checkCudaErrors(cudaStreamCreate(&proc_streams_[ii]));

        //Create cufft plans
        checkCudaErrors(cufftCreate(&cufft_plans_[ii]));
        size_t workSize;
        long long workNN = fft_size_;
        checkCudaErrors(cufftXtMakePlanMany(cufft_plans_[ii], 1, &workNN, NULL, 1, 1, CUDA_C_16F,
                                            NULL, 1, 1, CUDA_C_16F, 1, &workSize, CUDA_C_16F));

        //Associate streams to plans
        checkCudaErrors(cufftSetStream(cufft_plans_[ii], proc_streams_[ii]));
    }
}

namespace FFGPU2 {

    __global__ void cpxFloat2Half(float* input, half* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            //Convert each complex value from float to half
            int offset = 2 * ii;
            float* ptr1 = input + offset;//input location for this thread
            half* ptr2 = output + offset;//output location for this thread
            *ptr2 = *ptr1;
            *(ptr2 + 1) = *(ptr1 + 1);
        }
    }

    __global__ void cpxBfloat162Float(nv_bfloat16* input, float* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            //Convert each complex value from float to half
            int offset = 2 * ii;
            nv_bfloat16* ptr1 = input + offset;//input location for this thread
            float* ptr2 = output + offset;//output location for this thread
            *ptr2 = *ptr1;
            *(ptr2 + 1) = *(ptr1 + 1);
        }
    }

    __global__ void cpxBfloat162Half(nv_bfloat16* input, half* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            //Convert each complex value from float to half
            int offset = 2 * ii;
            nv_bfloat16* ptr1 = input + offset;//input location for this thread
            half* ptr2 = output + offset;//output location for this thread
            *(ptr2) = (float)(*ptr1);
            *(ptr2 + 1) = (float)*(ptr1 + 1);
        }
    }

    __global__ void cpxHalf2Bfloat16(half* input, nv_bfloat16* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            //Convert each complex value from float to half
            int offset = 2 * ii;
            half* ptr1 = input + offset;//input location for this thread
            nv_bfloat16* ptr2 = output + offset;//output location for this thread
            //Note: is this really the only way to do this?
            *(ptr2) = (float)(*ptr1);
            *(ptr2 + 1) = (float)*(ptr1 + 1);
        }
    }

    //Vectorized complex multiply
    __global__ void vectorCpxMpy(half* input1, half* input2, half* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            int offset = 2 * ii;
            half* ptr1 = input1 + offset;//input1 location for this thread
            half* ptr2 = input2 + offset;//input2 location for this thread
            half* ptr3 = output + offset;//output location for this thread

            half aa = *(ptr1);
            half bb = *(ptr1 + 1);
            half cc = *(ptr2);
            half dd = *(ptr2 + 1);

            *(ptr3) = aa * cc - bb * dd;
            *(ptr3 + 1) = aa * dd + bb * cc;
        }

    }

    __global__ void vectorCpxScale(half* input1, half* output, half scale, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            int offset = 2 * ii;
            half* ptr1 = input1 + offset;//input1 location for this thread
            half* ptr2 = output + offset;//output location for this thread

            half aa = *(ptr1);
            half bb = *(ptr1 + 1);

            *(ptr2) = aa * scale;
            *(ptr2 + 1) = bb * scale;
        }
    }

    __global__ void vectorCpxAdd(half* input1, half* input2, half* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            int offset = 2 * ii;
            half* ptr1 = input1 + offset;//input1 location for this thread
            half* ptr2 = input2 + offset;//input2 location for this thread
            half* ptr3 = output + offset;//output location for this thread

            half aa = *(ptr1);
            half bb = *(ptr1 + 1);
            half cc = *(ptr2);
            half dd = *(ptr2 + 1);

            *(ptr3) = aa + cc;
            *(ptr3 + 1) = bb + dd;
        }
    }

}