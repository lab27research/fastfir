#include "FastFirGPU2.h"

#include "math_utils.h"

#include <algorithm>

namespace FFGPU2 {
    __global__ void cpxHalf2Float(half* input, float* output, int NN);
    __global__ void cpxbfloat162Float(nv_bfloat16* input, float* output, int NN);
    __global__ void vectorCpxMpy(float* input1, float* input2, float* output, int NN);
    __global__ void vectorCpxScale(float* input1, float* output, float scale, int NN);
    __global__ void vectorCpxAdd(float* input1, float* input2, float* output, int NN);
}

FastFirGPU2::FastFirGPU2(float* mask, int mask_samps, int input_samps,
                         int buffers_per_call, bool contiguous)
    : FastFir(mask, mask_samps, input_samps, buffers_per_call, contiguous) {

    //Choose an FFT Size equal to next power of 2
    fft_size_ = FastFir::getFFTSize(mask_samps_, input_samps_);

    //Allocate device memory
    size_t h_io_nv_bfloat16_buffer_bytes = sizeof(nv_bfloat16) * 2 * buffers_per_call * fft_size_;
    size_t d_io_nv_bfloat16_buffer_bytes = sizeof(nv_bfloat16) * 2 * buffers_per_call_ * fft_size_;
    size_t d_io_buffer_bytes = sizeof(float) * 2 * buffers_per_call_ * fft_size_;
    size_t mask_buffer_bytes = sizeof(float) * 2 * fft_size_;
    HOST_MALLOC(&h_io_nv_bfloat16_buffer_, h_io_nv_bfloat16_buffer_bytes);
    DEVICE_MALLOC((void**)&d_io_nv_bfloat16_buffer_, d_io_nv_bfloat16_buffer_bytes);
    DEVICE_MALLOC((void**)&d_io_buffer_, d_io_buffer_bytes);
    DEVICE_MALLOC((void**)&d_mask_buffer_, mask_buffer_bytes);

    //Initialize mask buffer
    size_t mask_bytes = sizeof(float) * 2 * mask_samps_;
    size_t non_mask_bytes = sizeof(float) * 2 * (fft_size_ - mask_samps_);
    checkCudaErrors(cudaMemcpy(d_mask_buffer_, mask, mask_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(&d_mask_buffer_[2 * mask_samps_], 0, non_mask_bytes));

    cufftHandle temp_plan;
    checkCudaErrors(cufftCreate(&temp_plan));
    size_t workSize;
    checkCudaErrors(cufftMakePlan1d(temp_plan, fft_size_, CUFFT_C2C, 1, &workSize));
    checkCudaErrors(cufftExecC2C(temp_plan, (cufftComplex*)d_mask_buffer_, (cufftComplex*)d_mask_buffer_, CUFFT_FORWARD));
    checkCudaErrors(cufftDestroy(temp_plan));

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
    checkCudaErrors(cudaMemset(d_io_buffer_, 0, d_io_buffer_bytes));
    for (int ii = 0; ii < (std::min)((int)cufft_plans_.size(), buffers_per_call_); ii++) {
        float* d_io_ptr = &d_io_buffer_[2 * ii * fft_size_];
        checkCudaErrors(cufftExecC2C(cufft_plans_[ii], (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_FORWARD));
        checkCudaErrors(cufftExecC2C(cufft_plans_[ii], (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_INVERSE));
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
    DEVICE_FREE(d_io_buffer_);
    DEVICE_FREE(d_mask_buffer_);

}

void FastFirGPU2::run(float* input, float* output) {
    nvtxRangePushA("FastFirGPU2::run");
    size_t input_bytes = sizeof(nv_bfloat16) * 2 * input_samps_;
    size_t non_input_bytes = sizeof(nv_bfloat16) * 2 * (fft_size_ - input_samps_);
    int output_samps_0sided = FastFir::getOutputSampsNoTransient(mask_samps_, input_samps_);
    int output_samps_1sided = FastFir::getOutputSamps1Sided(mask_samps_, input_samps_);
    int output_samps_2sided = FastFir::getOutputSamps2Sided(mask_samps_, input_samps_);
    size_t output_bytes_0sided = sizeof(float) * 2 * output_samps_0sided;
    size_t output_bytes_1sided = sizeof(float) * 2 * output_samps_1sided;
    size_t output_bytes_2sided = sizeof(float) * 2 * output_samps_2sided;
    int left_transient_samps = output_samps_1sided - output_samps_0sided;

    ////Determine kernal parameters
    int tpb = getBestTPB(0);
    //For kernels processing full fft size
    int num_blocks1 = getNumBlocks(tpb, fft_size_);
    //For kernels processing only transients
    int num_blocks2 = getNumBlocks(tpb, left_transient_samps);

    //Calculate fft scaling
    float scale = (float)(((double)1.0) / fft_size_);

    //Output pointer movement depends on if we are using contiguous buffers
    float* h_output_ptr = output;

    int num_proc_streams = (int)proc_streams_.size();
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        int proc_stream_index = ii % num_proc_streams;

        //Choose streams
        cudaStream_t proc_stream = proc_streams_[proc_stream_index];

        //Choose cufft plans
        cufftHandle proc_plan = cufft_plans_[proc_stream_index];

        //Note: we are going to transfer the nv_bfloat16 samples to the end of the float buffer
        nv_bfloat16* d_io_nv_bfloat16_ptr = &d_io_nv_bfloat16_buffer_[2 * ii * fft_size_];
        float* d_io_ptr = &d_io_buffer_[2 * ii * fft_size_];

        //Convert input data from float to nv_bfloat16
        nv_bfloat16* h_io_nv_bfloat16_ptr = &h_io_nv_bfloat16_buffer_[2 * ii * input_samps_];
        float* h_io_ptr = &input[2 * ii * input_samps_];

        nvtxRangePushA("cpu conversion");
        cpxvec_float2bfloat16_avx(h_io_ptr, h_io_nv_bfloat16_ptr, input_samps_);
        nvtxRangePop();

        //Transfer1 : H->D : Move input samples to device and zero pad (transfers are nv_bfloat16 buffers)
        checkCudaErrors(cudaMemcpyAsync(d_io_nv_bfloat16_ptr, h_io_nv_bfloat16_ptr, input_bytes, cudaMemcpyHostToDevice, proc_stream));
        checkCudaErrors(cudaMemsetAsync(&d_io_nv_bfloat16_ptr[2 * input_samps_], 0, non_input_bytes, proc_stream));

        //Run kernel to convert input from nv_bfloat16 to float
        FFGPU2::cpxbfloat162Float << <num_blocks1, tpb, 0, proc_stream >> > (d_io_nv_bfloat16_ptr, d_io_ptr, fft_size_);

        //Run fwd fft
        checkCudaErrors(cufftExecC2C(proc_plan, (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_FORWARD));

        //Run cpx mpy/scaling kernel
        FFGPU2::vectorCpxMpy << <num_blocks1, tpb, 0, proc_stream >> > (d_io_ptr, d_mask_buffer_, d_io_ptr, fft_size_);
        checkCudaErrors(cudaPeekAtLastError());

        FFGPU2::vectorCpxScale << <num_blocks1, tpb, 0, proc_stream >> > (d_io_ptr, d_io_ptr, scale, fft_size_);
        checkCudaErrors(cudaPeekAtLastError());

        //Run rev fft
        checkCudaErrors(cufftExecC2C(proc_plan, (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_INVERSE));

        if (contiguous_) {
            //For contiguous, add in transient from previous kernels (need to wait until they are finished)
            // Note: for all buffers except the first
            if (ii != 0) {
                checkCudaErrors(cudaStreamWaitEvent(proc_stream, kernels_done_events_[ii - 1]));
                float* prev_d_io_ptr = &d_io_buffer_[2 * (ii - 1) * fft_size_];
                FFGPU2::vectorCpxAdd << <num_blocks2, tpb, 0, proc_stream >> > (d_io_ptr, &prev_d_io_ptr[2 * output_samps_1sided], d_io_ptr, left_transient_samps);
                checkCudaErrors(cudaPeekAtLastError());
            }
        }
        checkCudaErrors(cudaEventRecord(kernels_done_events_[ii], proc_stream));


        //Transfer2 : D->H : Move output samples to host
        if (!contiguous_) {
            //Simply move data to its respective output buffer
            checkCudaErrors(cudaMemcpyAsync(h_output_ptr, d_io_ptr, output_bytes_2sided, cudaMemcpyDeviceToHost, proc_stream));

            h_output_ptr += 2 * output_samps_2sided;
        }
        else {
            //We need to add in overlaps
            if (ii != buffers_per_call_ - 1) {
                //Simply move the first data into output buffer
                checkCudaErrors(cudaMemcpyAsync(h_output_ptr, d_io_ptr, output_bytes_1sided, cudaMemcpyDeviceToHost, proc_stream));
                h_output_ptr += 2 * output_samps_1sided;
            }
            else {
                //Copy full 2-sided result for last buffer
                checkCudaErrors(cudaMemcpyAsync(h_output_ptr, d_io_ptr, output_bytes_2sided, cudaMemcpyDeviceToHost, proc_stream));
                h_output_ptr += 2 * output_samps_2sided;
            }

        }
    }

    //Synchronize all streams
    cudaDeviceSynchronize();
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
        checkCudaErrors(cufftMakePlan1d(cufft_plans_[ii], fft_size_, CUFFT_C2C, 1, &workSize));

        //Associate streams to plans
        checkCudaErrors(cufftSetStream(cufft_plans_[ii], proc_streams_[ii]));
    }
}

namespace FFGPU2 {

    __global__ void cpxHalf2Float(half* input, float* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            //Convert each complex value from float to half
            int offset = 2 * ii;
            half* ptr1 = input + offset;//input location for this thread
            float* ptr2 = output + offset;//output location for this thread
            *ptr2 = *ptr1;
            *(ptr2 + 1) = *(ptr1 + 1);
        }
    }

    __global__ void cpxbfloat162Float(nv_bfloat16* input, float* output, int NN) {
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

    //Vectorized complex multiply
    __global__ void vectorCpxMpy(float* input1, float* input2, float* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            int offset = 2 * ii;
            float* ptr1 = input1 + offset;//input1 location for this thread
            float* ptr2 = input2 + offset;//input2 location for this thread
            float* ptr3 = output + offset;//output location for this thread

            float aa = *(ptr1);
            float bb = *(ptr1 + 1);
            float cc = *(ptr2);
            float dd = *(ptr2 + 1);

            *(ptr3) = aa * cc - bb * dd;
            *(ptr3 + 1) = aa * dd + bb * cc;
        }

    }

    __global__ void vectorCpxScale(float* input1, float* output, float scale, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            int offset = 2 * ii;
            float* ptr1 = input1 + offset;//input1 location for this thread
            float* ptr2 = output + offset;//output location for this thread

            float aa = *(ptr1);
            float bb = *(ptr1 + 1);

            *(ptr2) = aa * scale;
            *(ptr2 + 1) = bb * scale;
        }
    }

    __global__ void vectorCpxAdd(float* input1, float* input2, float* output, int NN) {
        //One dimensional grid/block configuration
        int ii = blockIdx.x * blockDim.x + threadIdx.x;

        if (ii < NN) {
            int offset = 2 * ii;
            float* ptr1 = input1 + offset;//input1 location for this thread
            float* ptr2 = input2 + offset;//input2 location for this thread
            float* ptr3 = output + offset;//output location for this thread

            float aa = *(ptr1);
            float bb = *(ptr1 + 1);
            float cc = *(ptr2);
            float dd = *(ptr2 + 1);

            *(ptr3) = aa + cc;
            *(ptr3 + 1) = bb + dd;
        }
    }

}