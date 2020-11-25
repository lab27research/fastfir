#include "FastFirGPU1.h"

FastFirGPU1::FastFirGPU1(float* mask, int mask_samps, int input_samps,
                         int buffers_per_call, bool contiguous)
    : FastFir(mask, mask_samps, input_samps, buffers_per_call, contiguous) {

    //Choose an FFT Size equal to next power of 2
    fft_size_ = FastFir::getFFTSize(mask_samps_, input_samps_);

    //Allocate device memory
    DEVICE_MALLOC((void**)&d_io_buffer_, 2 * buffers_per_call_ * fft_size_ * sizeof(float));
    DEVICE_MALLOC((void**)&d_mask_buffer_, 2 * fft_size_ * sizeof(float));

    //Initialize mask buffer
    cudaMemcpy(d_mask_buffer_, mask, 2 * mask_samps_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(&d_mask_buffer_[2 * mask_samps_], 0, 2 * (fft_size_ - mask_samps_) * sizeof(float));

    cufftHandle temp_plan;
    cufftCreate(&temp_plan);
    size_t workSize;
    cufftMakePlan1d(temp_plan, fft_size_, CUFFT_C2C, 1, &workSize);
    cufftExecC2C(temp_plan, (cufftComplex*)d_mask_buffer_, (cufftComplex*)d_mask_buffer_, CUFFT_FORWARD);
    cufftDestroy(temp_plan);

    //Initialize transfer streams
    cudaStreamCreate(&transfer1_stream_);
    cudaStreamCreate(&transfer2_stream_);

    //Default to 2 streams
    initProcStreams(1);

    //Create one event per processing buffer
    transfer1_done_events_.resize(buffers_per_call_);
    kernels_done_events_.resize(buffers_per_call_);
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        cudaEventCreate(&transfer1_done_events_[ii]);
        cudaEventCreate(&kernels_done_events_[ii]);
    }

    //Execute plans at least once to ensure no first-call overhead
    cudaMemset(d_io_buffer_, 0, 2 * buffers_per_call_ * fft_size_ * sizeof(float));
    for (int ii = 0; ii < fwd_plans_.size(); ii++) {
        float* d_io_ptr = &d_io_buffer_[2 * ii * fft_size_];
        cufftExecC2C(fwd_plans_[ii], (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_FORWARD);
    }
}

FastFirGPU1::~FastFirGPU1() {
    //Destroy events
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        cudaEventDestroy(transfer1_done_events_[ii]);
        cudaEventDestroy(kernels_done_events_[ii]);
    }

    //Destroy streams/plans
    initProcStreams(0);

    //Free device memory
    DEVICE_FREE(d_io_buffer_);
    DEVICE_FREE(d_mask_buffer_);

}

void FastFirGPU1::run(float* input, float* output) {
    //TODO:: only dealing with non-contiguous for now, need to add mods for contiguous
    int output_samps_2sided = FastFir::getOutputSamps2Sided(mask_samps_, input_samps_);

    //Calculate kernel execution parameters
    //Note: Assuming single GPU configuration
    int tpb = getMaxThreadsPerBlock(0);
    int num_blocks = getNumBlocks(tpb, fft_size_);

    //Calculate fft scaling
    float scale = ((double)1.0) / fft_size_;

    int num_proc_streams = proc_streams_.size();
    for (int ii = 0; ii < buffers_per_call_; ii++) {
        int proc_stream_index = ii % num_proc_streams;

        //Choose streams
        cudaStream_t stream1 = transfer1_stream_;
        cudaStream_t stream2 = proc_streams_[proc_stream_index];
        cudaStream_t stream3 = transfer2_stream_;

        //Choose cufft plans
        cufftHandle fwd_plan = fwd_plans_[proc_stream_index];
        cufftHandle rev_plan = rev_plans_[proc_stream_index];

        //Set buffer pointers
        float* d_io_ptr = &d_io_buffer_[2 * ii * fft_size_];
        float* h_input_ptr = &input[2 * ii * input_samps_];
        float* h_output_ptr = &output[2 * ii * output_samps_2sided];

        //Transfer1 : H->D : Move input samples to device and zero pad
        cudaMemcpyAsync(d_io_ptr, h_input_ptr, 2 * input_samps_ * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemsetAsync(&d_io_ptr[2 * input_samps_], 0, 2 * (fft_size_ - input_samps_) * sizeof(float));
        cudaEventRecord(transfer1_done_events_[ii], stream1);

        //Run fwd fft
        cudaStreamWaitEvent(stream2, transfer1_done_events_[ii]);
        cufftExecC2C(fwd_plan, (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_FORWARD);

        //Run cpx mpy/scaling kernel
        vectorCpxMpy << <num_blocks, tpb, 0, stream2 >> > (d_io_ptr, d_mask_buffer_, d_io_ptr, fft_size_);
        vectorCpxScale << <num_blocks, tpb, 0, stream2 >> > (d_io_ptr, d_io_ptr, scale, fft_size_);

        //Run rev fft
        cufftExecC2C(rev_plan, (cufftComplex*)d_io_ptr, (cufftComplex*)d_io_ptr, CUFFT_INVERSE);
        cudaEventRecord(kernels_done_events_[ii], stream2);

        //Transfer2 : D->H : Move output samples to host
        cudaStreamWaitEvent(stream3, kernels_done_events_[ii]);
        cudaMemcpyAsync(h_output_ptr, d_io_ptr, 2 * output_samps_2sided * sizeof(float), cudaMemcpyDeviceToHost, stream3);
    }

    //Synchronize all streams
    cudaStreamSynchronize(transfer1_stream_);
    for (int ii = 0; ii < num_proc_streams; ii++) {
        cudaStreamSynchronize(proc_streams_[ii]);
    }
    cudaStreamSynchronize(transfer2_stream_);
}


//Allows override of number of streams
void FastFirGPU1::setNumProcStreams(int num_streams) {
    initProcStreams(num_streams);
}

void FastFirGPU1::initProcStreams(int num_streams) {
    //De-allocate any currently created strams/plans
    if (proc_streams_.size() == 0) {
        for (int ii = 0; ii < proc_streams_.size(); ii++) {
            cufftDestroy(fwd_plans_[ii]);
            cufftDestroy(rev_plans_[ii]);
            cudaStreamDestroy(proc_streams_[ii]);
        }
    }

    //Initialize new configuration
    fwd_plans_.resize(num_streams);
    rev_plans_.resize(num_streams);
    proc_streams_.resize(num_streams);
    for (int ii = 0; ii < num_streams; ii++) {
        //Initialize streams
        cudaStreamCreate(&proc_streams_[ii]);

        //Create cufft plans
        cufftCreate(&fwd_plans_[ii]);
        cufftCreate(&rev_plans_[ii]);
        size_t workSize;
        cufftMakePlan1d(fwd_plans_[ii], fft_size_, CUFFT_C2C, 1, &workSize);
        cufftMakePlan1d(rev_plans_[ii], fft_size_, CUFFT_C2C, 1, &workSize);

        //Associate streams to plans
        cufftSetStream(fwd_plans_[ii], proc_streams_[ii]);
        cufftSetStream(rev_plans_[ii], proc_streams_[ii]);
    }
}

//Vectorized complex multiply
__global__ void vectorCpxMpy(float* input1, float* input2, float* output, int NN) {
    //One dimensional block configuration
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
    //One dimensional block configuration
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