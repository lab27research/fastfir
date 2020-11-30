#pragma once
//Create a virtual class that all implementations can derive from
class FastFir
{
public:
    //Constructs a Fast Fir implementation that will process "buffers_per_call" sets of
    // size "input_samps" using specified input mask
    FastFir(float* mask, int mask_samps, int input_samps, int buffers_per_call, bool contiguous);
    ~FastFir();

    //Returns the total output samps given constructor parameters
    // (includes buffers_per_call and contiguous settings)
    int getTotalOutputSamps();

    //Other common utilities

    //Returns the output samps for each no-transient buffer (N-M+1)
    static int getOutputSampsNoTransient(int mask_samps, int input_samps);

    //Returns the output samps for each one-sided transient buffer (N)
    static int getOutputSamps1Sided(int mask_samps, int input_samps);

    //Returns the output samps for each two-sided transient buffer (N+M-1)
    static int getOutputSamps2Sided(int mask_samps, int input_samps);

    //Returns the next power of two greater than or equal to getOutputSamps
    static int getFFTSize(int mask_samps, int input_samps);

    //Returns the FLOPs per buffer for the two-sided time domain implementation
    static double getTimeDomainFLOPs(int mask_samps, int input_samps);

    //Returns the FLOPs per buffer for the two-sided freq domain implementation
    static double getFreqDomainFLOPs(int mask_samps, int input_samps);

protected:
    //Implementation must process "buffers_per_call" sets of size "input_samps" that are
    // assumed to be packed at the pointer "input"
    //Results are placed in "buffers_per_call" sets of size getOutputSamps()
    //
    //Note that for each set of input samples of length ("input_samps"), a double-sided transient
    // will be computed.
    //Example for input_samps=4, mask_samps=2
    //Overlaps that will be computed:
    //mask (len M=2):    [1 1]         [1 1]         [1 1]         [1 1]         [1 1]
    //input (len N=4):     [1 2 3 4]   [1 2 3 4]   [1 2 3 4]   [1 2 3 4]   [1 2 3 4]
    //total outputs: = (M-1) + (N - M + 1) + (M-1) = N + M - 1 = 4 + 2 - 1 = 5
    //
    //If "contiguous" is set to true, each run call will then properly add the transients.  Otherwise it
    // will leave the transients as separate outputs.
    // Let M = mask_samps
    //     N = input_samps
    //     BPC = buffers_per_call
    // contiguous=True --> Total outputs placed in output array: (M-1) + (N*BPC + M - 1) + (M-1) = (N*BPC + M - 1)
    // contiguous=False --> Total outputs placed in output array: BPC * ( (M-1) + (N* + M - 1) + (M-1) ) = BPC*(N + M - 1)
    virtual void run(float* input, float* output) = 0;

    //Constructor parameters
    int mask_samps_;
    int input_samps_;
    int buffers_per_call_;
    bool contiguous_;
};


