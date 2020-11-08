#include "FastFirCPU.h"
#include "ImpulseSource.h"
#include "sse_utils.h"
#include "Stopwatch.h"

//Simulation to compare CPU vs GPU implementations for FFT-Based FIR filtering
//
//Assumptions: 1.) Input buffers cannot guarantee alignment or to be page locked
//                 (copies are necessary for alignment guarantees)
//
//

int main() {
	int buffers_per_call = 10;
	int input_samps = 1024;
	int mask_samps = 256;
	int output_samps = FastFir::getOutputSamps(mask_samps, input_samps);
	float* input;
	float* mask;
	float* output;
	ALIGNED_MALLOC(mask, 2 * mask_samps * buffers_per_call * sizeof(float));
	ALIGNED_MALLOC(input, 2 * input_samps * buffers_per_call * sizeof(float));
	ALIGNED_MALLOC(output, 2 * output_samps * buffers_per_call * sizeof(float));

	//Create CPU-Based FIR Filter
	FastFirCPU ff1(mask, mask_samps, input_samps, buffers_per_call);

	//Create input source
	double snr = 10;
	unsigned int samp0 = 100;
	ImpulseSource is(snr, samp0);

	//This is where we need to add test bench
	Stopwatch sw;
	int total_runs = 100000;
	for (int ii = 0; ii < total_runs/buffers_per_call; ii++) {

		//Fill input buffers
		for (int jj = 0; jj < buffers_per_call; jj++) {
			is.getBuffer(&input[2 * jj * input_samps], input_samps);
		}

		//Run algorithm
		ff1.run(input, output);
	}
	double runtime = sw.getElapsed();
	printf("Completed in %.9f seconds\n",runtime);
	printf("Average time per run: %.9f\n", runtime / total_runs);
}