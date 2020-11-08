#include "Stopwatch.h"

#include <windows.h>

#include <stdio.h>

//Note: This file is WINDOWS only for now!

Stopwatch::Stopwatch()
{
	//Determine performance counter frequency
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	clock_frequency_ = (double)freq.QuadPart;

	//Read current count value
	LARGE_INTEGER current_count;
	QueryPerformanceCounter(&current_count);
	creation_count_ = current_count.QuadPart;
	last_count_ = current_count.QuadPart;
}

Stopwatch::~Stopwatch()
{
}

double Stopwatch::getElapsed()
{
	//Read current count
	LARGE_INTEGER current_count;
	QueryPerformanceCounter(&current_count);

	//Compute count difference
	long long count_diff = ((long long)current_count.QuadPart) - last_count_;

	//Convert to seconds
	return ((double) count_diff) / ((double) clock_frequency_);
}
