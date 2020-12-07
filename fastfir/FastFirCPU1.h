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

#include "fftw3.h"

#include "FastFir.h"

//Time domain CPU implementation of the FastFir (reference design)
class FastFirCPU1 : public FastFir
{
public:
    //Specify filter mask to be used for convolution, how many input samps per buffer, and
    // the number of buffers per call
    FastFirCPU1(float* mask, int mask_samps, int input_samps, int buffers_per_call = 1, bool contiguous = false);
    ~FastFirCPU1();
    void run(float* input, float* output);

private:
    float* mask_buffer_;

};