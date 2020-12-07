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

//Freq domain CPU implementation of the FastFir (overlap and add implementation)
class FastFirCPU2 : public FastFir
{
public:
    FastFirCPU2(float* mask, int mask_samps, int input_samps,
                int buffers_per_call = 1, bool contiguous = false);
    ~FastFirCPU2();
    void run(float* input, float* output);

private:
    int fft_size_;
    float* io_buffer_;
    float* mask_buffer_;
    fftwf_plan fwd_plan_;
    fftwf_plan rev_plan_;

};

