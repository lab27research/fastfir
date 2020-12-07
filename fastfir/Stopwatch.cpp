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
    return ((double)count_diff) / ((double)clock_frequency_);
}
