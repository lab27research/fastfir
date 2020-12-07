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

//A simple class for estimating runtimes

class Stopwatch
{
public:
    Stopwatch();
    ~Stopwatch();

    //Returns the time elapse since either instantiation or
    // the last getElapsed() call
    double getElapsed();

private:
    long long creation_count_;
    long long last_count_;
    double clock_frequency_;
};

