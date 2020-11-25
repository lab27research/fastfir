#pragma once
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

