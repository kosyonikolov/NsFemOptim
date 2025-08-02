#include <NavierStokes/progressTracker.h>

#include <iostream>

ProgressTracker::ProgressTracker(const int numTimeSteps, const float printIntervalMs)
    : printIntervalMs(printIntervalMs), numTimeSteps(numTimeSteps)
{
    lastPrintTime = 0;
}

void ProgressTracker::update(const int iteration)
{
    const auto elapsedTime = globalSw.millis();
    const auto delta = elapsedTime - lastPrintTime;
    if (delta >= printIntervalMs)
    {
        const int numDone = iteration + 1;
        const int numRemaining = numTimeSteps - iteration;
        float avgIterTime = -1;
        float remTime = -1;
        if (numDone > 0)
        {
            avgIterTime = elapsedTime / numDone;
            remTime = numRemaining * avgIterTime;
        }

        const int percent = 100 * numDone / (numTimeSteps + 1);

        std::cout << std::format("{} / {} ({}%): avgIterTime = {} ms, elapsed = {} s, remaining = {} s\n",
                                    numDone, numTimeSteps + 1, percent, avgIterTime, elapsedTime / 1000.0f, remTime / 1000.0f);

        lastPrintTime = elapsedTime;
    }
}