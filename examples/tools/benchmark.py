from dataclasses import dataclass

import numpy as np
import time


BENCHMARK_REPEATS           = 7
BENCHMARK_CUTOFF_TOLERANCE  = 0.05
BENCHMARK_CUTOFF_WINDOW     = 4
BENCHMARK_FIT_MIN_POINTS    = 6


@dataclass
class BenchmarkResult:

    sample_range       : np.ndarray
    times              : np.ndarray
    cutoff             : int
    fitted             : np.ndarray
    exponent           : float
    reference_exponent : int


class Benchmark:
    """Utilities for measuring and analysing computational scaling."""

    def __init__(self,
                 repeats        : int   = BENCHMARK_REPEATS,
                 tolerance      : float = BENCHMARK_CUTOFF_TOLERANCE):

        self.repeats        = repeats
        self.tolerance      = tolerance

        self.result         : BenchmarkResult | None = None


    def run(self,
            sample_range : np.ndarray,
            operation,
            setup = None) -> None:

        sample_range = np.asarray(sample_range)
        times        = []

        for n in sample_range:

            if setup is None:
                runtime = self.measure(lambda: operation(n))
            else:
                runtime = self.measure(operation, setup = lambda n = n: setup(n))

            times.append(runtime)

        times  = np.asarray(times)
        cutoff = self._cutoff(sample_range,
                              times)

        fitted, exponent, reference_exponent= self._fit(sample_range,
                                                        times,
                                                        cutoff)

        self.result = BenchmarkResult(sample_range       = sample_range,
                                      times              = times,
                                      cutoff             = cutoff,
                                      fitted             = fitted,
                                      exponent           = exponent,
                                      reference_exponent = reference_exponent)


    def measure(self,
                operation,
                setup = None) -> float:

        times = []

        for _ in range(self.repeats):

            state = setup() if setup is not None else None
            start = time.perf_counter()

            operation(state) if setup is not None else operation()

            times.append(time.perf_counter() - start)

        return float(np.median(times))


    def _cutoff(self,
                sample_range : np.ndarray,
                times        : np.ndarray) -> int:
        """Returns the first stable power-law scaling cutoff."""

        maximum_cutoff = len(sample_range) - BENCHMARK_FIT_MIN_POINTS

        cutoffs   = sample_range[:maximum_cutoff + 1]
        exponents = []

        for cutoff in cutoffs:
            fit = sample_range >= cutoff
            exponents.append(np.polyfit(np.log(sample_range[fit]),
                             np.log(times[fit]),
                             1)[0])

        exponents = np.asarray(exponents)
        for i, cutoff in enumerate(cutoffs):

            window = exponents[i:i + BENCHMARK_CUTOFF_WINDOW]

            if len(window) == BENCHMARK_CUTOFF_WINDOW and np.ptp(window) < self.tolerance:
                return int(cutoff)

        raise RuntimeError("No stable scaling region found.")


    def _fit(self,
             sample_range : np.ndarray,
             times        : np.ndarray,
             cutoff       : int) -> tuple[np.ndarray, float]:
        """Fits a power law over the stable scaling region."""

        fit = sample_range >= cutoff

        exponent, _         = np.polyfit(np.log(sample_range[fit]),
                                        np.log(times[fit]),
                                        1)

        reference_exponent  = round(exponent)  
        log_coefficient     = np.mean(np.log(times[fit]) - reference_exponent*np.log(sample_range[fit]))
        fitted              = (np.exp(log_coefficient) * sample_range[fit]**reference_exponent)

        return fitted, float(exponent), reference_exponent



    def plot(self,
             ax,
             label : str,
             color : str) -> None:
        """Plots the benchmark result and fitted scaling."""

        if self.result is None:
            raise RuntimeError("Benchmark has not been run.")

        result = self.result
        fit    = result.sample_range >= result.cutoff

        ax.loglog(result.sample_range,
                  result.times,
                  "o-",
                  color = color,
                  alpha = 0.3,
                  label = label)

        if result.reference_exponent == 1:
            fit_label = r"$O(N)$"
        else:
            fit_label = rf"$O(N^{{{result.reference_exponent}}})$"

        ax.loglog(result.sample_range[fit],
                  result.fitted,
                  "--",
                  color = color,
                  alpha = 0.6,
                  label = fit_label)