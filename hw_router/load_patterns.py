import numpy as np
import time
import math

class RequestPattern:
    def __init__(self, pattern="poisson", rate=5.0,
                 spike_intensity=10.0, spike_period=20.0,
                 burst_duration=3.0):
        """
        Args:
            pattern: {"poisson", "microburst", "sustained"}
            rate: base arrival rate λ (req/sec)
            spike_intensity: multiplier during microburst
            spike_period: seconds between microbursts
            burst_duration: seconds of burst inside each period
        """
        self.pattern = pattern.lower()
        self.rate = rate
        self.spike_intensity = spike_intensity
        self.spike_period = spike_period
        self.burst_duration = burst_duration
        self._t0 = time.time()

    def next_delay(self):
        t = (time.time() - self._t0) % self.spike_period

        if self.pattern == "poisson":
            rate = self.rate

        elif self.pattern == "microburst":
            # Smooth burst window with jitter
            in_burst = t < self.burst_duration
            if in_burst:
                rate = self.rate * self.spike_intensity
            else:
                rate = self.rate

            # add small jitter so bursts aren't phase-locked
            rate *= np.random.uniform(0.9, 1.1)

        elif self.pattern == "sustained":
            # true overload: rate = 3× base
            # for H100, this is enough to saturate decode/prefill
            rate = self.rate * 3.0

            # jitter to simulate natural fluctuations
            rate *= np.random.uniform(0.95, 1.05)

        else:
            rate = self.rate

        return np.random.exponential(1.0 / rate)
