import numpy
from GlobalSettings import Settings
from Singleton import Singleton

@Singleton
class Grid():
    def __init__(self):
        # Time
        self.time_size = Settings.time_size
        self.time_limit = Settings.time_limit
        self.time_step = self.time_limit / self.time_size
        self.pulse_shift = Settings.pulse_shift
        self.time_grid = numpy.zeros(self.time_size, dtype=numpy.float64)
        self.time_delta = numpy.zeros(self.time_size, dtype=numpy.float64)
        self.__createTimeScale()

        # Space
        self.X0 = Settings.x0
        self.space_size = Settings.space_size
        self.space_limit = Settings.space_limit
        self.space_step = self.space_limit / self.space_size
        self.space_grid = numpy.zeros(self.space_size, dtype=numpy.float64)
        self.space_delta = numpy.zeros(self.space_size, dtype=numpy.float64)
        self.__createSpaceScale()

        # Frequency
        self.freq_grid = numpy.zeros(self.time_size, dtype=numpy.float64)
        self.__createFreqScale()

    def __createTimeScale(self):
        for i in range(self.time_size):
            self.time_grid[i] = self.time_limit / self.time_size * i - self.pulse_shift
            if i > 0:
                self.time_delta[i] = self.time_grid[i] - self.time_grid[i - 1]

    def __createSpaceScale(self):

        G = numpy.log(self.space_limit / self.X0 / (self.space_size - 1)) / (self.space_size - 2)

        self.space_delta[0] = 0
        for i in range(self.space_size):
            self.space_grid[i] = self.X0 * i * numpy.exp(G * (i - 1))
            if i > 0:
                self.space_delta[i] = self.space_grid[i] - self.space_grid[i - 1]

    def __createFreqScale(self):
        for i in range(self.time_size):
            if i <= self.time_size / 2:
                self.freq_grid[i] = 2 * numpy.pi * i / self.time_limit
            else:
                self.freq_grid[i] = 2 * numpy.pi * (i - self.time_size) / self.time_limit