from GlobalSettings import Settings
from Singleton import Singleton
import numpy

@Singleton
class DispersionIndex():
    def __init__(self):
        self.a = Settings.a
        self.b = Settings.b
        self.N0 = Settings.N0
        self.c = numpy.float64(3e-7)  # [fs]

    def n(self, w):
        if w < 1e-30:
            return 1
        else:
            return self.N0 + self.a * self.c * w * w - self.b * self.c / (w * w)

    def dng(self, w):
        if w < 1e-30:
            return 1
        else:
            return 2.0 * self.a * self.c * w * w + 2.0 * self.b * self.c / (w * w)

    def d2ndw2(self, w):
        if w < 1e-30:
            return 1
        else:
            return 2 * self.a * self.c - 6 * self.b * self.c / (w * w * w * w)

