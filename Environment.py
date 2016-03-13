import numpy
from DispersionIndex import DispersionIndex
from GlobalSettings import Settings
from Singleton import Singleton

@Singleton
class PhysicalEnvironment():
    def __init__(self):
        dispersionIndex = DispersionIndex.Instance()

        
        self.c = numpy.float64(3e-7)  # [fs]
        self.w = numpy.float64(2.0 * numpy.pi * self.c / Settings.wave_length)
        self.dnL = numpy.abs(1.0 / 2.0 * dispersionIndex.dng(self.w))
        self.aw3 = numpy.float64(self.dnL * self.w / self.c)
        self.T = Settings.wave_length / self.c
        self.N = dispersionIndex.n(self.w)
        self.k0 = self.w * dispersionIndex.n(self.w) / self.c
        self.beta2 = (2.0 * dispersionIndex.dng(self.w) / self.w + self.w * dispersionIndex.d2ndw2(self.w)) / self.c
        self.Ldiff = 2.0 * numpy.pi * Settings.x_width**2.0 / 4.0 / Settings.wave_length * self.N
        self.Ldisp = Settings.pulse_length * Settings.pulse_length / numpy.abs(self.beta2) / 4.0
        self.Lnl = Settings.x_width / 2.0 * numpy.sqrt(self.N / Settings.n2e / Settings.intensity)
        self.D = self.c * self.c / (2.0 * self.w * self.w * dispersionIndex.n(self.w) * self.dnL * Settings.x_width**2)
        if Settings.use_cubic:
            self.G = 4.0 * Settings.n2e * Settings.intensity / self.dnL
        else:
            self.G = 0