# coding=utf-8
import numpy as np


class EnergyError:
    def __init__(self):
        self.begin_energy = np.float64(1)
        self.end_energy = np.float64(1)

    def begin(self, context):
        context.update_energy()
        self.__begin(context.energy[0])

    def end(self, context):
        context.update_energy()
        self.__end(context.energy[0])

    def get_error(self):
        return np.abs(1.0 - self.end_energy / self.begin_energy)

    def __begin(self, energy):
        self.begin_energy = energy
        self.end_energy = energy

    def __end(self, energy):
        self.end_energy = energy
