import numpy as np
from Errors.EnergyError import EnergyError


class ComputationalErrors:
    def __init__(self):
        self.linear_error = EnergyError()
        self.nonlinear_error = EnergyError()

    def get_error(self):
        return self.linear_error.get_error() + self.nonlinear_error.get_error()