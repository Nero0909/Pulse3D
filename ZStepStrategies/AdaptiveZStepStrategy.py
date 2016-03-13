import numpy
import sys
from GlobalSettings import Settings


class AdaptiveZStepStrategy:
    def __init__(self):
        self.max_iteration_number = -sys.maxint - 1
        self.max_error = sys.float_info.min
        self.adaptive_dz = Settings.dz
        self.possible_iteration_number = Settings.possible_iteration_number
        self.possible_error = Settings.possible_error
        self.layer_number = Settings.layer_number
        self.layer_count = 0

    def calculateDz(self, current_dz, iteration_number):
        if self.needUpdateDz(iteration_number):
            if current_dz < 0:
                current_dz = self.adaptive_dz

            if self.needIncreaseDz():
                current_dz *= 1.5

            if self.needDecreaseDz(iteration_number):
                current_dz /= 2.0
            self.resetStatistic()
        else:
            if self.layer_count == self.layer_number:
                self.resetStatistic()

            self.layer_count += 1
            self.max_iteration_number = max(self.max_iteration_number, iteration_number)

        return current_dz

    def needUpdateDz(self, iteration_number):
        return self.needIncreaseDz() or self.needDecreaseDz(iteration_number)

    def needIncreaseDz(self):
        if self.layer_count == self.layer_number:
            return self.max_iteration_number < self.possible_iteration_number or self.max_error < self.possible_error / 2.0
        return False

    def needDecreaseDz(self, iteration_number):
        return iteration_number > self.possible_iteration_number

    def resetStatistic(self):
        self.layer_count = 0
        self.max_iteration_number = -sys.maxint - 1
        self.max_error = sys.float_info.min

