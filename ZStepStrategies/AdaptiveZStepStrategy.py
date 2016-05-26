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

    def calculate_dz(self, current_dz, iteration_number, errors):
        if self.need_update_dz(iteration_number, errors):
            if current_dz < 0:
                current_dz = self.adaptive_dz

            if self.need_increase_dz():
                current_dz *= 1.5

            if self.need_decrease_dz(iteration_number, errors):
                current_dz /= 2.0
            self.reset_statistic()
        else:
            if self.layer_count == self.layer_number:
                self.reset_statistic()

            self.layer_count += 1
            self.max_iteration_number = max(self.max_iteration_number, iteration_number)
            self.max_error = max(self.max_error, errors)

        return current_dz

    def need_update_dz(self, iteration_number, errors):
        return self.need_increase_dz() or\
               self.need_decrease_dz(iteration_number, errors)

    def need_increase_dz(self):
        if self.layer_count == self.layer_number:
            return self.max_iteration_number < self.possible_iteration_number and\
                   self.max_error < self.possible_error / 2.0
        return False

    def need_decrease_dz(self, iteration_number, errors):
        return iteration_number > self.possible_iteration_number or errors > self.possible_error

    def reset_statistic(self):
        self.layer_count = 0
        self.max_iteration_number = -sys.maxint - 1
        self.max_error = sys.float_info.min
