import time
import numpy as np


class Statistic:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = time.time()

    def print_error(self, x, x_cap):
        error = np.abs(1 - x_cap / x)
        error_median = np.median(error)
        print("----------------")
        print("Error is {} %".format(error_median * 100))
        print("----------------")

    def print_profile_info(self, context):
        print ("Profiling info:")
        print("----------------")
        print("Nonlinear computing time: {0} seconds".format(context.nlnr_computing_time / 10**9))
        print("Dispersion computing time: {0} seconds".format(context.disp_computing_time / 10**9))
        print("Difraction computing time: {0} seconds".format(context.diff_computing_time / 10**9))
        print("Energy computing time: {0} seconds".format(context.energy_computing_time / 10**9))
        print("Copy from buffer time: {0} seconds".format(context.copy_computing_time / 10**9))
        print("Number of layers: {0}".format(context.layer))

    def set_start_time(self):
        self.start_time = time.time()

    def set_end_time(self):
        self.end_time = time.time() - self.start_time

    def print_total_time(self):
        print("----------------")
        print("Total time: {0} seconds".format(self.end_time))
