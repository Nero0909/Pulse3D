import numpy
from GlobalSettings import Settings
from Environment import PhysicalEnvironment
from Grid import Grid

class Gauss():
    def __init__(self):
        self.grid = Grid.Instance()
        self.physEnv = PhysicalEnvironment.Instance()
        
        self.phi = Settings.phase # [rad]
        self.alpha = Settings.chirp # [rad/s^2]
        self.gamma_x = Settings.x_curvature
        self.m = Settings.m
        self.n = Settings.n

    def __createGaussDistr(self, j, i):
        x = self.grid.space_grid[j] * Settings.x_width
        t = self.grid.time_grid[i] / self.physEnv.w
        return numpy.exp(-2**(2*self.n-1) * (x/Settings.x_width)**(2*self.n)) *\
               numpy.exp(-2**(2*self.m-1) * (t/Settings.pulse_length)**(2**self.m)) *\
               numpy.sin(self.physEnv.w * (1 + self.alpha/Settings.pulse_length * t) * t + self.phi *\
                        - self.gamma_x / 2.0 * self.physEnv.k0 * x * x)

    def fillField(self, field):
        for j in range(self.grid.space_size):
            for i in range(self.grid.time_size):
                field[j,i] = self.__createGaussDistr(j,i)