import numpy
import matplotlib.pyplot as plt
from GlobalSettings import Settings
from Grid import Grid
from mayavi import mlab

class Graph():
    def __init__(self, field):
        self.grid = Grid.Instance()
        self.fieldSym = numpy.zeros((self.grid.space_size*2, self.grid.time_size), dtype=numpy.complex128)
        self.SSym = numpy.zeros(self.grid.space_size * 2, dtype=numpy.float64)
        self.T, self.SSym = numpy.meshgrid(self.grid.time_grid, self.SSym)
        self.__getSymmetricField(field)

    def __getSymmetricField(self, field):
        for j in range(self.grid.space_size):
            for i in range(self.grid.time_size):
                self.fieldSym[self.grid.space_size + j, i] = field[j, i]
                self.fieldSym[self.grid.space_size - 1 - j, i] = field[j, i]

            self.SSym[self.grid.space_size + j] = self.grid.space_grid[j]
            self.SSym[self.grid.space_size - 1 - j] = -self.grid.space_grid[j]


    def plot3D(self, title = ''):
        s = mlab.surf(self.SSym / Settings.x_width, self.T, self.fieldSym.real, warp_scale="auto")
        if title != '':    
           mlab.title(title, height = 0.1, size = 0.4)
        mlab.show()


    def plot2D(self, field):
        fig = plt.figure()
        fig.suptitle('ES and ET')
        ax = fig.add_subplot(1, 2, 2)
        plt.plot(self.T[0, :], field.real[0, :])

        ax = fig.add_subplot(1, 2, 1)
        maxArg = field.real.argmax(axis=1)[0]
        minArg = field.real.argmin(axis=1)[0]
        plt.plot(self.grid.space_grid / Settings.x_width, field.real[:, maxArg])
        plt.plot(self.grid.space_grid / Settings.x_width, field.real[:, minArg])
        plt.show()