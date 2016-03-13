__author__ = 'Nero'

import os
import numpy
import time
from XmlDeserialize import Deserializer
from GlobalSettings import Settings
from GaussDistribution import Gauss
from Plot import Graph
from Grid import Grid
from ComputationalContext import ComputationalContext


def doSteps(context):
    while not context.isStop():
        context.doStep(-1)


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    onlyPlot = False
    deserializer = Deserializer('LBProjects/adaptive_unl')
    deserializer.deserialize(Settings)
    Settings.toFemtoseconds()

    grid = Grid.Instance()
    field = numpy.zeros((grid.space_size, grid.time_size), dtype=numpy.complex128)

    gauss = Gauss()
    gauss.fillField(field)

    context = ComputationalContext(field)
    context.fillData()

    if (onlyPlot):
        lBulletGraph = Graph(fieldxtxt)
        lBulletGraph.plot3D()
    else:
        t = time.time()

        doSteps(context)
        print context.layer

        context.copyFromBuffer(field)
        print time.time() - t
        graph = Graph(field)
        graph.plot3D()


if __name__ == '__main__':
    main()
