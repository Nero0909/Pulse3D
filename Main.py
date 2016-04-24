import os
import numpy
import time
from XmlDeserialize import Deserializer
from GlobalSettings import Settings
from GaussDistribution import Gauss
from Plot import Graph
from Grid import Grid
from ComputationalContext import ComputationalContext


def print_profile_info(context):
    print ("Profiling info:")
    print("----------------")
    print("Nonlinear computing time: {0} seconds".format(context.nlnr_computing_time / 10**9))
    print("Dispersion computing time: {0} seconds".format(context.disp_computing_time / 10**9))
    print("Difraction computing time: {0} seconds".format(context.diff_computing_time / 10**9))
    print("Copy from buffer time: {0} seconds".format(context.copy_computing_time / 10**9))
    print("Number of layers: {0}".format(context.layer))
    print("----------------")


def doSteps(context):
    while not context.isStop():
        context.doStep(-1)


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    onlyPlot = False
    deserializer = Deserializer('LBProjects/adaptive_full')
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
        context.copyFromBuffer(field)

        print_profile_info(context)
        print("Total time: {0} s".format(time.time() - t))

        graph = Graph(field)
        graph.plot3D()


if __name__ == '__main__':
    main()
