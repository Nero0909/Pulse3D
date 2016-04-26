import os
import numpy
import time
from XmlDeserialize import Deserializer
from GlobalSettings import Settings
from GaussDistribution import Gauss
from Plot import Graph
from Grid import Grid
from ComputationalContext import ComputationalContext
from Statistic import Statistic


def doSteps(context):
    while not context.isStop():
        context.doStep(-1)


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    onlyPlot = False
    deserializer = Deserializer('LBProjects/nonl_uni')
    deserializer.deserialize(Settings)
    Settings.toFemtoseconds()

    grid = Grid.Instance()
    field = numpy.zeros((grid.space_size, grid.time_size), dtype=numpy.complex128)

    gauss = Gauss()
    gauss.fillField(field)

    context = ComputationalContext(field)
    context.fillData()

    statistic = Statistic()

    if (onlyPlot):
        lBulletGraph = Graph(nonl_unitxt)
        lBulletGraph.plot3D()
    else:
        statistic.set_start_time()

        doSteps(context)
        context.copyFromBuffer(field)

        statistic.set_end_time()
        statistic.print_profile_info(context)
        statistic.print_total_time()
        
        #statistic.print_error(nonl_unitxt, field)

        graph = Graph(field)
        graph.plot3D()


if __name__ == '__main__':
    main()
