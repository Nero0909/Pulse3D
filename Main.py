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


def do_steps(context):
    while not context.is_stop():
        context.do_step(-1)


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    onlyPlot = False
    deserializer = Deserializer('LBProjects/test_1024_64_bi')
    deserializer.deserialize(Settings)
    Settings.toFemtoseconds()

    grid = Grid.Instance()
    field = numpy.zeros((grid.space_size, grid.time_size), dtype=numpy.complex128)

    gauss = Gauss()
    gauss.fillField(field)

    context = ComputationalContext(field)
    context.fill_data()

    statistic = Statistic()

    if (onlyPlot):
        lBulletGraph = Graph(fieldxtxt)
        #lBulletGraph.plot2D(big_intenstxt)
        #lBulletGraph.plot3D()
    else:
        statistic.set_start_time()

        do_steps(context)
        context.copy_from_buffer(field)

        statistic.set_end_time()
        statistic.print_profile_info(context)
        statistic.print_total_time()
        
        #statistic.print_error(fieldxtxt, field)

        graph = Graph(field)
        graph.plot3D()
        #graph.plot2DCompare(field, fieldxtxt, 63)


if __name__ == '__main__':
    main()
