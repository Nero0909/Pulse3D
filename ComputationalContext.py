# coding=utf-8
import pyopencl as cl
import numpy
from GlobalSettings import Settings
from DispersionIndex import DispersionIndex
from OpenCLSettings import OpenCLSettings
from Environment import PhysicalEnvironment
from ZStepStrategies.AdaptiveZStepStrategy import AdaptiveZStepStrategy
from ZStepStrategies.UniformZStepStrategy import UniformZStepStrategy
from Grid import Grid
from pyfft.cl import Plan


class ComputationalContext:
    def __init__(self, field):
        self.ocl = OpenCLSettings.Instance()
        self.dispIndex = DispersionIndex.Instance()
        self.grid = Grid.Instance()
        self.physConst = PhysicalEnvironment.Instance()
        self.ocl = OpenCLSettings.Instance()

        self.layer = numpy.int32(0)

        # Distance
        self.Z = numpy.int32(0)
        self.z_limit = Settings.z
        self.dz = Settings.dz
        self.current_dz = numpy.float64()
        self.calculated_dz = Settings.dz

        if Settings.z_strategy == "Uniform":
            self.z_step_strategy = UniformZStepStrategy()
        else:
            self.z_step_strategy = AdaptiveZStepStrategy()

        self.global_iteration_number = numpy.int32(0)
        self.unlinear_iterations = numpy.zeros(self.grid.space_size, dtype=numpy.int32)

        self.E_next = numpy.zeros((self.grid.space_size, self.grid.time_size), dtype=numpy.float64)
        self.D = numpy.zeros(self.grid.time_size, dtype=numpy.float64)
        self.K = numpy.zeros(self.grid.time_size, dtype=numpy.float64)

        self.field_shape = field.shape

        mf = cl.mem_flags
        self.A1_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.grid.space_grid.nbytes)
        self.A2_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.grid.space_grid.nbytes)
        self.A3_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.grid.space_grid.nbytes)
        self.D_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.D.nbytes)
        self.K_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.K.nbytes)
        self.space_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.grid.space_grid)
        self.space_delta_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.grid.space_delta)
        self.plan1D = Plan(self.grid.time_size, dtype=numpy.complex128, queue=self.ocl.queue)
        self.field_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=field)
        self.field_buf_real = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, field.real.nbytes)
        self.e_next_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.E_next)
        self.e_05_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.E_next)
        # self.global_iteration_number_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.global_iteration_number.nbytes)
        self.unlinear_iterations_buf = cl.Buffer(self.ocl.ctx, mf.READ_WRITE, self.unlinear_iterations.nbytes)

        self.nlnr_computing_time = numpy.float64(0)
        self.diff_computing_time = numpy.float64(0)
        self.disp_computing_time = numpy.float64(0)
        self.copy_computing_time = numpy.float64(0)

    def computeK(self, K):
        ng = self.dispIndex.n(self.physConst.w) + self.dispIndex.dng(self.physConst.w)
        for i in range(self.grid.time_size):
            K[i] = self.grid.freq_grid[i] * \
                   (self.dispIndex.n(numpy.abs(self.grid.freq_grid[i] * self.physConst.w)) - ng) / self.physConst.dnL

    def computeD(self, D):
        for i in range(self.grid.time_size):
            if numpy.abs(self.grid.freq_grid[i]) < 10e-8:
                D[i] = 0
            else:
                D[i] = self.physConst.D / self.grid.freq_grid[i]

    def fillData(self):
        self.computeD(self.D)
        self.computeK(self.K)
        self.ocl.initial_prg.ComputeA(self.ocl.queue, self.grid.space_grid.shape, None,
                                      self.A1_buf, self.A2_buf, self.A3_buf, self.space_buf, self.space_delta_buf,
                                      self.grid.space_size)

        cl.enqueue_copy(self.ocl.queue, self.D_buf, self.D)
        cl.enqueue_copy(self.ocl.queue, self.K_buf, self.K)

    def isStop(self):
        return self.Z >= self.z_limit

    def updateZ(self):
        self.Z += self.current_dz
        self.layer += 1

    def updateDz(self):
        if self.calculated_dz < 0:
            self.calculated_dz = Settings.dz

        dz = self.calculated_dz
        if self.z_limit < self.Z + dz:
            dz = self.z_limit - self.Z

        self.current_dz = dz

        if self.current_dz < 0 and numpy.abs(self.current_dz) < 1e-10:
            self.current_dz = 0

    def doStep(self, dz):
        iteration_number = 0

        while True:
            if dz < 0:
                self.calculated_dz = self.z_step_strategy.calculateDz(self.calculated_dz, self.global_iteration_number)
                self.updateDz()
            else:
                self.current_dz = dz

            self.linear()
            self.nonlinear()

            if iteration_number > 50:
                break
            iteration_number += 1

            if not (dz < 0 and self.z_step_strategy.needUpdateDz(self.global_iteration_number)):
                break

        self.updateZ()

    def linear(self):
        # Обратное преобразование Фурье
        self.plan1D.execute(self.field_buf, batch=self.grid.space_size, inverse=True)
        # self.ocl.prg.Conj(self.ocl.queue, self.field_shape, None,  self.field_buf, self.grid.space_size)

        if Settings.use_difraction:
            # Применяем оператор дифракции
            dif_evt = self.ocl.linear_prg.Diff(self.ocl.queue, (self.grid.time_size, 1), None,
                                               self.field_buf, self.A1_buf, self.A2_buf, self.A3_buf,
                                               self.space_buf, self.space_delta_buf, self.D_buf,
                                               self.grid.space_size, self.grid.time_size, self.current_dz)
            dif_evt.wait()
            self.diff_computing_time += dif_evt.profile.end - dif_evt.profile.start

        # Применяем оператор дисперсии
        disp_evt = self.ocl.linear_prg.Disp(self.ocl.queue, self.field_shape, None,
                                            self.K_buf, self.field_buf, self.current_dz, self.grid.space_size,
                                            self.grid.time_size)
        disp_evt.wait()
        self.disp_computing_time += disp_evt.profile.end - disp_evt.profile.start

        # Прямое преобразование Фурье
        self.plan1D.execute(self.field_buf, batch=self.grid.space_size, inverse=False)

    def nonlinear(self):
        if Settings.use_cubic:
            dt = self.grid.time_delta[1]
            k = numpy.float64(self.physConst.G * self.current_dz / dt / 24.0)
            max_error = numpy.float64(1e-6)
            iteration = numpy.int32(0)

            evt = self.ocl.nonlinear_prg.CubicUnlinean1DSolve(self.ocl.queue, (self.grid.space_size, 1), None,
                                                              self.field_buf, self.unlinear_iterations_buf,
                                                              k, dt, max_error, iteration, self.grid.time_size)

            evt.wait()
            self.nlnr_computing_time += evt.profile.end - evt.profile.start

            cl.enqueue_copy(self.ocl.queue, self.unlinear_iterations, self.unlinear_iterations_buf)
            self.global_iteration_number = numpy.ndarray.max(self.unlinear_iterations)

    def copyFromBuffer(self, field):
        evt = cl.enqueue_copy(self.ocl.queue, field, self.field_buf)
        evt.wait()
        self.copy_computing_time += evt.profile.end - evt.profile.start
