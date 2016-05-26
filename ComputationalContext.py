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
from Errors.ComputationalErrors import ComputationalErrors


class ComputationalContext:
    def __init__(self, field):
        self.ocl = OpenCLSettings.Instance()
        self.dispIndex = DispersionIndex.Instance()
        self.grid = Grid.Instance()
        self.physConst = PhysicalEnvironment.Instance()
        self.ocl = OpenCLSettings.Instance()
        self.errors = ComputationalErrors()

        self.need_update_energy = False
        self.energy = numpy.empty(1).astype(numpy.float64)
        self.layer = numpy.int32(0)
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
        self.test_field = numpy.zeros((self.grid.space_size, self.grid.time_size), dtype=numpy.complex128)

        self.mf = cl.mem_flags
        self.A1_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, self.grid.space_grid.nbytes)
        self.A2_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, self.grid.space_grid.nbytes)
        self.A3_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, self.grid.space_grid.nbytes)
        self.D_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, self.D.nbytes)
        self.K_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, self.K.nbytes)
        self.space_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.grid.space_grid)
        self.space_delta_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.grid.space_delta)
        self.plan1D = Plan(self.grid.time_size, dtype=numpy.complex128, queue=self.ocl.queue)
        self.field_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=field)
        self.unlinear_iterations_buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, self.unlinear_iterations.nbytes)

        self.local_buf = cl.LocalMemory(8*self.ocl.n_threads)
        self.buffers, self.sizes = self.__compute_partial_buffers(self.grid.space_size * self.grid.time_size,
                                                                  self.ocl.n_threads)

        self.nlnr_computing_time = numpy.float64(0)
        self.diff_computing_time = numpy.float64(0)
        self.disp_computing_time = numpy.float64(0)
        self.copy_computing_time = numpy.float64(0)
        self.energy_computing_time = numpy.float64(0)

    def compute_K(self, K):
        ng = self.dispIndex.n(self.physConst.w) + self.dispIndex.dng(self.physConst.w)
        for i in range(self.grid.time_size):
            K[i] = self.grid.freq_grid[i] * \
                   (self.dispIndex.n(numpy.abs(self.grid.freq_grid[i] * self.physConst.w)) - ng) / self.physConst.dnL

    def compute_D(self, D):
        for i in range(self.grid.time_size):
            if numpy.abs(self.grid.freq_grid[i]) < 10e-8:
                D[i] = 0
            else:
                D[i] = self.physConst.D / self.grid.freq_grid[i]

    def fill_data(self):
        self.compute_D(self.D)
        self.compute_K(self.K)
        self.ocl.initial_prg.ComputeA(self.ocl.queue, self.grid.space_grid.shape, None,
                                      self.A1_buf, self.A2_buf, self.A3_buf, self.space_buf, self.space_delta_buf,
                                      self.grid.space_size)

        cl.enqueue_copy(self.ocl.queue, self.D_buf, self.D)
        cl.enqueue_copy(self.ocl.queue, self.K_buf, self.K)

    def is_stop(self):
        return self.Z >= self.z_limit

    def update_z(self):
        self.Z += self.current_dz
        self.layer += 1

    def update_dz(self):
        if self.calculated_dz < 0:
            self.calculated_dz = Settings.dz

        dz = self.calculated_dz
        if self.z_limit < self.Z + dz:
            dz = self.z_limit - self.Z

        self.current_dz = dz

        if self.current_dz < 0 and numpy.abs(self.current_dz) < 1e-10:
            self.current_dz = 0

    def update_energy(self):
        if self.need_update_energy:
            self.compute_energy()
        self.need_update_energy = False

    def compute_energy(self):
        evt = self.ocl.error_prg.CalculateEnergy(self.ocl.queue, (self.sizes[0],), (self.ocl.n_threads,),
                                                 self.field_buf, self.space_buf, self.space_delta_buf,
                                                 self.buffers[0], self.local_buf, self.grid.time_size, self.grid.space_size)
        evt.wait()
        self.energy_computing_time += evt.profile.end - evt.profile.start

        for i in range(1, len(self.buffers)):
            evt = self.ocl.error_prg.Reduce(self.ocl.queue, (self.sizes[i],), (min(self.ocl.n_threads, self.sizes[i]),),
                                            self.buffers[i-1], self.buffers[i], self.local_buf)
            evt.wait()
            self.energy_computing_time += evt.profile.end - evt.profile.start

        evt = cl.enqueue_copy(self.ocl.queue, self.energy, self.buffers[-1])
        evt.wait()
        self.copy_computing_time += evt.profile.end - evt.profile.start

    def do_step(self, dz):
        iteration_number = 0
        while True:
            self.need_update_energy = True

            if dz < 0:
                self.calculated_dz = self.z_step_strategy.calculate_dz(self.calculated_dz, self.global_iteration_number,
                                                                       self.errors.get_error())
                self.update_dz()
            else:
                self.current_dz = dz

            self.linear()
            self.nonlinear()

            if iteration_number > 50:
                break
            iteration_number += 1

            if not (dz < 0 and self.z_step_strategy.need_update_dz(self.global_iteration_number,
                                                                   self.errors.get_error())):
                break

        self.update_z()

    def linear(self):
        self.errors.linear_error.begin(self)
        # Обратное преобразование Фурье
        cl.enqueue_copy(self.ocl.queue, self.test_field, self.field_buf)

        self.plan1D.execute(self.field_buf, batch=self.grid.space_size, inverse=True)

        cl.enqueue_copy(self.ocl.queue, self.test_field, self.field_buf)

        if Settings.use_difraction:
            # Применяем оператор дифракции
            dif_evt = self.ocl.linear_prg.Diff(self.ocl.queue, (self.grid.time_size,), None,
                                               self.field_buf, self.A1_buf, self.A2_buf, self.A3_buf,
                                               self.space_buf, self.space_delta_buf, self.D_buf,
                                               self.grid.space_size, self.grid.time_size, self.current_dz)
            dif_evt.wait()
            self.diff_computing_time += dif_evt.profile.end - dif_evt.profile.start

        cl.enqueue_copy(self.ocl.queue, self.test_field, self.field_buf)

        # Применяем оператор дисперсии
        disp_evt = self.ocl.linear_prg.Disp(self.ocl.queue, self.field_shape, None,
                                            self.K_buf, self.field_buf, self.current_dz, self.grid.space_size,
                                            self.grid.time_size)
        disp_evt.wait()

        cl.enqueue_copy(self.ocl.queue, self.test_field, self.field_buf)

        self.disp_computing_time += disp_evt.profile.end - disp_evt.profile.start

        # Прямое преобразование Фурье
        self.plan1D.execute(self.field_buf, batch=self.grid.space_size, inverse=False)

        self.need_update_energy = True
        self.errors.linear_error.end(self)

    def nonlinear(self):
        if Settings.use_cubic:
            self.errors.nonlinear_error.begin(self)

            dt = self.grid.time_delta[1]
            k = numpy.float64(self.physConst.G * self.current_dz / dt / 24.0)
            max_error = numpy.float64(1e-6)
            iteration = numpy.int32(0)

            cl.enqueue_copy(self.ocl.queue, self.test_field, self.field_buf)

            evt = self.ocl.nonlinear_prg.CubicUnlinean1DSolve(self.ocl.queue, (self.grid.space_size, 1), None,
                                                              self.field_buf, self.unlinear_iterations_buf,
                                                              k, dt, max_error, iteration, self.grid.time_size)

            evt.wait()
            self.nlnr_computing_time += evt.profile.end - evt.profile.start

            cl.enqueue_copy(self.ocl.queue, self.test_field, self.field_buf)

            cl.enqueue_copy(self.ocl.queue, self.unlinear_iterations, self.unlinear_iterations_buf)
            self.global_iteration_number = numpy.ndarray.max(self.unlinear_iterations)

            self.need_update_energy = True
            self.errors.nonlinear_error.end(self)

    def copy_from_buffer(self, field):
        evt = cl.enqueue_copy(self.ocl.queue, field, self.field_buf)
        evt.wait()
        self.copy_computing_time += evt.profile.end - evt.profile.start

    def __compute_partial_buffers(self, size, n_threads):
        partial_sums = list()
        sizes = list()
        break_flag = True

        sizes.append(size)
        while break_flag:
            if size / n_threads == 0:
                break_flag = False
                size = 1
            else:
                size = size / n_threads
            buf_np = numpy.empty(size, dtype=numpy.float64)
            buf = cl.Buffer(self.ocl.ctx, self.mf.READ_WRITE, size=buf_np.nbytes)
            partial_sums.append(buf)
            sizes.append(size)

        return partial_sums, sizes
