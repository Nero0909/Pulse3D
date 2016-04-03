import pyopencl as cl
from Singleton import Singleton

@Singleton
class OpenCLSettings:
    def __init__(self):
        # Obtain an OpenCL platform.
        self.platform = cl.get_platforms()[0]

        # Obtain a device id for at least one device (accelerator).
        self.device = self.platform.get_devices()[0]

        # Initialize context
        self.ctx = cl.Context([self.device])

        # Initialize command-queue
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Read kernels from .cl files
        with open('Kernels\NonlinearKernels.cl') as f:
            nonlinear_kernels = f.read()
        with open('Kernels\LinearKernels.cl') as f:
            linear_kernels = f.read()
        with open('Kernels\InitialDataKernels.cl') as f:
            initial_kernels = f.read()

        # Compile program
        self.initial_prg = cl.Program(self.ctx, initial_kernels).build()
        self.linear_prg = cl.Program(self.ctx, linear_kernels).build()
        self.nonlinear_prg = cl.Program(self.ctx, nonlinear_kernels).build()