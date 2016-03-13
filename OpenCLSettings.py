import pyopencl as cl
from Kernels.InitialDataKernels import kernels as initialKernels
from Kernels.LinearKernels import kernels as linearKernels
from Kernels.NonlinearKernels import kernels as nonlinearKernels
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
        self.queue = cl.CommandQueue(self.ctx)

        # Compile program
        self.initial_prg = cl.Program(self.ctx, initialKernels).build()
        self.linear_prg = cl.Program(self.ctx, linearKernels).build()
        self.nonlinear_prg = cl.Program(self.ctx, nonlinearKernels).build()