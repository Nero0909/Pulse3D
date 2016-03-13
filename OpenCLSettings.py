import Kernels
import pyopencl as cl
from Singleton import Singleton

@Singleton
class OpenCLSettings():
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
        self.prg = cl.Program(self.ctx, Kernels.code).build()