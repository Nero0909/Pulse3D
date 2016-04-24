import pyopencl as cl
import numpy
import os


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    N = numpy.int32(8)
    S = numpy.int32(4)

    platform = cl.get_platforms()[0]

    # Obtain a device id for at least one device (accelerator).
    device = platform.get_devices()[0]

    # Initialize context
    ctx = cl.Context([device])

    # Initialize command-queue
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags

    field = numpy.arange(0, S*N, dtype=numpy.complex128).reshape((S, N))

    field_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=field)
    loc_buf = cl.LocalMemory(8*S)
    loc_buf_2 = cl.LocalMemory(8*S)

    test_prg = cl.Program(ctx, """
    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif

    #define PYOPENCL_DEFINE_CDOUBLE
    #include "pyopencl-complex.h"
    #pragma OPENCL EXTENSION cl_amd_printf : enable

    __kernel void Test(__global cdouble_t* field,
                       int N, int S)
    {
        int x = get_local_id(0)+get_group_id(0)*get_local_size(0);
        int y = get_local_id(1)+get_group_id(1)*get_local_size(1);

        int ind = x + y*S;

        field[ind].x = 10;
    }
    """).build()

    test_prg.Test(queue, field.shape, None, field_buf, N, S)

    print field
    print(""
          "---------------------------"
          "")

    cl.enqueue_copy(queue, field, field_buf)

    print field


if __name__ == '__main__':
    main()