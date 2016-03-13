import pyopencl as cl
import numpy
import os


def main():
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    N = numpy.int32(8)
    S = numpy.int32(4)
    mutableVar = numpy.zeros(1, dtype=numpy.float64)

    platform = cl.get_platforms()[0]

    # Obtain a device id for at least one device (accelerator).
    device = platform.get_devices()[0]

    # Initialize context
    ctx = cl.Context([device])

    # Initialize command-queue
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags

    field = numpy.zeros((S, N), dtype=numpy.float64)
    test_array = numpy.arange(1, 10, dtype=numpy.float64)
    test_array_len = numpy.int32(len(test_array))

    field_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=field)
    mutable_var_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mutableVar)
    test_array_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_array)

    test_prg = cl.Program(ctx, """
    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif

    #define PYOPENCL_DEFINE_CDOUBLE
    #include "pyopencl-complex.h"
    #pragma OPENCL EXTENSION cl_amd_printf : enable

    void Increment(__global double *a, int index)
    {
        a[index]++;
    }

    __kernel void FindMaxIteration(const unsigned int size, __global double* maxIterations,
                      __global double *iterations)
    {
        int x = get_global_id(0);

        double max = iterations[0];
        for(int i=1; i < size; i++)
        {
            Increment(iterations, i);
            if(iterations[i] > max)
            {
                max = iterations[i];
            }
        }
        maxIterations[0] = pow(max,3);

    }
    """).build()

    prg = cl.Program(ctx, """
    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #endif

    #define PYOPENCL_DEFINE_CDOUBLE
    #include "pyopencl-complex.h"
    #pragma OPENCL EXTENSION cl_amd_printf : enable

    __kernel void test(const unsigned int N, const unsigned int S, __global int* mutableVar,
                      __global double *a)
    {
        int x = get_global_id(0);
        int y = get_global_id(1);
        cdouble_t I = (cdouble_t)(0,1);

        int ind = x*N;

        printf((__constant char *)"%i: ", mutableVar[0]);

        for(int i=0; i < N; i++)
        {
            mutableVar[0]++;
        }
    }
    """).build()
    # prg.test(queue, (S, 1), None, N, S, mutable_var_buf, test_array_buf)

    test_prg.FindMaxIteration(queue, (1, ), None, test_array_len, mutable_var_buf, test_array_buf)

    cl.enqueue_copy(queue, test_array, test_array_buf)
    cl.enqueue_copy(queue, mutableVar, mutable_var_buf)
    print test_array
    print mutableVar

if __name__ == '__main__':
    main()