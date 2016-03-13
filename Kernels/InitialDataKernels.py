kernels = """
#if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#define PYOPENCL_DEFINE_CDOUBLE
#include "pyopencl-complex.h"
#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void ComputeA(
__global double* A1,
__global double* A2,
__global double* A3,
__global double* r,
__global double* dr,
int S)
{
    int ind = get_local_id(0);

    A1[0] = 0;
    A1[S-1] = 0;
    A2[0] = 0.5 * 4 / (r[1] * r[1]);
    A2[S - 1] = 1;
    A3[0] = -0.5 * 4 / (r[1] * r[1]);
    A3[S - 1] = 0;
    if(ind > 0 && ind < S-1)
    {
        A1[ind] = -0.5 * (2 - dr[ind + 1] / r[ind]) / ((dr[ind] + dr[ind + 1]) * dr[ind]);
        A2[ind] = -0.5 * ((dr[ind + 1] - dr[ind]) / r[ind] - 2) / (dr[ind + 1] * dr[ind]);
        A3[ind] = -0.5 * (2 + dr[ind] / r[ind]) / ((dr[ind] + dr[ind + 1]) * dr[ind + 1]);
    }
}
"""