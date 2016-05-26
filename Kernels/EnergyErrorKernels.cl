#if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by this OpenCL device."
#endif

#define PYOPENCL_DEFINE_CDOUBLE
#include "pyopencl-complex.h"


__kernel void CalculateEnergy(
__global cdouble_t *field,
__global double *r,
__global double *dr,
__global double *partialSums,
__local double *localSums,
int N,
int S)
{
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int local_id = get_local_id(0);
    int group_size = get_local_size(0);

    if(global_id / N == 0)
    {
        localSums[local_id] = field[global_id].x * field[global_id].x * r[0] * dr[1];
    }
    else if(global_id / N == S - 1)
    {
        localSums[local_id] = field[global_id].x * field[global_id].x * r[S-1] * dr[S-1];
    }
    else
    {
        localSums[local_id] = field[global_id].x * field[global_id].x * r[global_id / N] * (dr[global_id / N] + dr[(global_id / N) + 1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride = group_size / 2; stride > 0; stride >>= 1)
    {
        if(local_id < stride)
        {
            localSums[local_id] += localSums[local_id + stride];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_id == 0)
    {
        partialSums[group_id] = localSums[local_id];
    }
}


__kernel void Reduce(
__global double *x,
__global double *partialSums,
__local double *localSums)
{
    uint global_id = get_global_id(0);
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);

    localSums[local_id] = x[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint stride = group_size / 2; stride > 0; stride >>= 1)
    {
        if(local_id < stride)
        {
            localSums[local_id] += localSums[local_id + stride];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_id == 0)
    {
        partialSums[group_id] = localSums[local_id];
    }
}
