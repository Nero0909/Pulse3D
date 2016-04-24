#if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by this OpenCL device."
#endif

#define PYOPENCL_DEFINE_CDOUBLE
#include "pyopencl-complex.h"
#pragma OPENCL EXTENSION cl_amd_printf : enable

double CalcEnext(
double k,
double* E_next,
double* E,
double* F,
int size)
{
    double temp, dist = 0;

    for(int i=0; i<size; i++)
    {
        double E_i_plus_1 = (i < size - 1) ? E[i + 1] : 0;
        double E_i_minus_1 = (i > 0) ? E[i - 1] : 0;

        temp = E_next[i];
        E_next[i] = F[i] +
            k * (pow(E[i] + E_i_minus_1, 3) - pow(E_i_plus_1 + E[i], 3));
        dist = dist + pow(E_next[i] - temp, 2);
    }

    return sqrt(dist);
}

void CalcE05dz(
double* E_05_dz,
double* E,
double* E_dz,
int size)
{
    for(int i=0; i<size; i++)
    {
        E_05_dz[i] = 0.5 * (E[i] + E_dz[i]);
    }
}

__kernel void CubicUnlinean1DSolve(
__global cdouble_t* F,
__global int* iterations,
double k, double dt, double max_err,
int iter, int size)
{
    int x = get_global_id(0);
    int ind = x*size;

    int max_iter = 50;
    int it = 1;

    double E_next[#time_size#];
    double E_05[#time_size#];
    double F_local[#time_size#];

    for(int i=0; i < size; i++)
    {
        F_local[i] = F[ind+i].x;
        E_next[0] = 0;
        E_05[0] = 0;
    }

    CalcEnext(k, E_next, F_local, F_local, size);

    for(it=1; it<max_iter; it++)
    {
        CalcE05dz(E_05, E_next, F_local, size);
        if(CalcEnext(k, E_next, E_05, F_local, size) < max_err && iter == 0)
            break;
        if (it == iter && iter != 0)
            break;
    }

    for(int i=0; i<size; i++)
    {
        F[ind+i].x = E_next[i];
    }

    iterations[x] = it;
}