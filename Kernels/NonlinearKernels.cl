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


double CalcEnext(
double k,
__global double* E_next,
__global double* E,
__global double* F,
int size, int ind)
{
    double temp, dist = 0;

    for(int i=0; i<size; i++)
    {
        double E_i_plus_1 = (i < size - 1) ? E[ind + i + 1] : 0;
        double E_i_minus_1 = (i > 0) ? E[ind + i - 1] : 0;

        temp = E_next[ind + i];
        E_next[ind+i] = F[ind+i] +
            k * (pow(E[ind+i] + E_i_minus_1, 3) - pow(E_i_plus_1 + E[ind+i], 3));
        dist = dist + pow(E_next[ind + i] - temp, 2);
    }

    return sqrt(dist);
}

void CalcE05dz(
__global double* E_05_dz,
__global double* E,
__global double* E_dz,
int size, int ind)
{
    for(int i=0; i<size; i++)
    {
        E_05_dz[ind+i] = 0.5 * (E[ind+i] + E_dz[ind+i]);
    }
}

__kernel void CubicUnlinean1DSolve(
__global double* F,
__global double* E_next,
__global double* E_05,
__global int* iterations,
double k, double dt, double max_err,
int iter, int size)
{
    int x = get_global_id(0);
    int ind = x*size;

    int max_iter = 50;
    int it = 1;

    CalcEnext(k, E_next, F, F, size, ind);
    for(it=1; it<max_iter; it++)
    {
        CalcE05dz(E_05, E_next, F, size, ind);
        if(CalcEnext(k, E_next, E_05, F, size, ind) < max_err && iter == 0)
            break;
        if (it == iter && iter != 0)
            break;
    }

    for(int i=0; i<size; i++)
    {
        F[ind+i] = E_next[ind+i];
    }

    iterations[x] = it;
}

__kernel void ComplexToDouble(
__global cdouble_t* Field,
__global double* Field_double,
int S, int N)
{
    int x = get_local_id(0)+get_group_id(0)*get_local_size(0);
    int y = get_local_id(1)+get_group_id(1)*get_local_size(1);
    int ind = x + y*S;

    Field_double[ind] = (double)(cdouble_real(Field[ind]));
}

__kernel void DoubleToComplex(
__global cdouble_t* Field,
__global double* Field_double,
int S, int N)
{
    int x = get_local_id(0)+get_group_id(0)*get_local_size(0);
    int y = get_local_id(1)+get_group_id(1)*get_local_size(1);
    int ind = x + y*S;

    Field[ind] = (cdouble_t)(Field_double[ind], cdouble_imag(Field[ind]));
}

__kernel void FindMaxIteration(const unsigned int size, __global int* maxIterations,
                  __global double *iterations)
{
    int x = get_global_id(0);

    double max = iterations[0];
    for(int i=1; i < size; i++)
    {
        if(iterations[i] > max)
        {
            max = iterations[i];
        }
    }
    maxIterations[0] = max;
}