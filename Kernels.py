code = """
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define PYOPENCL_DEFINE_CDOUBLE
#include "pyopencl-complex.h"
#pragma OPENCL EXTENSION cl_amd_printf : enable


__kernel void Disp(
__global double* K,
__global cdouble_t* Field,
double dz, int S, int N)
{
    int x = get_local_id(0)+get_group_id(0)*get_local_size(0);
    int y = get_local_id(1)+get_group_id(1)*get_local_size(1);

    int ind = x + y*S;

    cdouble_t minusI = (cdouble_t)(0,-1);

    double CosK = cos(K[ind % N] * dz);
    double SinK = sin(K[ind % N] * dz);

    double re = cdouble_real(Field[ind]);
    double imag = cdouble_imag(Field[ind]);

    Field[ind] = (cdouble_t)(re * CosK - imag * SinK, re * SinK + imag * CosK);

    //Field[ind] = cdouble_mul(Field[ind], cdouble_exp(-K[ind % N]*dz));
}

__kernel void Diff(
__global cdouble_t* Field,
__global double* A1,
__global double* A2,
__global double* A3,
__global double* r,
__global double* dr,
__global double* D,
int S, int N, double dz)
{
    int x = get_global_id(0);

    cdouble_t ac[256];
    cdouble_t bc[256];
    cdouble_t gc[256];
    cdouble_t B[256];
    //cdouble_t E[256];

    cdouble_t C = (cdouble_t)(0, D[x] * dz);

    cdouble_t k = C;

    //E[0] = Field[x];
    B[0] = cdouble_rmul( 2.0 , cdouble_divider( cdouble_add(Field[x + N], -Field[x]) , (r[1] * r[1]) ) );
    B[S-1] = (cdouble_t)(0,0);

    ac[0] = cdouble_radd(1.0 , cdouble_mulr(k,A2[0]));
    bc[0] = cdouble_add(Field[x], cdouble_mul(C,B[0]));
    gc[0] = 0;

    for(int i = 1; i < S; i++)
    {

        if(i < S-1)
        {
            B[i] = (cdouble_t)(
                0.5 * (((2 + dr[i] / r[i]) * cdouble_real(Field[x + N*(i + 1)]) / dr[i + 1] + (2 - dr[i + 1] / r[i]) * cdouble_real(Field[x + N*(i - 1)]) / dr[i]) / (dr[i] + dr[i + 1]) +
                cdouble_real(Field[x + N*i]) * ((dr[i + 1] - dr[i]) / r[i] - 2) / (dr[i] * dr[i + 1])),
                0.5 * (((2 + dr[i] / r[i]) * cdouble_imag(Field[x + N*(i + 1)]) / dr[i + 1] + (2 - dr[i + 1] / r[i]) * cdouble_imag(Field[x + N*(i - 1)]) / dr[i]) / (dr[i] + dr[i + 1]) +
                cdouble_imag(Field[x + N*i]) * ((dr[i + 1] - dr[i]) / r[i] - 2) / (dr[i] * dr[i + 1]))
                             );
        }

        gc[i] = cdouble_divide(cdouble_mulr(k,A1[i]), ac[i-1]);
        ac[i] = cdouble_radd( 1.0 , cdouble_mulr(k, A2[i]) - cdouble_mul( cdouble_mulr(k, A3[i-1]), gc[i] ) );
        bc[i] = Field[x + N*i] + cdouble_mul(k, B[i]) - cdouble_mul(bc[i-1], gc[i]);
    }

    /*
    for(int i=0; i < S; i++)
    {
        Field[x + N*i] = bc[i];
    }
    */

    Field[x + N*(S-1)] = cdouble_divide(bc[S-1] , ac[S-1]);
    for(int i = S - 2; i >= 0; i--)
    {
        Field[x + N*i] = cdouble_divide( (bc[i] - cdouble_mul( cdouble_mulr(k, A3[i]) , Field[x + N*(i+1)]) ) , ac[i]);
    }
}


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

__kernel void ComputeB(
__global cdouble_t* B,
__global cdouble_t* Field,
__global double* r,
__global double* dr,
int S, int N)
{
    /*
    if(x == 13)
    {
        //printf((__constant char *)"%i: ", x);

        for(int i = 0; i < S; i++)
        {
            printf((__constant char *)"(%0.8f %0.8f)", cdouble_real(ac[i]), cdouble_imag(ac[i]));
        }

        //printf((__constant char *)"%s", "\r");
    }
    */
}

__kernel void ComputeD(
__global double* D,
__global double* Freq_scale,
double D_const, int N)
{
    int ind = get_local_id(0);

    if(fabs(Freq_scale[ind]) < 10e-8)
    {
        D[ind] = 0;
    }
    else
    {
        D[ind] = D_const / Freq_scale[ind];
    }
}

__kernel void Conj(
__global cdouble_t* Field,
int S)
{
    int x = get_local_id(0)+get_group_id(0)*get_local_size(0);
    int y = get_local_id(1)+get_group_id(1)*get_local_size(1);

    int ind = x + y*S;

    Field[ind] = cdouble_conj(Field[ind]);
}

__kernel void Norm(
__global cdouble_t* Field,
int S, int N)
{
    int x = get_local_id(0)+get_group_id(0)*get_local_size(0);
    int y = get_local_id(1)+get_group_id(1)*get_local_size(1);

    int ind = x + y*S;

    Field[ind] = (cdouble_t)(cdouble_real(Field[ind])/(double)N,cdouble_imag(Field[ind])/(double)N);
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
"""