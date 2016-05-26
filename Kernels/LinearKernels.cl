#if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by this OpenCL device."
#endif

#define PYOPENCL_DEFINE_CDOUBLE
#include "pyopencl-complex.h"


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

    cdouble_t ac[#space_size#];
    cdouble_t bc[#space_size#];
    cdouble_t gc[#space_size#];
    cdouble_t B[#space_size#];

    cdouble_t C = (cdouble_t)(0, D[x] * dz);
    cdouble_t k = C;

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