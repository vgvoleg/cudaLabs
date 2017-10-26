#include "blas_omp.h"
#include <omp.h>

void saxpy_omp(int n, float a, float *x, int incx, float *y, int incy) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if ((i*incy<n) && (i*incx<n)) {
            y[i*incy] += a*x[i*incx];
        }
    }
}
void daxpy_omp(int n, double a, double *x, int incx, double *y, int incy){
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if ((i*incy<n) && (i*incx<n)) {
            y[i*incy] += a*x[i*incx];
        }
    }
}