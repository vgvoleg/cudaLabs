#include "blas_cpu.h"

void saxpy_cpu(int n, float a, float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; i++) {
        if ((i*incy<n) && (i*incx<n)) {
            y[i*incy] += a*x[i*incx];
        }
    }
}
void daxpy_cpu(int n, double a, double *x, int incx, double *y, int incy){
    for (int i = 0; i < n; i++) {
        if ((i*incy<n) && (i*incx<n)) {
            y[i*incy] += a*x[i*incx];
        }
    }
}