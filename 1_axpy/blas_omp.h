#pragma once

void saxpy_omp(int n, float a, float *x, int incx, float *y, int incy);
void daxpy_omp(int n, double a, double *x, int incx, double *y, int incy);