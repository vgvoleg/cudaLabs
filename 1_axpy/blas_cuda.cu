#include "blas_cuda.h"
#include <cuda_runtime_api.h>

__global__ void saxpy_GPU(int n, float a, float *x, int incx, float *y, int incy) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if ( (i*incy < n) && (i*incx < n) ) {
    y[i*incy] += a*x[i*incx];
  }
}

__global__ void daxpy_GPU(int n, double a, double *x, int incx, double *y, int incy) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if ( (i*incy < n) && (i*incx < n) ) {
    y[i*incy] += a*x[i*incx];
  }
}

void saxpy_gpu(int n, float a, float *x, int incx, float *y, int incy) {
  float *x_gpu, *y_gpu;
  cudaMalloc((void**)&x_gpu, n*sizeof(float));
  cudaMalloc((void**)&y_gpu, n*sizeof(float));

  cudaMemcpy(x_gpu, x, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, n*sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  saxpy_GPU <<< num_blocks, BLOCK_SIZE >>> (n, a, x_gpu, incx, y_gpu, incy);

  cudaMemcpy(y, y_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(x_gpu);
  cudaFree(y_gpu);
}
