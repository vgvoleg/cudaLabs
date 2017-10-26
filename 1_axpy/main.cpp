#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include "blas_cpu.h"
#include "blas_omp.h"
#include "blas_cuda.h"

const int ARRAY_SIZE = 100000000;

float* createFloatArrFromIntArr(int* arrI) {
    float* arrF = new float[ARRAY_SIZE];
    for (int i = 0; i<ARRAY_SIZE; i++) {
        arrF[i] = (float) arrI[i];
    }
    return arrF;
}

double* createDoubleArrFromIntArr(int* arrI) {
    double* arrD = new double[ARRAY_SIZE];
    for (int i = 0; i<ARRAY_SIZE; i++) {
        arrD[i] = (double) arrI[i];
    }
    return arrD;
}

void printFArr (float *a) {
    printf("{");  
    for (int i = 0; i<ARRAY_SIZE-1; i++) {
        printf("%f,", a[i]);
    }
    printf("%f}\n",a[ARRAY_SIZE-1]);
}

void printDArr (double *a) {
    printf("{");  
    for (int i = 0; i< ARRAY_SIZE-1; i++) {
        printf("%f,", a[i]);
    }
    printf("%f}\n",a[ARRAY_SIZE-1]);
}

int main() {
    int *arr = new int[ARRAY_SIZE];
    for (int i = 0; i<ARRAY_SIZE; i++){
        arr[i] = i + 1;
    }

    clock_t start, finish;
    float time;

    /* CPU */
    float *arrF1_cpu = createFloatArrFromIntArr(arr);
    float *arrF2_cpu = createFloatArrFromIntArr(arr);

    start = clock();
    saxpy_cpu(ARRAY_SIZE, 1.0f, arrF1_cpu, 1, arrF2_cpu, 1);
    finish = clock();
    time = (float)(finish - start)/CLOCKS_PER_SEC;
    std::cout <<"Result on cpu: " << std::fixed << std::setprecision(3) << time << std::endl;
    //printFArr(arrF2_cpu);

    /* OpenMP */
    float *arrF1_omp = createFloatArrFromIntArr(arr);
    float *arrF2_omp = createFloatArrFromIntArr(arr);

    start = clock();
    saxpy_omp(ARRAY_SIZE, 1.0f, arrF1_omp, 1, arrF2_omp, 1);
    finish = clock();
    time = (float)(finish - start)/CLOCKS_PER_SEC;
    std::cout <<"Result on omp: " << std::fixed << std::setprecision(3) << time << std::endl;
    //printFArr(arrF2_omp);

    /* GPU */
    float *arrF1_gpu = createFloatArrFromIntArr(arr);
    float *arrF2_gpu = createFloatArrFromIntArr(arr);

    start = clock();
    saxpy_gpu(ARRAY_SIZE, 1.0f, arrF1_gpu, 1, arrF2_gpu, 1);
    finish = clock();
    time = (float)(finish - start)/CLOCKS_PER_SEC;
    std::cout <<"Result on gpu: " << std::fixed << std::setprecision(3) << time << std::endl;
    //printFArr(arrF2_gpu);




    return 0;
}
