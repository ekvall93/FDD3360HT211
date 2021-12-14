#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }


void saxpy_cpu(int n, float a, float *x, float *y, float *z)
{
    for (int i=0; i<n; i++) {
        z[i] += a*x[i] + y[i];
    } 
}

void saxpy_gpu(int n, float a, float *x, float *y, float * z)
{   
    #pragma acc data present(x[0:n], y[0:n], z[0:n])
    {  
        #pragma acc kernels loop independent
        for (int i=0; i<n; i++) {
            z[i] += a * x[i] + y[i];
        }
    }  
}

int main(int argc, char** argv) {

    int ARRAY_SIZE = atoi(argv[1]);
    
    float *x, *y, *z, *d_z;
    x = (float*)malloc(ARRAY_SIZE*sizeof(float));
    y = (float*)malloc(ARRAY_SIZE*sizeof(float));
    z = (float*)malloc(ARRAY_SIZE*sizeof(float));
    d_z = (float*)malloc(ARRAY_SIZE*sizeof(float));
    float a = 2.0F;

    for (int i =0; i<ARRAY_SIZE; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 0.0f;
        d_z[i] = 0.0f;
    }

    double cpuStart = cpuSecond();

    for (int i=0; i < 100; i++) {
        saxpy_cpu(ARRAY_SIZE, a, x, y, z);
 
    }
    
    double cpuElapsed = cpuSecond() - cpuStart;

    double gpuStart = cpuSecond();
    
    #pragma acc data copyin(x[0:ARRAY_SIZE], y[0:ARRAY_SIZE], d_z[0:ARRAY_SIZE]) copyout(d_z[0:ARRAY_SIZE])
    {   
        for (int i=0; i < 100; i++) {
            saxpy_gpu(ARRAY_SIZE, a, x, y, d_z);
        }
    }

    double gpuElapsed = cpuSecond() - gpuStart;
    
    float maxError = 0.0f;
    for (int i=0; i<ARRAY_SIZE; i++){
        maxError += abs(d_z[i]-z[i]);
    }

    if (maxError!=0.0f) {
        printf("Not correct!!\n"); 
    } 

    free(x);
    free(y);
    free(z);
    free(d_z);

    printf("%f\t%f\t%d\n", cpuElapsed, gpuElapsed, ARRAY_SIZE);
}