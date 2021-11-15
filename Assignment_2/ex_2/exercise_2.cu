#include <stdio.h>
#include <sys/time.h>
#include <cstdlib>
#include <assert.h>


__global__
void saxpy(int n, float a, float *x, float *y)
{
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 if (i < n) y[i] = a*x[i] + y[i];
}

__host__
void saxpy_cpu(int n, float a, float *x, float *y)
{
    for (int i=0; i<n; i++) {
        if (i < n) y[i] = a*x[i] + y[i];
    } 
}

int main(int argc, char** argv) {

    if (argc != 2 )
    {
        printf("You need to provide array size\n");
        assert(false); // or return -1;
    }
    int ARRAY_SIZE = atoi(argv[1]);
    
    float *x, *y, *z, *d_x, *d_y; //z for CPU calcualtions
    x = (float*)malloc(ARRAY_SIZE*sizeof(float));
    y = (float*)malloc(ARRAY_SIZE*sizeof(float));
    z = (float*)malloc(ARRAY_SIZE*sizeof(float));
    float a = 2.3f;

    cudaMalloc(&d_x, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&d_y, ARRAY_SIZE*sizeof(float));

    for (int i =0; i<ARRAY_SIZE; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    /* Run saxpy on CPU */
    printf("Computing SAXPY on the CPU… ");
    saxpy_cpu(ARRAY_SIZE, a, x, z);
    printf("Done!\n");
    
    printf("Computing SAXPY on the GPU… ");
    saxpy<<<(ARRAY_SIZE+255)/256, 256>>>(ARRAY_SIZE, a, d_x, d_y);
    cudaMemcpy(y, d_y, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Done!\n");

    printf("Comparing the output for each implementation… ");

    float maxError = 0.0f;
    for (int i=0; i<ARRAY_SIZE; i++){
        maxError = max(maxError, abs(y[i]-z[i]));
    }

    if (maxError==0.0f) {
        printf("Correct!!\n");
    } else {
        printf("Not correct!!\n"); 
    }
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    free(z);
}