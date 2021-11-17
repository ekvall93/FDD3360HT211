#include <curand.h>
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <assert.h>


__host__
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }


__global__ void setup_random_state(curandState *state) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    curand_init(123456789, index, 0, &state[index]);
}

__global__ void pi_monte_carlo(curandState *state, int *count, int m) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;

    extern __shared__ int cache[];
    cache[threadIdx.x] = 0;
    __syncthreads();

    int temp=0;
    while(temp < m) {
        float x = curand_uniform(&state[index]);
        float y = curand_uniform(&state[index]);
        float r = x*x + y*y;

        if (r <= 1){
            cache[threadIdx.x]++;
        }
        temp++;
    }

    // Reucde all counts to cache's first index
    int i = blockDim.x/2;
    while(i != 0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        i /= 2;
        __syncthreads();
    }
    //Sum upp all counts
    if (threadIdx.x == 0){
        atomicAdd(count, cache[0]);
    }
}


int main(int argc, char** argv) {

    if (argc != 4 )
    {
        printf("You need to provide number of particles and number of iterations\n");
        assert(false); // or return -1;
    }

    int TPB = atoi(argv[1]);
    int BLOCKS = atoi(argv[2]);
    unsigned int THREAD_ITER = atoi(argv[3]);
    
    unsigned int n = BLOCKS * TPB;
    unsigned int NUM_ITER = n * THREAD_ITER;

    int *count;
    int *d_count;
    curandState *d_state;
    float pi;

    /* allocate memory */
    count = (int*)malloc(n*sizeof(int));

    /* Start GPU session */
    double gpuStart = cpuSecond();

    cudaMalloc((void**)&d_count, n*sizeof(int));
    cudaMalloc((void**)&d_state, n*sizeof(curandState));
    cudaMemset(d_count, 0, sizeof(int));

    /* Set random state */
    setup_random_state<<< BLOCKS, TPB >>>(d_state);
    /* Run mote carlo */
    pi_monte_carlo<<< BLOCKS, TPB, TPB * sizeof(int)>>>(d_state, d_count, THREAD_ITER);
    cudaMemcpy(count, d_count, sizeof(int), cudaMemcpyDeviceToHost);


    double gpuElapsed = cpuSecond() - gpuStart;

    /* Approximate pi from count data */
    pi = *count*4.0/(n * THREAD_ITER);
    /* Validate result */
    if (abs(pi - M_PI) > M_PI / 100) {
        printf("Approximation is off!");
    }

    printf("%f\t%i\t%i\t%i\t%f\n",gpuElapsed, NUM_ITER, TPB, BLOCKS, pi); 
    free(count);
    cudaFree(d_count);
    cudaFree(d_state);
}