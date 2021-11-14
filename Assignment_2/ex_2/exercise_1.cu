#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

__global__ void helloCuda() {
    printf("Hello from cuda device");
}

int main() {
    helloCuda << <1,1> > ();
    printf("Hello from CPU world...\n");
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}