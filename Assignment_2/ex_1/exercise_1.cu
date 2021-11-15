#include <stdio.h>

__global__ void cuda_hello(){
    int myId = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Hello World! My threadId is %d \n", myId);
 
}

int main() {
    cuda_hello<<<1,256>>>(); 
    cudaDeviceSynchronize();
    return 0;
}
