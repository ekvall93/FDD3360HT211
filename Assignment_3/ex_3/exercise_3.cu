#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <iostream> 

struct Vector
{
    float x, y, z;
};

struct Particle
{
    Vector position;
    Vector velocity;
};

__host__
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }

__host__ float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
};

__host__ void updateParticlesCPU(Particle *Particles, int n_particles, float dt, float acc) {
    for (int i =0; i<n_particles; i++) {
        Particles[i].velocity.x += dt*acc;
        Particles[i].velocity.y += dt*acc;
        Particles[i].velocity.z += dt*acc;


        Particles[i].position.x += Particles[i].velocity.x * dt;
        Particles[i].position.y += Particles[i].velocity.y * dt;
        Particles[i].position.z += Particles[i].velocity.z * dt;
    }
}

__global__ void updateParticlesGPU(Particle *Particles, int n_particles, float dt, float acc, int offset) {
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < n_particles) {
    
        Particles[i].velocity.x += dt*acc;
        Particles[i].velocity.y += dt*acc;
        Particles[i].velocity.z += dt*acc;


        Particles[i].position.x += Particles[i].velocity.x * dt;
        Particles[i].position.y += Particles[i].velocity.y * dt;
        Particles[i].position.z += Particles[i].velocity.z * dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 5 )
    {
        printf("You need to provide number of particles and number of iterations\n");
        assert(false); // or return -1;
    }
    int NUM_PARTICLES = atoi(argv[1]);
    int NUM_ITER = atoi(argv[2]);
    int TPB = atoi(argv[3]);
    int nStreams = atoi(argv[4]);


    cudaProfilerStart();
    Particle *ParticlesCPU;
    Particle *ParticlesGPU;
    Particle *d_ParticlesGPU;
    
    ParticlesCPU = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    
    cudaHostAlloc(&ParticlesGPU, NUM_PARTICLES*sizeof(Particle),cudaHostAllocDefault);
    cudaMalloc(&d_ParticlesGPU, NUM_PARTICLES*sizeof(Particle));
   
    /* Init particle positions and velcoity */
    Vector P, V;
    for (int i =0; i<NUM_PARTICLES; i++) {
        P = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        V = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        ParticlesGPU[i] = {P, V};
        ParticlesCPU[i] = {P, V};
    }


    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    const int streamSize = NUM_PARTICLES / nStreams;
    
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    
    int GRID_SIZE = (streamSize + TPB - 1) / TPB;
    
    for (int k =0; k<NUM_ITER; k++) {

        for (int i = 0; i < nStreams; ++i) {
            int offset = i * streamSize;
            cudaMemcpyAsync(&d_ParticlesGPU[offset], &ParticlesGPU[offset], streamSize * sizeof(Particle), cudaMemcpyHostToDevice, stream[i]);

            updateParticlesGPU<<<GRID_SIZE, TPB, 0, stream[i]>>>(d_ParticlesGPU, NUM_PARTICLES, 1, 0.02, offset);

            cudaMemcpyAsync(&ParticlesGPU[offset], &d_ParticlesGPU[offset], streamSize * sizeof(Particle), cudaMemcpyDeviceToHost, stream[i]);
            
          }

    }
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamSynchronize(stream[i]);
    }

    /* Didn't manage to get this to work :( */
    /* for (int k =0; k<NUM_ITER; k++) {

        for (int j = 0; j < nStreams; ++j)
            {
                int offset = j * streamSize;
                
                cudaMemcpyAsync(&d_ParticlesGPU[offset], &ParticlesGPU[offset], streamSize * sizeof(Particle), cudaMemcpyHostToDevice, stream[j]);
                cudaStreamSynchronize(stream[j]);

                
            }

        for (int i = 0; i < nStreams; ++i) {
            cudaStreamSynchronize(stream[i]);
        }
        
        for (int j = 0; j < nStreams; ++j)
            {
                int offset = j * streamSize;
                
                updateParticlesGPU<<<GRID_SIZE, TPB, 0, stream[j]>>>(d_ParticlesGPU, NUM_PARTICLES, 1, 0.02, offset);
                cudaStreamSynchronize(stream[j]);
                
            }

        for (int j = 0; j < nStreams; ++j)
            {
                int offset = j * streamSize;

                
                
                cudaMemcpyAsync(&ParticlesGPU[offset], &d_ParticlesGPU[offset], streamSize * sizeof(Particle), cudaMemcpyDeviceToHost, stream[j]);
                cudaStreamSynchronize(stream[j]);
                
            }
    } */

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    free(ParticlesCPU);
    cudaFreeHost(ParticlesGPU);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(d_ParticlesGPU);

    cudaProfilerStop();
}