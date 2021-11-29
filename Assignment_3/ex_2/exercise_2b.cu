#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>


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
        Particles[i].position.z += Particles[i].velocity.y * dt;
    }
}

__global__ void updateParticlesGPU(Particle *Particles, int n_particles, float dt, float acc) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_particles) {
        Particles[i].velocity.x += dt*acc;
        Particles[i].velocity.y += dt*acc;
        Particles[i].velocity.z += dt*acc;


        Particles[i].position.x += Particles[i].velocity.x * dt;
        Particles[i].position.y += Particles[i].velocity.y * dt;
        Particles[i].position.z += Particles[i].velocity.y * dt;
    }
}

int main(int argc, char** argv) {

    if (argc != 4 )
    {
        printf("You need to provide number of particles and number of iterations\n");
        assert(false); // or return -1;
    }
    int NUM_PARTICLES = atoi(argv[1]);
    int NUM_ITER = atoi(argv[2]);
    int TPB = atoi(argv[3]);


    cudaProfilerStart();
    Particle *ParticlesGPU;
    
    cudaMallocManaged(&ParticlesGPU, NUM_PARTICLES*sizeof(Particle));

    /* Init particle positions and velcoity */
    Vector P, V;
    for (int i =0; i<NUM_PARTICLES; i++) {
        P = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        V = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        ParticlesGPU[i] = {P, V};
    }

    /* Start GPU session */
    double gpuStart = cpuSecond();
    
    int GRID_SIZE = (NUM_PARTICLES + TPB - 1) / TPB; 

    for (int i =0; i<NUM_ITER; i++) {
        updateParticlesGPU<<<GRID_SIZE, TPB>>>(ParticlesGPU, NUM_PARTICLES, 1, 0.02);
    }
    cudaDeviceSynchronize();

    double gpuElapsed = cpuSecond() - gpuStart;

    printf("This took %f seconds\n", gpuElapsed);

    
    cudaProfilerStop();
}