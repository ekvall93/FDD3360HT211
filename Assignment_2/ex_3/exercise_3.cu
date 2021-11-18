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


    Particle *ParticlesCPU;
    Particle *ParticlesGPU;
    Particle *d_ParticlesGPU;
    ParticlesCPU = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
    ParticlesGPU = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));

    /* Init particle positions and velcoity */
    Vector P, V;
    for (int i =0; i<NUM_PARTICLES; i++) {
        P = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        V = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        ParticlesCPU[i] = {P, V};
        ParticlesGPU[i] = {P, V};
    }

    /* Start GPU session */
    double gpuStart = cpuSecond();
    
    cudaMalloc(&d_ParticlesGPU, NUM_PARTICLES*sizeof(Particle));
    cudaMemcpy(d_ParticlesGPU, ParticlesGPU, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);

    int GRID_SIZE = (NUM_PARTICLES + TPB - 1) / TPB; 

    for (int i =0; i<NUM_ITER; i++) {
        updateParticlesGPU<<<GRID_SIZE, TPB>>>(d_ParticlesGPU, NUM_PARTICLES, 1, 0.02);
    }

    cudaMemcpy(ParticlesGPU, d_ParticlesGPU, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);

    double gpuElapsed = cpuSecond() - gpuStart;


    /* Start CPU session */
    double cpuStart = cpuSecond();
    for (int i =0; i<NUM_ITER; i++) {
        updateParticlesCPU(ParticlesCPU, NUM_PARTICLES, 1, 0.02);
    };
    double cpuElapsed = cpuSecond() - cpuStart;

    /* Check that GPU and CPU session have the same result */
    float maxError = 0.0f;
    for (int i=0; i<NUM_PARTICLES; i++){
        maxError = max(maxError, 
            abs(ParticlesCPU[i].velocity.x - ParticlesGPU[i].velocity.x)
        );
        maxError = max(maxError, 
            abs(ParticlesCPU[i].velocity.y - ParticlesGPU[i].velocity.y)
        );
        maxError = max(maxError, 
            abs(ParticlesCPU[i].velocity.z - ParticlesGPU[i].velocity.z)
        );
        maxError = max(maxError, 
            abs(ParticlesCPU[i].position.x - ParticlesGPU[i].position.x)
        );
        maxError = max(maxError, 
            abs(ParticlesCPU[i].position.y - ParticlesGPU[i].position.y)
        );
        maxError = max(maxError, 
            abs(ParticlesCPU[i].position.z - ParticlesGPU[i].position.z)
        );
    }

    /* Verify the same result */
    if (maxError!=0.0f) {
        printf("Failed!!\n");
    }

    printf("%f\t%f\t%i\t%i\t%i\n", cpuElapsed, gpuElapsed, NUM_PARTICLES, NUM_ITER, TPB);
    



    free(ParticlesCPU);
    free(ParticlesGPU);
    cudaFree(d_ParticlesGPU);

    cudaProfilerStop();
}