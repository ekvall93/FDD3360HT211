#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

struct Vector
{
    float x, y, z;
};

struct Particle
{
    struct Vector position;
    struct Vector velocity;
};


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
};


void updateParticlesCPU(struct Particle *Particles, int n_particles, float dt, float acc) {
    for (int i =0; i<n_particles; i++) {
        Particles[i].velocity.x += dt*acc;
        Particles[i].velocity.y += dt*acc;
        Particles[i].velocity.z += dt*acc;

        Particles[i].position.x += Particles[i].velocity.x * dt;
        Particles[i].position.y += Particles[i].velocity.y * dt;
        Particles[i].position.z += Particles[i].velocity.z * dt;
    }
}

const char *updateParticlesGPU = 
"struct Vector                                                          \n"
"{                                                                      \n"
"   float x, y, z;                                                      \n"
"};                                                                     \n"
"struct Particle                                                        \n"
"{                                                                      \n"
"   struct Vector position;                                             \n"
"   struct Vector velocity;                                             \n" 
"};                                                                     \n"
"__kernel                                                               \n"
"void updateParticlesGPU(                                               \n"
"__global struct Particle *Particles,                                   \n"
"            const int n_particles,                                     \n"
"            const float dt,                                            \n"
"            const float acc) {                                         \n"
"    int index = get_global_id(0);                                      \n"
"    if (index < n_particles) {                                         \n"
"    Particles[index].velocity.x += dt*acc;                             \n"
"    Particles[index].velocity.y += dt*acc;                             \n"
"    Particles[index].velocity.z += dt*acc;                             \n"
"                                                                       \n"
"    Particles[index].position.x += Particles[index].velocity.x * dt;   \n"
"    Particles[index].position.y += Particles[index].velocity.y * dt;   \n"
"    Particles[index].position.z += Particles[index].velocity.z * dt;   \n"
"       }                                                               \n"
"}                                                                      \n";

int main(int argc, char** argv) {

    if (argc != 4 )
    {
        printf("You need to provide number of particles and number of iterations\n");
        return -1;
    }
    int NUM_PARTICLES = atoi(argv[1]);
    int NUM_ITER = atoi(argv[2]);
    size_t localSize = atoi(argv[3]);


    float dt = 1;
    float acc = 2;


    struct Particle *ParticlesCPU;
    struct Particle *ParticlesGPU;

    cl_mem d_ParticlesGPU;
    


    size_t bytes = NUM_PARTICLES*sizeof(struct Particle);
    ParticlesCPU = (struct Particle*)malloc(bytes);
    ParticlesGPU = (struct Particle*)malloc(bytes);

    
    for (int i =0; i<NUM_PARTICLES; i++) {
        struct Vector P = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        struct Vector V = {RandomFloat(0,10), RandomFloat(0,10), RandomFloat(0,10)};
        ParticlesCPU[i].position = P;
        ParticlesCPU[i].velocity = V;

        ParticlesGPU[i].position = P;
        ParticlesGPU[i].velocity = V;
    }

     double cpuStart = cpuSecond();
    for (int i =0; i<NUM_ITER; i++) {
        updateParticlesCPU(ParticlesCPU, NUM_PARTICLES, dt, acc);
    };
    double cpuElapsed = cpuSecond() - cpuStart;

    cl_platform_id * platforms; cl_uint     n_platform;
 	cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);

	platforms = (cl_platform_id *)alloca(sizeof(cl_platform_id) * n_platform);
 	err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);
    
	cl_device_id *device_list; cl_uint n_devices;
	err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices); CHK_ERROR(err);

	device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
	err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL); CHK_ERROR(err);
    
	cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
	cl_program program = clCreateProgramWithSource(context, 1,(const char **)&updateParticlesGPU, NULL, &err);
	err = clBuildProgram(program, 1, device_list, 0, 0, 0); CHK_ERROR(err);
	
    cl_kernel kernel = clCreateKernel(program, "updateParticlesGPU", &err);
    d_ParticlesGPU = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);

    double gpuStart = cpuSecond();


    err = clEnqueueWriteBuffer(command_queue, d_ParticlesGPU, CL_TRUE, 0,
                                  bytes, ParticlesGPU, 0, NULL, NULL);CHK_ERROR(err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_ParticlesGPU);CHK_ERROR(err);
    err  = clSetKernelArg(kernel, 1, sizeof(int), &NUM_PARTICLES);CHK_ERROR(err);
    err  = clSetKernelArg(kernel, 2, sizeof(float), &dt);CHK_ERROR(err);
    err  = clSetKernelArg(kernel, 3, sizeof(float), &acc);CHK_ERROR(err);


    size_t globalSize;
    globalSize = ceil(NUM_PARTICLES/(float)localSize)*localSize;

    for (int i =0; i<NUM_ITER; i++) {
        err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL); CHK_ERROR(err);
    }

    err = clFlush(command_queue); CHK_ERROR(err);
    err = clFinish(command_queue); CHK_ERROR(err);

    clEnqueueReadBuffer(command_queue, d_ParticlesGPU, CL_TRUE, 0,
                                bytes, ParticlesGPU, 0, NULL, NULL );
    
    double gpuElapsed = cpuSecond() - gpuStart;


    float maxError = 0.0f;
    for (int i=0; i<NUM_PARTICLES; i++){
        
        maxError = fmax(maxError, 
            abs(ParticlesCPU[i].velocity.x - ParticlesGPU[i].velocity.x)
        );
        maxError = fmax(maxError, 
            abs(ParticlesCPU[i].velocity.y - ParticlesGPU[i].velocity.y)
        );
        maxError = fmax(maxError, 
            abs(ParticlesCPU[i].velocity.z - ParticlesGPU[i].velocity.z)
        );
        maxError = fmax(maxError, 
            abs(ParticlesCPU[i].position.x - ParticlesGPU[i].position.x)
        );
        maxError = fmax(maxError, 
            abs(ParticlesCPU[i].position.y - ParticlesGPU[i].position.y)
        );
        maxError = fmax(maxError, 
            abs(ParticlesCPU[i].position.z - ParticlesGPU[i].position.z)
        );
    }

    /* Verify the same result */
    if (maxError!=0.0f) {
        printf("Failed!!\n");
    }

    printf("%f\t%f\t%i\t%i\t%zu\n", cpuElapsed, gpuElapsed, NUM_PARTICLES, NUM_ITER, localSize);
    clReleaseMemObject(d_ParticlesGPU);
    free(ParticlesCPU);
    free(ParticlesGPU);
}


const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALId_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALId_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALId_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}
