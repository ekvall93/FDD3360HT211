// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <string.h>
// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

const char *saxpy = 
"__kernel                                   \n"
"void saxpy(__global double *A,            \n"
"            __global double *B,            \n"
"            __global double *C,            \n"
"            const unsigned int n,          \n"
"            const float a)          \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    if (index < n){                        \n"
"       C[index] += a * A[index] + B[index];}    \n"
"}                                          \n";


void saxpy_cpu(const float *A, const float *B, float *restrict C, int n, float a) {
  for (int i = 0; i < n; ++i) {
    C[i] += a * A[i] + B[i];
  }
}


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }


int main(int argc, char** argv) {

    if (argc != 2 )
    {
        printf("You need to provide array size\n");
        return -1;
    }
    unsigned int n = atoi(argv[1]);

    int N_ITER = 100;
  // Length of vectors
  /* unsigned int n = 100000; */

  /* CPU version */
  float *A = (float *)malloc(n * sizeof(float));
  float *B = (float *)malloc(n * sizeof(float));
  float *C= (float *)malloc(n * sizeof(float));

  // GPU version
  double *h_A;
  double *h_B;
  double *h_C;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(double);

  h_A = (double*)malloc(bytes);
  h_B = (double*)malloc(bytes);
  h_C = (double*)malloc(bytes);


  const float a = 2.0F;

  for (int i = 0; i < n; ++i) {
    A[i] = 1.0F;
    B[i] = 1.0F;
    C[i] = 0.0F;

    h_A[i] = 1.0F;
    h_B[i] = 1.0F;
  }

  double cpuStart = cpuSecond();

  for (int i=0; i<N_ITER; i++) {
    saxpy_cpu(A, B, C, n, a);
  }
  
  double cpuElapsed = cpuSecond() - cpuStart;
  

  

  // Device buffers
  cl_mem d_A;
  cl_mem d_B;
  cl_mem d_C;


  
	cl_platform_id * platforms; cl_uint     n_platform;


	// Find OpenCL Platforms
 	cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
	platforms = (cl_platform_id *)alloca(sizeof(cl_platform_id) * n_platform);
 	err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);
	cl_device_id *device_list; cl_uint n_devices;
	err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices); CHK_ERROR(err);
  
	device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
	err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL); CHK_ERROR(err);
  
	cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);

  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
	
	cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy, NULL, &err);

	
	err = clBuildProgram(program, 1, device_list, 0, 0, 0); CHK_ERROR(err);
	
  cl_kernel kernel = clCreateKernel(program, "saxpy", &err);
 
  // Create the input and output arrays in device memory for our calculation
  d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);


  double gpuStart = cpuSecond();


  // Write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(command_queue, d_A, CL_TRUE, 0,
                                  bytes, h_A, 0, NULL, NULL);CHK_ERROR(err);

  err = clEnqueueWriteBuffer(command_queue, d_B, CL_TRUE, 0,
                                  bytes, h_B, 0, NULL, NULL);CHK_ERROR(err);
  // Set the arguments to our compute kernel
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);CHK_ERROR(err);
  err = clSetKernelArg(kernel, 4, sizeof(float), &a);CHK_ERROR(err);

  size_t globalSize, localSize;
 
  localSize = 64;
  globalSize = ceil(n/(float)localSize)*localSize;

	for (int i=0; i<N_ITER; i++) {
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL); CHK_ERROR(err);
  }

  err = clFlush(command_queue); CHK_ERROR(err);
  err = clFinish(command_queue); CHK_ERROR(err);


  // Read the results from the device
  clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0,
                              bytes, h_C, 0, NULL, NULL );


  double gpuElapsed = cpuSecond() - gpuStart;
  

  //Sum up vector c and print result divided by n, this should equal 1 within error
  double error = 0;
  for(int i=0; i<n; i++)
      error += h_C[i] - C[i];

  if (error!=0) {
    printf("Wrong result!");
  }
  


  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);

  err = clReleaseKernel(kernel); CHK_ERROR(err);
  err = clReleaseProgram(program); CHK_ERROR(err);
  err = clReleaseCommandQueue(command_queue); CHK_ERROR(err);
  err = clReleaseContext(context); CHK_ERROR(err);


  //release host memory
  free(h_A);
  free(h_B);
  free(h_C);

  free(A);
  free(B);
  free(C);
  
  printf("%f\t%f\t%d\n", cpuElapsed, gpuElapsed, n);
  
  return 0;
}

// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
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
