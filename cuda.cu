#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "functions.h"
#include <random>

#define threadsPerBlock 1024
#define numBlocks (Nx / threadsPerBlock)
#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))

void HandleError( cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit( EXIT_FAILURE);
  }
}

__device__ float CUDA_kernel(float z) {
  float out;
  if (fabsf(z) <= 1.0) {
    out = (3./4)*(1.0 - z*z);
  }
  else {
    out = 0.0;
  }

  return out;
}

__global__ void CUDA_f(float* fs, float* xs, float* ys) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float s = 0;
  float x = xs[tid];
  float temp;

  for (int j = 0; j < Ny; j++) {
    s += CUDA_kernel((x - ys[j]) / h );
  }

  fs[tid] = s / (h*Ny);
}

int main() {
  float* xs = (float*) malloc(Nx * sizeof(float));
  float* fs = (float*) malloc(Nx * sizeof(float));
  float* ys = (float*) malloc(Ny * sizeof(float));

  float *xs_d, *fs_d, *ys_d;
  HANDLE_ERROR( cudaMalloc( (void**)&xs_d, Nx*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&fs_d, Nx*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&ys_d, Ny*sizeof(float) ) );

  /* Range of values */
  float xmax = 5.0;
  float xmin = -xmax;
  //srand(0);
  std::default_random_engine generator;
  std::normal_distribution<float> d(0, 1.0);

  /* Fill the sample array with random numbers */
  for (int i = 0; i < Ny; i++ ) {
    //ys[i] = 1.0;
    //ys[i] = xmax * (float)rand() / (float)(RAND_MAX);
    ys[i] = d(generator);
  }
  /* Do a linspace */
  float inc = (xmax - xmin) / (float)Nx;
  for (int i = 0; i < Nx; i++ ) {
    xs[i] = i*inc + xmin;
    fs[i] = i*inc + xmin;
  }

  HANDLE_ERROR( cudaMemcpy(xs_d, xs, Nx*sizeof(float), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(ys_d, ys, Ny*sizeof(float), cudaMemcpyHostToDevice) );

  /* Launch CUDA kernel */
  CUDA_f<<<numBlocks, threadsPerBlock>>>(fs_d, xs_d, ys_d);
  HANDLE_ERROR( cudaGetLastError() );

  HANDLE_ERROR( cudaMemcpy(fs, fs_d, Nx*sizeof(float), cudaMemcpyDeviceToHost) );

  cudaFree(xs_d);
  cudaFree(ys_d);
  cudaFree(fs_d);

  /* write output */
  writeOutput("cuda.csv", xs, ys, fs);

  return 0;

}