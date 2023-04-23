#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "functions.h"
#include <random>
#include <time.h>

#define threadsPerBlock 512
#define numBlocks (Nx / threadsPerBlock)
#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))
#define myKernel Gaussian_kernel

void HandleError( cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit( EXIT_FAILURE);
  }
}

__device__ float Epa_kernel(float z) {
  float out;
  if (fabsf(z) <= 1.0) {
    out = (3./4)*(1.0 - z*z);
  }
  else {
    out = 0.0;
  }

  return out;
}

__device__ float Gaussian_kernel(float z) {
  return (1/sqrtf(6.0)*expf(-z*z/2.0));
}

__device__ float CUDA_test(float z) {
  return 0.0;
}

#define TRIALS 1
__global__ void CUDA_f(float* fs, float* xs, float* ys) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float s = 0;
  //float x = xs[tid];

  for (int j = 0; j < Ny; j++) {
    s += myKernel((xs[tid] - ys[j]) / h );
  }


  fs[tid] = s / (h*Ny);
}

__global__ void CUDA_unrolled_f(float* fs, float* xs, float* ys) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float s1, s2, s3, s4, s5, s6, s7, s8;
  s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = 0;
  float x = xs[tid];

  for (int j = 0; j < Ny/8; j++) {
    s1 += myKernel((x - ys[8*j]) / h );
    s2 += myKernel((x - ys[8*j+1]) / h);
    s3 += myKernel((x - ys[8*j+2]) / h);
    s4 += myKernel((x - ys[8*j+3]) / h);
    s5 += myKernel((x - ys[8*j+4]) / h);
    s6 += myKernel((x - ys[8*j+5]) / h);
    s7 += myKernel((x - ys[8*j+6]) / h);
    s8 += myKernel((x - ys[8*j+7]) / h);

  }
  fs[tid] = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / (h*Ny);
}

__global__ void CUDA_16_unrolled_f(float* fs, float* xs, float* ys) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16;
  s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 =  0;
  s9 = s10 = s11 = s12 = s13 = s14 = s15 = s16 =  0;
  float x = xs[tid];

  for (int j = 0; j < Ny/16; j++) {
    s1 += myKernel((x - ys[8*j]) / h );
    s2 += myKernel((x - ys[8*j+1]) / h);
    s3 += myKernel((x - ys[8*j+2]) / h);
    s4 += myKernel((x - ys[8*j+3]) / h);
    s5 += myKernel((x - ys[8*j+4]) / h);
    s6 += myKernel((x - ys[8*j+5]) / h);
    s7 += myKernel((x - ys[8*j+6]) / h);
    s8 += myKernel((x - ys[8*j+7]) / h);
    s9 += myKernel((x - ys[8*j+8]) / h );
    s10 += myKernel((x - ys[8*j+9]) / h);
    s11 += myKernel((x - ys[8*j+10]) / h);
    s12 += myKernel((x - ys[8*j+11]) / h);
    s13 += myKernel((x - ys[8*j+12]) / h);
    s14 += myKernel((x - ys[8*j+13]) / h);
    s15 += myKernel((x - ys[8*j+14]) / h);
    s16 += myKernel((x - ys[8*j+15]) / h);

  }
  fs[tid] = (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8+s10+s11+s12+s13+s14+s15+s16) / (h*Ny);
}

#define BLOCK_SIZE 1*threadsPerBlock
__global__ void CUDA_shared_f(float* fs, float* xs, float* ys) {
  /* Shared memory storing y[i] */
  __shared__ float yy[BLOCK_SIZE];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int id = threadIdx.x;
  float x = xs[tid];
  int M = BLOCK_SIZE/threadsPerBlock;

  float s = 0;
  for (int i = 0; i < Ny; i+=BLOCK_SIZE) {
    for (int k = 0; k < M; k++) {
      yy[k*M + id] = ys[k*M + id + i];
    }
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; j++) {
      s += myKernel((x - yy[j]) / h);
    }

    __syncthreads();
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
  writeOutput("cuda_slow.csv", xs, ys, fs);

  CUDA_shared_f<<<numBlocks, threadsPerBlock>>>(fs_d, xs_d, ys_d);
  HANDLE_ERROR( cudaGetLastError() );

  CUDA_unrolled_f<<<numBlocks, threadsPerBlock>>>(fs_d, xs_d, ys_d);
  HANDLE_ERROR( cudaGetLastError() );

  CUDA_16_unrolled_f<<<numBlocks, threadsPerBlock>>>(fs_d, xs_d, ys_d);
  HANDLE_ERROR( cudaGetLastError() );
  //HANDLE_ERROR( cudaMemcpy(fs, fs_d, Nx*sizeof(float), cudaMemcpyDeviceToHost) );
  //writeOutput("cuda_fast.csv", xs, ys, fs);


  cudaFree(xs_d);
  cudaFree(ys_d);
  cudaFree(fs_d);


  return 0;

}
