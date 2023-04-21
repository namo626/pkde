#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <random>
#include <immintrin.h>
#include <mpi.h>

float kernel(float z)
{
  if (abs(z) <= 1.0)
  {
    return (3. / 4) * (1.0 - z * z);
  }
  else
  {
    return 0.0;
  }
}

void slow_f(float *fs, float *xs, float *ys)
{
  float s;
  for (int i = 0; i < Nx; i++)
  {
    s = 0.0;
    for (int j = 0; j < Ny; j++)
    {
      s += kernel((xs[i] - ys[j]) / h);
    }
    fs[i] = s / (h * Ny);
  }
}

void SIMD_f(float *fs, float *xs, float *ys)
{

  __m256 hs = _mm256_set1_ps(h);
  __m256 ny = _mm256_set1_ps(Ny);
  __m256 ones = _mm256_set1_ps(1.0);

  for (int i = 0; i < Nx; i += 8)
  {
    __m256 f = _mm256_setzero_ps();
    __m256 x = _mm256_loadu_ps(&xs[i]);

    for (int j = 0; j < Ny; j++)
    {
      __m256 y = _mm256_broadcast_ss(&ys[j]);
      __m256 z = _mm256_sub_ps(x, y);
      z = _mm256_div_ps(z, hs);

      /* using the kernel */
      z = _mm256_mul_ps(z, z);
      z = _mm256_sub_ps(ones, z);
      f = _mm256_add_ps(f, z);
    }
    /* Divide by n*h */
    f = _mm256_div_ps(f, ny);

    /* Store result */
    _mm256_storeu_ps(&fs[i], f);
  }
}

void unrolled_f(float *fs, float *xs, float *ys)
{
  float s1, s2, s3, s4, s5;
  float s6, s7, s8, s9, s10;
  for (int i = 0; i < Nx; i += 10)
  {
    s1 = s2 = s3 = s4 = s5 = 0.0;
    s6 = s7 = s8 = s9 = s10 = 0.0;
    for (int j = 0; j < Ny; j++)
    {
      s1 += kernel(xs[i] - ys[j]);
      s2 += kernel(xs[i + 1] - ys[j]);
      s3 += kernel(xs[i + 2] - ys[j]);
      s4 += kernel(xs[i + 3] - ys[j]);
      s5 += kernel(xs[i + 4] - ys[j]);
      s6 += kernel(xs[i + 5] - ys[j]);
      s7 += kernel(xs[i + 6] - ys[j]);
      s8 += kernel(xs[i + 7] - ys[j]);
      s9 += kernel(xs[i + 8] - ys[j]);
      s10 += kernel(xs[i + 9] - ys[j]);
    }
    fs[i] = s1;
    fs[i + 1] = s2;
    fs[i + 2] = s3;
    fs[i + 3] = s4;
    fs[i + 4] = s5;
    fs[i + 5] = s6;
    fs[i + 6] = s7;
    fs[i + 7] = s8;
    fs[i + 8] = s9;
    fs[i + 9] = s10;
  }
}

void mpi_f(float *fs, float *xs, float *ys)
{
  int rank, size;

  // Initialize MPI
  MPI_Init(NULL, NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_Nx = Nx / size;                                    // Divide Nx among processors
  float *local_fs = (float *)malloc(local_Nx * sizeof(float)); // Allocate local memory for fs

  // Scatter xs to all processors
  float *local_xs = (float *)malloc(local_Nx * sizeof(float));
  MPI_Scatter(xs, local_Nx, MPI_FLOAT, local_xs, local_Nx, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Broadcast ys to all processors
  MPI_Bcast(ys, Ny, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Compute local fs
  for (int i = 0; i < local_Nx; i++)
  {
    float s = 0.0;
    for (int j = 0; j < Ny; j++)
    {
      s += kernel((local_xs[i] - ys[j]) / h);
    }
    local_fs[i] = s / (h * Ny);
  }

  // Gather local fs to fs on processor 0
  MPI_Gather(local_fs, local_Nx, MPI_FLOAT, fs, local_Nx, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Clean up local memory
  free(local_fs);
  free(local_xs);

  // Clean up and finalize MPI
  MPI_Finalize();
}

int main(int argc, char *argv[])
{
  int option = atoi(argv[1]);
  /* Allocation */
  float *xs = (float *)malloc(Nx * sizeof(float));
  float *fs = (float *)malloc(Nx * sizeof(float));
  float *ys = (float *)malloc(Ny * sizeof(float));

  /* Range of values */
  float xmax = 5.0;
  float xmin = -xmax;
  // srand(0);
  std::default_random_engine generator;
  std::normal_distribution<float> d(0, 1.0);

  /* Fill the sample array with random numbers */
  for (int i = 0; i < Ny; i++)
  {
    // ys[i] = 1.0;
    // ys[i] = xmax * (float)rand() / (float)(RAND_MAX);
    ys[i] = d(generator);
  }

  /* Do a linspace */
  float inc = (xmax - xmin) / (float)Nx;
  for (int i = 0; i < Nx; i++)
  {
    xs[i] = i * inc + xmin;
  }

  clock_t tic = clock();

  printf("Option = %d\n", option);
  switch (option)
  {
  case 1:
    printf("Using naive implementation\n");
    slow_f(fs, xs, ys);
    break;

  case 2:
    printf("Using loop unrolling\n");
    unrolled_f(fs, xs, ys);
    break;

  case 3:
    printf("Using SIMD\n");
    SIMD_f(fs, xs, ys);
    break;

  case 4:
    printf("Using MPI\n");
    mpi_f(fs, xs, ys);
    break;

  default:
    printf("Using naive implementation\n");
    slow_f(fs, xs, ys);
  }

  clock_t toc = clock();

  /* Check result */
  /* for (int i = 0; i < Nx; i++) { */
  /*   assert(fabs(fs[i]) <= 1e-8); */
  /* } */
  printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

  /* Write to csv. Format is xs, fs */
  writeOutput("mpi.csv", xs, ys, fs);

  return 0;
}
