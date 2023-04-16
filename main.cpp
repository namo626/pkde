#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <random>

int main(int argc, char *argv[]) {
  int option = atoi(argv[1]);
  /* Allocation */
  float* xs = (float*) malloc(Nx * sizeof(float));
  float* fs = (float*) malloc(Nx * sizeof(float));
  float* ys = (float*) malloc(Ny * sizeof(float));

  /* Range of values */
  float xmax = 10.0;
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
  }

  clock_t tic = clock();

  printf("Option = %d\n", option);
  switch(option) {
  case 1:
    printf("Using naive implementation\n");
    slow_f(fs,xs,ys);
    break;

  case 2:
    printf("Using loop unrolling\n");
    unrolled_f(fs, xs, ys);
    break;

  case 3:
    printf("Using SIMD\n");
    SIMD_f(fs, xs, ys);
    break;

  default:
    printf("Using naive implementation\n");
    slow_f(fs,xs,ys);
  }

  clock_t toc = clock();

  /* Check result */
  /* for (int i = 0; i < Nx; i++) { */
  /*   assert(fabs(fs[i]) <= 1e-8); */
  /* } */
  printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

  /* Write to csv. Format is xs, fs */
  FILE *fpt;
  fpt = fopen("output.csv", "w");
  for (int i = 0; i < Nx; i++) {
    fprintf(fpt, "%.5f, %.5f, %.5f\n", xs[i], ys[i], fs[i]);
  }

  fclose(fpt);


  return 0;
}
