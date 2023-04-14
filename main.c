#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

int main(int argc, char *argv[]) {
  int option = atoi(argv[1]);
  /* Allocation */
  float* xs = malloc(Nx * sizeof(float));
  float* fs = malloc(Nx * sizeof(float));
  float* ys = malloc(Ny * sizeof(float));

  /* Fill the sample array with random numbers */
  for (int i = 0; i < Ny; i++ ) {
    ys[i] = 1.0;
  }
  for (int i = 0; i < Nx; i++ ) {
    xs[i] = 2.0;
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
  for (int i = 0; i < Nx; i++) {
    assert(fabs(fs[i]) <= 1e-8);
  }
  printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);



  return 0;
}
