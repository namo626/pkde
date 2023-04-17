#include <stdio.h>
#include <stdlib.h>
#include "functions.h"

void writeOutput(char* fname, float* xs, float* ys, float* fs) {

  FILE *fpt;
  fpt = fopen(fname, "w");
  for (int i = 0; i < Nx; i++)
  {
    fprintf(fpt, "%.7f, %.7f, %.7f\n", xs[i], ys[i], fs[i]);
  }

  fclose(fpt);

}
