#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>

#define Nx (int)pow(2,18)
#define Ny Nx
#define h (3.49*1 / pow((float)Ny, 0.333))

void writeOutput(char* fname, float* xs, float* ys, float* fs);

#endif
