#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>

#define Nx 8000
#define Ny 8000
#define h (3.49*1 / pow((float)Ny, 0.333))

float kernel(float z);
void SIMD_f(float* fs, float* xs, float* ys);
void unrolled_f(float* fs, float* xs, float* ys);
void slow_f(float* fs, float* xs, float* ys);

#endif
