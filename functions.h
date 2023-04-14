#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#define Nx 800000
#define Ny 80000
#define h 1.0

float kernel(float z);
void SIMD_f(float* fs, float* xs, float* ys);
void unrolled_f(float* fs, float* xs, float* ys);
void slow_f(float* fs, float* xs, float* ys);

#endif
