#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#define Nx 8000
#define Ny 8000
#define h 12

float kernel(float z);
void SIMD_f(float* fs, float* xs, float* ys);
void unrolled_f(float* fs, float* xs, float* ys);
void slow_f(float* fs, float* xs, float* ys);

#endif
