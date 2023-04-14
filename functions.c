#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include "functions.h"


float kernel(float z) {
  return (1.0 - z*z/(h*h));
}

void slow_f(float* fs, float* xs, float* ys) {
  float s;
  for (int i = 0; i < Nx; i++) {
    s = 0.0;
    for (int j = 0; j < Ny; j++) {
      s += kernel(xs[i] - ys[j]);
    }
    fs[i] = s;
  }
}


void SIMD_f(float* fs, float* xs, float* ys) {

  __m256 hs = _mm256_set1_ps(h);
  __m256 ones = _mm256_set1_ps(1.0);

  for (int i = 0; i < Nx; i+=8) {
    __m256 f = _mm256_setzero_ps();
    __m256 x = _mm256_loadu_ps( &xs[i] );

    for (int j = 0; j < Ny; j++) {
      __m256 y = _mm256_broadcast_ss( &ys[j] );
      __m256 z = _mm256_sub_ps(x, y);

      /* using the kernel */
      z = _mm256_mul_ps(z,z);
      z = _mm256_div_ps(z, hs);
      z = _mm256_sub_ps(ones, z);
      f = _mm256_add_ps(f, z);
    }

    _mm256_storeu_ps(&fs[i], f);
  }
}

void unrolled_f(float* fs, float* xs, float* ys) {
  float s1, s2, s3, s4, s5;
  float s6, s7, s8, s9, s10;
  for (int i = 0; i < Nx; i+=10) {
    s1 = s2 = s3 = s4 = s5 = 0.0;
    s6 = s7 = s8 = s9 = s10 = 0.0;
    for (int j = 0; j < Ny; j++) {
      s1 += kernel(xs[i] - ys[j]);
      s2 += kernel(xs[i+1] - ys[j]);
      s3 += kernel(xs[i+2] - ys[j]);
      s4 += kernel(xs[i+3] - ys[j]);
      s5 += kernel(xs[i+4] - ys[j]);
      s6 += kernel(xs[i+5] - ys[j]);
      s7 += kernel(xs[i+6] - ys[j]);
      s8 += kernel(xs[i+7] - ys[j]);
      s9 += kernel(xs[i+8] - ys[j]);
      s10 += kernel(xs[i+9] - ys[j]);
    }
    fs[i] = s1;
    fs[i+1] = s2;
    fs[i+2] = s3;
    fs[i+3] = s4;
    fs[i+4] = s5;
    fs[i+5] = s6;
    fs[i+6] = s7;
    fs[i+7] = s8;
    fs[i+8] = s9;
    fs[i+9] = s10;
  }
}
