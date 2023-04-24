#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <assert.h>
#include <math.h>
#include <random>
#include <immintrin.h>
#include <chrono>
#include "functions.h"

using namespace std;

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

//float Gaussian_kernel(float z) {
//  return (1/1.77*expf(-z*z/2.0));
//}

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
  __m256 hny = _mm256_set1_ps(h*Ny);
  __m256 ones = _mm256_set1_ps(1.0);
  __m256 twos = _mm256_set1_ps(2.0);
  __m256 sqpi = _mm256_set1_ps(1.77);

  for (int i = 0; i < Nx; i += 8)
  {
    __m256 f = _mm256_setzero_ps();
    __m256 x = _mm256_loadu_ps(&xs[i]);

    for (int j = 0; j < Ny; j++)
    {
      __m256 y = _mm256_broadcast_ss(&ys[j]);
      __m256 z = _mm256_sub_ps(x, y);
      z = _mm256_div_ps(z, hs);

      /* using the Epo kernel */
      z = _mm256_mul_ps(z, z);
      z = _mm256_sub_ps(ones, z);
      f = _mm256_add_ps(f, z);
      //
      /* using the Gaussian kernel */
      //z = _mm256_mul_ps(z, z);
      //z = _mm256_div_ps(z, twos);
      //z = _mm256_exp_ps(z);
      //z = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z));
      //f = _mm256_add_ps(f, z);

    }
    /* Divide by n*h */
    f = _mm256_div_ps(f, hny);

    /* Store result */
    _mm256_storeu_ps(&fs[i], f);
  }
}

void SIMD_unrolled_f(float *fs, float *xs, float *ys)
{

  __m256 hs = _mm256_set1_ps(h);
  __m256 hny = _mm256_set1_ps(h*Ny);
  __m256 ones = _mm256_set1_ps(1.0);
  __m256 twos = _mm256_set1_ps(2.0);
  __m256 sqpi = _mm256_set1_ps(1.77);

  for (int i = 0; i < Nx; i += 8)
  {
    __m256 f = _mm256_setzero_ps();
    __m256 f2 = _mm256_setzero_ps();
    __m256 x = _mm256_loadu_ps(&xs[i]);

    for (int j = 0; j < Ny; j += 2)
    {
      __m256 y = _mm256_broadcast_ss(&ys[j]);
      __m256 z = _mm256_sub_ps(x, y);
      z = _mm256_div_ps(z, hs);

      /* using the Epo kernel */
      z = _mm256_mul_ps(z, z);
      z = _mm256_sub_ps(ones, z);
      f = _mm256_add_ps(f, z);

      /* using the Gaussian kernel */
      //z = _mm256_mul_ps(z, z);
      //z = _mm256_div_ps(z, twos);
      //z = _mm256_exp_ps(z);
      //z = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z));
      //f = _mm256_add_ps(f, z);

      /* fetch next y */
      y = _mm256_broadcast_ss(&ys[j+1]);
      __m256 z2 = _mm256_sub_ps(x, y);
      z2 = _mm256_div_ps(z2, hs);

      /* using the Epo kernel */
      z2 = _mm256_mul_ps(z2, z2);
      z2 = _mm256_sub_ps(ones, z2);
      f2 = _mm256_add_ps(f2, z2);

      /* using the Gaussian kernel */
      //z2 = _mm256_mul_ps(z2, z2);
      //z2 = _mm256_div_ps(z2, twos);
      //z2 = _mm256_exp_ps(z2);
      //z2 = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z2));
      //f2 = _mm256_add_ps(f2, z2);
    }
    /* Divide by n*h */
    f =  _mm256_add_ps(f, f2);
    f = _mm256_div_ps(f, hny);

    /* Store result */
    _mm256_storeu_ps(&fs[i], f);
  }
}

void SIMD_unrolled_4_f(float *fs, float *xs, float *ys)
{

  __m256 hs = _mm256_set1_ps(h);
  __m256 hny = _mm256_set1_ps(h*Ny);
  __m256 ones = _mm256_set1_ps(1.0);
  __m256 twos = _mm256_set1_ps(2.0);
  __m256 sqpi = _mm256_set1_ps(1.77);

  for (int i = 0; i < Nx; i += 8)
  {
    __m256 f = _mm256_setzero_ps();
    __m256 f2 = _mm256_setzero_ps();
    __m256 f3 = _mm256_setzero_ps();
    __m256 f4 = _mm256_setzero_ps();
    __m256 x = _mm256_loadu_ps(&xs[i]);

    for (int j = 0; j < Ny; j += 4)
    {
      __m256 y = _mm256_broadcast_ss(&ys[j]);
      __m256 z = _mm256_sub_ps(x, y);
      z = _mm256_div_ps(z, hs);

      /* using the Gaussian kernel */
      z = _mm256_mul_ps(z, z);
      z = _mm256_div_ps(z, twos);
      //z = _mm256_exp_ps(z);
      z = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z));
      f = _mm256_add_ps(f, z);

      /* fetch next y */
      y = _mm256_broadcast_ss(&ys[j+1]);
      __m256 z2 = _mm256_sub_ps(x, y);
      z2 = _mm256_div_ps(z2, hs);

      /* using the Gaussian kernel */
      z2 = _mm256_mul_ps(z2, z2);
      z2 = _mm256_div_ps(z2, twos);
      //z2 = _mm256_exp_ps(z2);
      z2 = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z2));
      f2 = _mm256_add_ps(f2, z2);

      /* fetch next y */
      y = _mm256_broadcast_ss(&ys[j+2]);
      __m256 z3 = _mm256_sub_ps(x, y);
      z3 = _mm256_div_ps(z3, hs);

      /* using the Gaussian kernel */
      z3 = _mm256_mul_ps(z3, z3);
      z3 = _mm256_div_ps(z3, twos);
      //z3 = _mm256_exp_ps(z3);
      z3 = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z3));
      f3 = _mm256_add_ps(f3, z3);

      /* fetch next y */
      y = _mm256_broadcast_ss(&ys[j+3]);
      __m256 z4 = _mm256_sub_ps(x, y);
      z4 = _mm256_div_ps(z4, hs);

      /* using the Gaussian kernel */
      z4 = _mm256_mul_ps(z4, z4);
      z4 = _mm256_div_ps(z4, twos);
      //z4 = _mm256_exp_ps(z4);
      z4 = _mm256_div_ps(ones, _mm256_mul_ps(sqpi, z4));
      f4 = _mm256_add_ps(f4, z4);
    }
    /* Divide by n*h */
    f =  _mm256_add_ps(f, f2);
    f3 =  _mm256_add_ps(f3, f4);
    f =  _mm256_add_ps(f, f3);
    f = _mm256_div_ps(f, hny);

    /* Store result */
    _mm256_storeu_ps(&fs[i], f);
  }
}
void unrolled_f(float *fs, float *xs, float *ys)
{
  float s1, s2, s3, s4, s5;
  float s6, s7, s8;
  float y;
  for (int i = 0; i < Nx; i += 8)
  {
    s1 = s2 = s3 = s4 = s5 = 0.0;
    s6 = s7 = s8 = 0.0;
    for (int j = 0; j < Ny; j++)
    {
      y = ys[j];
      s1 += kernel((xs[i] - y) / h);
      s2 += kernel((xs[i + 1] - y) / h);
      s3 += kernel((xs[i + 2] - y) / h);
      s4 += kernel((xs[i + 3] - y) / h);
      s5 += kernel((xs[i + 4] - y) / h);
      s6 += kernel((xs[i + 5] - y) / h);
      s7 += kernel((xs[i + 6] - y) / h);
      s8 += kernel((xs[i + 7] - y) / h);
    }
    fs[i] = s1 / (h*Ny);
    fs[i + 1] = s2 / (h*Ny);
    fs[i + 2] = s3 / (h*Ny);
    fs[i + 3] = s4 / (h*Ny);
    fs[i + 4] = s5 / (h*Ny);
    fs[i + 5] = s6 / (h*Ny);
    fs[i + 6] = s7 / (h*Ny);
    fs[i + 7] = s8 / (h*Ny);
  }
}

void unrolled_16_f(float *fs, float *xs, float *ys)
{
  float s1, s2, s3, s4, s5;
  float s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16;
  float y;
  for (int i = 0; i < Nx; i += 16)
  {
    s1 = s2 = s3 = s4 = s5 = 0.0;
    s6 = s7 = s8 = s9 = s10 = s11 = s12 = s13 = s14 = 0.0;
    s15 = s16 = 0;
    for (int j = 0; j < Ny; j++)
    {
      y = ys[j];
      s1 += kernel((xs[i] - y) / h);
      s2 += kernel((xs[i + 1] - y) / h);
      s3 += kernel((xs[i + 2] - y) / h);
      s4 += kernel((xs[i + 3] - y) / h);
      s5 += kernel((xs[i + 4] - y) / h);
      s6 += kernel((xs[i + 5] - y) / h);
      s7 += kernel((xs[i + 6] - y) / h);
      s8 += kernel((xs[i + 7] - y) / h);
      s9 += kernel((xs[i + 8] - y) / h);
      s10 += kernel((xs[i + 9] - y) / h);
      s11 += kernel((xs[i + 10] - y) / h);
      s12 += kernel((xs[i + 11] - y) / h);
      s13 += kernel((xs[i + 12] - y) / h);
      s14 += kernel((xs[i + 13] - y) / h);
      s15 += kernel((xs[i + 14] - y) / h);
      s16 += kernel((xs[i + 15] - y) / h);
    }
    fs[i] = s1 / (h*Ny);
    fs[i + 1] = s2 / (h*Ny);
    fs[i + 2] = s3 / (h*Ny);
    fs[i + 3] = s4 / (h*Ny);
    fs[i + 4] = s5 / (h*Ny);
    fs[i + 5] = s6 / (h*Ny);
    fs[i + 6] = s7 / (h*Ny);
    fs[i + 7] = s8 / (h*Ny);
    fs[i + 8] = s9 / (h*Ny);
    fs[i + 9] = s10 / (h*Ny);
    fs[i + 10] = s11 / (h*Ny);
    fs[i + 11] = s12 / (h*Ny);
    fs[i + 12] = s13 / (h*Ny);
    fs[i + 13] = s14 / (h*Ny);
    fs[i + 14] = s15 / (h*Ny);
    fs[i + 15] = s16 / (h*Ny);
  }
}

int main(int argc, char *argv[])
{
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

  std::clock_t tic, toc;
  tic = clock();
  slow_f(fs, xs, ys);
  toc = clock();
  printf("Slow f - elapsed: %.7f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  writeOutput("slow.csv",xs, ys, fs);

  tic = std::clock();
  unrolled_f(fs, xs, ys);
  toc = std::clock();
  printf("Unrolled f - elapsed: %.7f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  //printf("Unrolled f: %.7f seconds\n", (long double)chrono::duration_cast<chrono::seconds>(end-begin).count());
  //writeOutput("unrolled.csv", xs, ys, fs);
  
  tic = std::clock();
  unrolled_16_f(fs, xs, ys);
  toc = std::clock();
  printf("Unrolled 16 f - elapsed: %.7f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

  tic = std::clock();
  SIMD_f(fs, xs, ys);
  toc = std::clock();
  printf("SIMD f - elapsed: %.7f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  //writeOutput("simd.csv", xs, ys, fs);

  tic = std::clock();
  SIMD_unrolled_f(fs, xs, ys);
  toc = std::clock();
  printf("SIMD + unrolling f - elapsed: %.7f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  //writeOutput("simd2.csv", xs, ys, fs);

  //tic = std::clock();
  //SIMD_unrolled_4_f(fs, xs, ys);
  //toc = std::clock();
  //printf("SIMD + 4 unrolling f - elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
  //writeOutput("simd4.csv", xs, ys, fs);
  return 0;
}
