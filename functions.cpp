#include <stdio.h>
#include <stdlib.h>
#include "functions.h"

#include <stdio.h>
#include <stdlib.h>


void writeOutput(const char* fname, float* xs, float* ys, float* fs) {
    FILE* fpt;
    fpt = fopen(fname, "w");
    if (fpt == NULL) {
        printf("Failed to open file: %s\n", fname);
        return;
    }
    for (int i = 0; i < Nx; i++) {
        fprintf(fpt, "%.7f, %.7f, %.7f\n", xs[i], ys[i], fs[i]);
    }
    fclose(fpt);
}


void printElapsedTime(clock_t tic, clock_t toc) {
    double elapsedSeconds = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf("Elapsed: %f seconds\n", elapsedSeconds);
}


void checkFloatArray(float* fs) {
    for (int i = 0; i < Nx; i++) {
        assert(fabs(fs[i]) <= 1e-8);
    }
}