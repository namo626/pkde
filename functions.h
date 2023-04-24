#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>
#include <time.h>
#include <assert.h>



#define Nx (int)pow(2,18)
#define Ny Nx 
#define h (3.49*1 / pow((float)Ny, 0.333))



/**
 * Checks if the absolute value of elements in a float array are within a tolerance.
 * 
 * @param fs Pointer to the float array to be checked.
 */
void checkResult(float* fs);


/**
 * Prints the elapsed time in seconds between two clock_t values.
 * 
 * @param tic The start time clock_t value.
 * @param toc The end time clock_t value.
 */
void printElapsedTime(clock_t tic, clock_t toc);


/**
 * @brief Write output data to a file.
 *
 * This function writes output data to a file with the given filename.
 * The data to be written is provided in three float arrays: xs, ys, and fs,
 * each of size Nx. The data is formatted as comma-separated values (CSV)
 * with seven decimal places of precision for each float value.
 *
 * @param fname Pointer to a string containing the filename of the output file.
 * @param xs Pointer to a float array representing the xs data.
 * @param ys Pointer to a float array representing the ys data.
 * @param fs Pointer to a float array representing the fs data.
 * @param Nx Number of elements in the xs, ys, and fs arrays.
 */
void writeOutput(const char* fname, float* xs, float* ys, float* fs);

#endif
