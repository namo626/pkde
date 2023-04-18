all: serial cuda

serial: main.cpp functions.cpp functions.h
	g++ -o main -lm -g -O2 main.cpp functions.cpp -mavx2 -fno-unroll-loops -fno-peel-loops

cuda: cuda.cu functions.h functions.cpp
	nvcc -o cuda -O2 cuda.cu -gencode arch=compute_50,code=sm_50 functions.cpp

clean:
	rm main cuda
