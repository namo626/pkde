all: serial cuda mpi

serial: main.cpp functions.cpp functions.h
	g++ -o $@ -lm -g -O2 main.cpp functions.cpp -mavx2 -fno-unroll-loops -fno-peel-loops

cuda: cuda.cu functions.h functions.cpp
	nvcc -o $@ -Xptxas -O2 cuda.cu functions.cpp 
mpi: main.cpp functions.cpp functions.h
	mpiicc -o main main.cpp functions.cpp
clean:
	rm -rf serial cuda
