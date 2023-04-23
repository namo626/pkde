all: serial cuda

serial: main.cpp functions.cpp functions.h
	g++ -o main -lm -g -O2 main.cpp functions.cpp -mavx2 -fno-unroll-loops -fno-peel-loops

cuda: cuda.cu functions.h functions.cpp
	nvcc -o cuda -O2 cuda.cu functions.cpp -Xptxas -dlcm=cg

clean:
	rm main cuda
