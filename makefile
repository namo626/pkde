all: serial cuda

serial: main.cpp functions.cpp functions.h
	icpc -o main -g -O2 main.cpp functions.cpp -mavx2 -fno-unroll-loops -lm

cuda: cuda.cu functions.h functions.cpp
	nvcc -o cuda -O2 cuda.cu functions.cpp -Xptxas -dlcm=cg

clean:
	rm main cuda
