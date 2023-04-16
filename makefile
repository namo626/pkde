main: main.cpp functions.cpp functions.h
	g++ -o main -lm -g -O2 main.cpp functions.cpp -mavx2
