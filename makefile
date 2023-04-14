main: main.c functions.c functions.h
	gcc -o main -lm -g -O2 main.c functions.c -mavx2
