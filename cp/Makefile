CC = /usr/bin/nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -g
SOURSES = kp.cu
BIN = kp
all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)
