
CC = mpicc 
CFLAGS = -std=c99 -g

SRC = fluid2d.c main.c bmp.c jacobi.c
BIN = fluid2d

fluid: $(BIN)

run: $(BIN)
	qrsh -cwd mpirun -n 8 $(BIN) 100 256
	
$(BIN) : $(SRC)

clean:
	rm -f $(BIN) *.o















