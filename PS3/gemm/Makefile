CFLAGS = -I. -I/share/apps/atlas/include -std=c99 -O3
LDFLAGS = -L/share/apps/atlas/lib
LDLIBS = -latlas -lcblas -lm
all: gemm_naive gemm_atlas gemm_fast

gemm_naive : gemm.c naive.o
	gcc $(CFLAGS) $(LDFLAGS) gemm.c naive.o -o gemm_naive $(LDLIBS) 

gemm_atlas : gemm.c atlas.o
	gcc $(CFLAGS) $(LDFLAGS) gemm.c atlas.o -o gemm_atlas  $(LDLIBS)

gemm_fast : gemm.c fast.o
	gcc $(CFLAGS) $(LDFLAGS) -std=c99 -O3 -msse3 gemm.c fast.o -o gemm_fast $(LDLIBS)

%.o : %.c
	gcc $(CFLAGS) -msse3 -c $<

clean:
	rm gemm_* *.o
