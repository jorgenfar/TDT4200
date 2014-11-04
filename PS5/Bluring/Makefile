CFLAGS += -std=c99

all : blur_cuda


blur_cuda: bmp.o host_blur.o blur_cuda.cu
	nvcc -arch=sm_20  host_blur.o bmp.o blur_cuda.cu -o blur_cuda -lcudart
	
clean:
	rm *.o blur_cuda

run: 
	./blur_cuda
