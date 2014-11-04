all : mc_cuda mc_cl

mc_cuda : mc.cu
	nvcc -arch=sm_20 mc.cu -o mc_cuda -lcudart -lGL -lGLU -lglut -lGLEW

mc_cl : mc.c clutil.c mc.cl
	gcc -std=c99 mc.c clutil.c -o mc_cl -lGL -lGLU -lglut -lGLEW -lOpenCL

clean:
	rm -f mc_cl mc_cuda
