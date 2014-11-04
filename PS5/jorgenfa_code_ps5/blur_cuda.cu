#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//#include "bmp.h"
extern "C" void write_bmp(unsigned char* data, int width, int height);
extern "C" unsigned char* read_bmp(char* filename);
//#include "host_blur.h"
extern "C" void host_blur(unsigned char* inputImage, unsigned char* outputImage, int size);

void print_properties(){
	int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);

	cudaDeviceProp p;
	cudaSetDevice(0);
	cudaGetDeviceProperties (&p, 0);
	printf("Compute capability: %d.%d\n", p.major, p.minor);
	printf("Name: %s\n" , p.name);
	printf("\n\n");
}

__global__ void device_blur(unsigned char* input, unsigned char* output) {
    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i != 0 && j != 0 && i != 512 && j != 512) {
    	output[i*512 + j] = 0;
    	for(int k = -1; k < 2; k++){
    	    for(int l = -1; l < 2; l++){
    	        output[i * 512 + j] += (input[(i + k)*512 + (j + l)] / 9.0);
    	    }
    	}
    }
}


int main(int argc,char **argv) {
    //Prints some device properties, also to make sure the GPU works etc.
    print_properties();

    unsigned char* A = read_bmp("peppers.bmp");
    unsigned char* B = (unsigned char*)malloc(sizeof(unsigned char) * 512 * 512);

    //Allocate buffers for the input image and the output image on the device
    unsigned char* A_device;
    cudaMalloc((void**)&A_device, sizeof(unsigned char)*512*512);

    unsigned char* B_device;
    cudaMalloc((void**)&B_device, sizeof(unsigned char)*512*512);

    //Transfer the input image from the host to the device
    cudaMemcpy(A_device, A, sizeof(unsigned char)*512*512, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(unsigned char)*512*512, cudaMemcpyHostToDevice);

    //Launch the kernel which does the bluring
    //The grid consists of 4096 blocks, 64 threads per block
    dim3 grid(64, 64);
    dim3 block(8, 8);
    device_blur<<<grid, block>>>(A_device, B_device);

    //Transfer the result back to the host.
    cudaMemcpy(B, B_device, sizeof(unsigned char)*512*512, cudaMemcpyDeviceToHost);

    write_bmp(B, 512, 512);

	free(A);
	free(B);

	return 0;
}
