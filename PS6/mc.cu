#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "tables.h"

// OGL vertex buffer object
GLuint vbo;
struct cudaGraphicsResource *vbo_resource;

// Size of voxel grid
const int dim_x = 64;
const int dim_y = 64;
const int dim_z = 64;

float sim_time = 0.0;

// CUDA buffers
float* volume;
float4* vertices;

uint* edge_table;
uint* tri_table;
uint* num_verts_table;

//Block and grid size 
dim3 threadsPerBlock = dim3(8, 8, 8);
dim3 numBlocks = dim3(8, 8, 8);

//helper function on device for indexing 3d array
__device__ int arrayIndex(int x, int y, int z) {
	return x + (y*dim_x) + (z*dim_y*dim_x);
}

// Fill_volume kernel
__global__ void fill_volume(float* volume, float t, int dim_x, int dim_y, int dim_z){
    uint x = blockDim.x * blockIdx.x + threadIdx.x;
    uint y = blockDim.y * blockIdx.y + threadIdx.y;
    uint z = blockDim.z * blockIdx.z + threadIdx.z;
    uint threadId = x + dim_x*y + dim_x*dim_y*z;

	float dx = (float)x/dim_x;
	float dy = (float)y/dim_y;
	float dz = (float)z/dim_z;
	float f = abs(cos(0.01*t));
	volume[threadId] = f*(dx + dy + dz)/3;
	//printf("volume[threadId] = %f\n",f*(dx + dy + dz)/3);
}

// Get triangles kernel
__global__ void get_triangles(float* volume, float4* vertices, uint* tri_table, int dim_x, int dim_y, int dim_z){ 
    uint x = blockDim.x * blockIdx.x + threadIdx.x;
    uint y = blockDim.y * blockIdx.y + threadIdx.y;
    uint z = blockDim.z * blockIdx.z + threadIdx.z;
    uint threadId = x + dim_x*y + dim_x*dim_y*z;

    if (x < (dim_x-1) && y < (dim_y-1) && z < (dim_z-1)) {
        uint tableIndex = 0;
        tableIndex = (uint)(volume[arrayIndex(x,y,z)]<0.5);
        tableIndex += (uint)(volume[arrayIndex(x+1,y,z)]<0.5)*2;
        tableIndex += (uint)(volume[arrayIndex(x+1,y+1,z)]<0.5)*4;
        tableIndex += (uint)(volume[arrayIndex(x,y+1,z)]<0.5)*8;
        tableIndex += (uint)(volume[arrayIndex(x,y,z+1)]<0.5)*16;
        tableIndex += (uint)(volume[arrayIndex(x+1,y,z+1)]<0.5)*32;
        tableIndex += (uint)(volume[arrayIndex(x+1,y+1,z+1)]<0.5)*64;
        tableIndex += (uint)(volume[arrayIndex(x,y+1,z+1)]<0.5)*128;

		//printf("%d\n", tableIndex);

        for (int i = 0; i<15; i++) {
            float4 temp;
            //Check for the different cases in the tri_table array
            if (tri_table[16*tableIndex + i] == 255) {
                temp.x = 0.0;
                temp.y = 0.0;
                temp.z = 0.0;
                temp.w = 1.0;
				//printf("entered 255\n");
            }
            else if (tri_table[16*tableIndex + i] == 0) {

                temp.x = (((float)(x))/dim_x)+(1.0/(2*dim_x))+0.5/dim_x;
                temp.y = (((float)y)/dim_y)+0.5/dim_y;
                temp.z = (((float)z)/dim_z)+0.5/dim_z;
                temp.w = 1.0;       
				//printf("entered 0\n");
            }
            else if (tri_table[16*tableIndex + i] == 1) {
                temp.x = ((float)x/dim_x)+(1.0/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/(2*dim_y))+0.5/dim_y;
                temp.z = ((float)z/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 1\n");
            }
            else if (tri_table[16*tableIndex + i] == 2) {
                temp.x = ((float)x/dim_x)+(1.0/(2*dim_x))+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 2\n");
            }
            else if (tri_table[16*tableIndex + i] == 3) {
                temp.x = ((float)x/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/(2*dim_y))+0.5/dim_y;
                temp.z = ((float)z/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 3\n");
            }
            else if (tri_table[16*tableIndex + i] == 4) {
                temp.x = ((float)x/dim_x)+(1.0/(2*dim_x))+0.5/dim_x;
                temp.y = ((float)y/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 4\n");
            }
            else if (tri_table[16*tableIndex + i] == 5) {
                temp.x = ((float)x/dim_x)+(1.0/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/(2*dim_y))+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 5\n");
            }
            else if (tri_table[16*tableIndex + i] == 6) {
                temp.x = ((float)x/dim_x)+(1.0/(2*dim_x))+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 6\n");
            }
            else if (tri_table[16*tableIndex + i] == 7) {
                temp.x = ((float)x/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/(2*dim_y))+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/dim_z)+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 7\n");
            }
            else if (tri_table[16*tableIndex + i] == 8) {
                temp.x = ((float)x/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/(2*dim_z))+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 8\n");
            }
            else if (tri_table[16*tableIndex + i] == 9) {
                temp.x = ((float)x/dim_x)+(1.0/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/(2*dim_z))+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 9\n");
            }
            else if (tri_table[16*tableIndex + i] == 10) {
                temp.x = ((float)x/dim_x)+(1.0/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/(2*dim_z))+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 10\n");
            }
            else if (tri_table[16*tableIndex + i] == 11) {
                temp.x = ((float)x/dim_x)+0.5/dim_x;
                temp.y = ((float)y/dim_y)+(1.0/dim_y)+0.5/dim_y;
                temp.z = ((float)z/dim_z)+(1.0/(2*dim_z))+0.5/dim_z;
                temp.w = 1.0;
				//printf("entered 11\n");
            }
			//printf("vertices[%i].x = %f\n", i, temp.x);
			//printf("vertices[%i].y = %f\n", i, temp.y);
			//printf("vertices[%i].z = %f\n", i, temp.z);
			//printf("vertices[%i].w = %f\n", i, temp.w);
            vertices[threadId*15+i] = temp;
    	}
    }

}


// Set up and call get_triangles kernel
void call_get_triangles(){

    // CUDA taking over vertices buffer from OGL
    size_t num_bytes; 
    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&vertices, &num_bytes, vbo_resource);

    // Insert call to get_triangles kernel here
    get_triangles<<<numBlocks, threadsPerBlock>>>(volume, vertices, tri_table, dim_x, dim_y, dim_z);
	cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // CUDA giving back vertices buffer to OGL
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

// Set up and call fill_volume kernel
void call_fill_volume(float t){
    fill_volume<<<numBlocks, threadsPerBlock>>>(volume, t, dim_x, dim_y, dim_z);
	cudaDeviceSynchronize();
}


// Creating vertex buffer in OpenGL
void init_vertex_buffer(){
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, dim_x*dim_y*dim_z*15*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);	
  cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

// The display function is called at each iteration of the
// OGL main loop. It calls the kernels, and draws the result
void display(){
    sim_time+= 0.1;

    // Call kernels
    call_fill_volume(sim_time);
    call_get_triangles();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //Rotate camera
    glTranslatef(0.5,0.5,0.5);
    glRotatef(2*sim_time, 0.0, 0.0, 1.0);
    glTranslatef(-0.5,-0.5,-0.5);

    //Draw wireframe
    glTranslatef(0.5,0.5,0.5);
    glColor3f(0.0, 0.0, 0.0);
    glutWireCube(1);
    glTranslatef(-0.5,-0.5,-0.5);

    // Render vbo as buffer of points
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.7, 0.1, 0.3);
    glDrawArrays(GL_TRIANGLES, 0, dim_x*dim_y*dim_z*15);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

void init_GL(int *argc, char **argv){

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(512, 512);
    glutCreateWindow("CUDA Marching Cubes");
    glutDisplayFunc(display);

    glewInit();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat diffuse[] = {1.0,1.0,1.0,1.0};
    GLfloat ambient[] = {0.0,0.0,0.0,1.0};
    GLfloat specular[] = {1.0,1.0,1.0,1.0};
    GLfloat pos[] = {1.0,1.0,0.0,1.0};

    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

    glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE ) ;
    glEnable ( GL_COLOR_MATERIAL ) ;

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, 512, 512);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, 1, 0.1, 10.0);
    gluLookAt(1.5,1.5,1.5,0.5,0.5,0.5,0,0,1);
}



int main(int argc, char **argv) {
    
    // Setting up OpenGL
    init_GL(&argc, argv);

    // Setting up OpenGL on CUDA device 0
    cudaGLSetGLDevice(0);

    // Creating vertices buffer in OGL/CUDA
    init_vertex_buffer();

    // Allocate memory for volume
    cudaMalloc((float**)&volume, sizeof(float)*dim_x*dim_y*dim_z);

    // Allocate memory and transfer tables
    cudaMalloc((uint**)&edge_table, sizeof(uint)*256);
    cudaMalloc((uint**)&tri_table, sizeof(uint)*256*16);
    cudaMalloc((uint**)&num_verts_table, sizeof(uint)*256);

    cudaMemcpy(edge_table, edgeTable, sizeof(uint)*256, cudaMemcpyHostToDevice);
    cudaMemcpy(tri_table, triTable, sizeof(uint)*256*16, cudaMemcpyHostToDevice);
    cudaMemcpy(num_verts_table, numVertsTable, sizeof(uint)*256, cudaMemcpyHostToDevice);

    glutMainLoop();
	//sim_time = 0.1;
    //call_fill_volume(sim_time);
    //call_get_triangles();

}
