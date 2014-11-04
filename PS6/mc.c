#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/glx.h>

#include "clutil.h"
typedef unsigned int uint;
#include "tables.h"


// OGL vertex buffer object
GLuint vbo;

// For OCL error codes
cl_int err;

// Various OCL variables
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;

// OCL kernels
cl_kernel fill_volume_kernel;
cl_kernel get_triangles_kernel;

// OCL Memory buffers
cl_mem volume;
cl_mem vertices;
cl_mem edge_table;
cl_mem tri_table;
cl_mem num_verts_table;

// Size of voxel grid
const int dim_x = 64;
const int dim_y = 64;
const int dim_z = 64;

float sim_time = 0.0;


// Set up and call fill_volume kernel
void fill_volume(){

}

// Set up and call get_triangles kernel
void get_triangles(){

    // OCL taking over vertices buffer from OGL
    glFlush();
    err = clEnqueueAcquireGLObjects(queue, 1, &vertices, 0,0,0);

	//Set up an call kernel here
    
    // OCL giving vertices buffer back to OGL
    err = clEnqueueReleaseGLObjects(queue, 1, &vertices, 0,0,0);
    err = clFlush(queue);
}

void init_vertex_buffer(){
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, dim_x*dim_y*dim_z*15*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Creating OCL buffer from OGL vbo
    vertices = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, vbo, &err);
	clError("Error creating vertices buffer",err);
}


// The display function is called at each iteration of the
// OGL main loop. It calls the kernels, and draws the result
void display(){
    sim_time+= 0.1;

    // Call kernels
    fill_volume();
    get_triangles();

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

    // Render vbo as buffer of vertices 
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.7, 0.1, 0.3);
    glDrawArrays(GL_TRIANGLES, 0, dim_x*dim_y*dim_z*15);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

// Set up OpenGL
void init_GL(int *argc, char **argv){

    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(512, 512);
    glutCreateWindow("OpenCL Marching Cubes");
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

// Set up OpenCL
void init_CL(){
    err = clGetPlatformIDs(1, &platform, NULL);
    clError("Couldn't get platform ID", err);
    printPlatformInfo(platform);


    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    clError("Couldn't get device ID", err);
    printDeviceInfo(device);

    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
    };
    context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    clError("Couldn't get context", err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    clError("Couldn't create command queue", err);
}



int main(int argc, char** argv) {

    // Setting up OpenGL
    init_GL(&argc, argv);

    // Setting up OpenCL
    init_CL();

    // Creating vertices buffer in OGL
    init_vertex_buffer();
    
    // Build kernels

    // Allocate memory for volume

    // Allocate memory and transfer tables


    glutMainLoop();
}
