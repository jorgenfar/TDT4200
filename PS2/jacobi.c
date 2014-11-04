#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "global.h"
#include <omp.h>

// Indexing macro for local pres arrays
#define LP(row,col) ((row)+border)*(local_width + 2*border) + ((col) + border)
	
// Distribute the diverg (bs) from rank 0 to all the processes
void distribute_diverg(){
	//Every process calls a receive to their own local_diverg array, starting at index 0, and with source process being 0
	//A non-blocking receive is used because process 0 sends data to itself
	MPI_Request req;
	MPI_Irecv(&local_diverg[0], local_height*local_width, MPI_FLOAT, 0, 1, cart_comm, &req);
	//Process 0 iterates through all the dimensions in the dims array, starting at process 0, and sends a block of diverg_vector to each process 
	int proc = 0;
	if (rank == 0) {
		for (int i = 0; i<dims[0]; i++) {
			for (int j = 0; j<dims[1]; j++) {
				MPI_Send(&diverg[imageSize + 3 + j*local_width + (imageSize+2)*local_height*i], 1, diverg_vector, proc++, 1, cart_comm);
			}
		}
	}
	MPI_Wait(&req, MPI_STATUS_IGNORE);
}

// Gather the results of the computation at rank 0
void gather_pres(){
	//Each process (including process 0) sends their own local_pres to destination 0 using the local_pres_vector MPI datatype.
	//A non-blocking send is used to avoid deadlock because process 0 is sending to itself. 
	MPI_Request req;
	MPI_Isend(&local_pres[LP(0,0)], 1, local_pres_vector, 0, 1, cart_comm, &req);
	//Process 0 iterates through all the dimensions in the dims array and receives the data sent from each process, including itself
	//NB: The MPI dataype I created for distributing divergs is also used here (diverg_vector). This works because the same datatype functionality is required. 		
	int proc = 0;
	if (rank == 0) {
		for (int i = 0; i<dims[0]; i++) {
			for (int j = 0; j<dims[1]; j++) {
				MPI_Recv(&pres[imageSize + 3 + j*local_width + (imageSize+2)*local_height*i], 1, diverg_vector, proc++, 1, cart_comm, MPI_STATUS_IGNORE); 
			}
		}
	}
	MPI_Wait(&req, MPI_STATUS_IGNORE);	
}

// Exchange borders between processes during computation
void exchange_borders(){
	//Each process runs through it´s neighbours clockwise and sends the border to that neighbour.
	//A corresping receive is called after each send to avoid a deadlock. 
	
	if (north != -2) MPI_Send(&local_pres[LP(0,0)], 1, border_row_t, north, 1, cart_comm); 									//send border to north
	if (south != -2) MPI_Recv(&local_pres[LP(local_height,0)], 1, border_row_t, south, 1, cart_comm, MPI_STATUS_IGNORE);	//receive border from south

	if (east != -2) MPI_Send(&local_pres[LP(0,local_width - 1)], 1, border_col_t, east, 1, cart_comm);						//send border to east
	if (west != -2) MPI_Recv(&local_pres[LP(0,-1)], 1, border_col_t, west, 1, cart_comm, MPI_STATUS_IGNORE);				//receive border from west

	if (south != -2) MPI_Send(&local_pres[LP(local_height - 1,0)], 1, border_row_t, south, 1, cart_comm);					//send border to south	
	if (north != -2) MPI_Recv(&local_pres[LP(-1,0)], 1, border_row_t, north, 1, cart_comm, MPI_STATUS_IGNORE);				//receive border from north

	if (west != -2) MPI_Send(&local_pres[LP(0,0)], 1, border_col_t, west, 1, cart_comm);									//send border to west	
	if (east != -2) MPI_Recv(&local_pres[LP(0,local_width)], 1, border_col_t, east, 1, cart_comm, MPI_STATUS_IGNORE);		//receive border from east

}

// One jacobi iteration
void jacobi_iteration(){
	//Iterates through the image, and checks if it is looking at a pixel in the corner of the global image, 
	//an edge of the global image, or a normal pixel which is a pixel that is not at a global border.
	//This check is done by checking if the pixel is at an edge in the local array, and then checking if neighbours in that direction exist.  
	//The local_pres value we are currently looking at is then set to the average of the local pres0 values surrounding it,
	//minues the corresponding value from the local_diverg (bs) array. 

	//local_pres0 is set to the value of local_pres at the start of each iteration.
	memcpy(local_pres0, local_pres, (sizeof(float)*(local_width + 2*border)*(local_height+2*border)));
	
	for (int i = 0; i<local_height; i++) {
		for (int j = 0; j<local_width; j++) {
			//Currently looking at element in global top left corner?
			if (i == 0 && j == 0 && north == -2 && west  == -2) {
				local_pres[LP(i,j)] = (1.0/2)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Currently looking at element in global top right corner?
			else if (i == 0 && j == local_height - 1 && north == -2 && east == -2) {
				local_pres[LP(i,j)] = (1.0/2)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Currently looking at element in global bottom left corner?
			else if (i == local_height - 1 && j == 0 && south == -2 && west == -2 ) {
				local_pres[LP(i,j)] = (1.0/2)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Currently looking at element in global bottom right corner?
			else if (i == local_height - 1 && j == local_width - 1 && south == -2 && east == -2) {
				local_pres[LP(i,j)] = (1.0/2)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Currently looking at element at top of global array? 
			else if (i == 0 && north == -2)   {
				local_pres[LP(i,j)] = (1.0/3)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Currently looking at element at bottom of global array?
			else if (i == (local_height - 1) && south == -2)	{ 	
				local_pres[LP(i,j)] = (1.0/3)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Currently looking at leftmost element in global array?
			else if (j == 0 && west == -2) { 						 
				local_pres[LP(i,j)] = (1.0/3)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}		 
			//Currently looking at rightmost element in global array? 
			else if (j == (local_width - 1) && east == -2) { 		
				local_pres[LP(i,j)] = (1.0/3)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}
			//Base case.
			else {
				local_pres[LP(i,j)] = (1.0/4)*(local_pres0[LP(i + 1,j)] + local_pres0[LP(i - 1,j)] + local_pres0[LP(i,j + 1)] + local_pres0[LP(i,j - 1)] - local_diverg[i*local_width + j]);
			}	
		}
	}
}

// Solve linear system with jacobi method
void jacobi (int iter) {
	//Each rank initialize their own local_pres and local_pres0 arrays for each CFD iteration. 
	 #pragma omp parallel for 
	 for (int i=-1; i<local_height+1; i++) {
	 	if (rank == 0) printf("num threads: %i\n", omp_get_num_threads());
	 	for (int j=-1; j<local_width+1; j++) {
	 		local_pres0[LP(i,j)] = 0;
	 		local_pres[LP(i,j)] = 0;
	 	}
	 }
    distribute_diverg();
    //Jacobi iterations.
    for (int k=0; k<iter; k++) {
	   	exchange_borders();	
    	jacobi_iteration();
    }
    gather_pres();
}
