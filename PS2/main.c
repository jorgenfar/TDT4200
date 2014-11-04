#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "config.h"
#include "global.h" 
#include "bmp.h"

struct Configuration config; // Stores arrays needed for the CFD computation, should not be modified

int iterations,              // Number of CFD iterations (not Jacobi iterations)
    imageSize;               // Width/height of the simulation domain/output image.
unsigned char* imageBuffer;  // Buffer to hold the image to be written to file

int rank,                      // Rank of this process
    size,                      // Total number of processes
    dims[2],                   // Process grid dimensions
    coords[2],                 // Process grid coordinates of current process
    periods[2] = {0,0},        // Periodicity of process grid
    north, south, east, west,  // Ranks of neighbours in process grid
    local_height, local_width, // Size of local subdomain
    border = 1;                // Border thickness

MPI_Comm cart_comm;  // Cartesian communicator

// MPI datatypes, you might need to add some
// remember to also include them in global.h to
// make them visible in other files
MPI_Datatype border_row_t,
             border_col_t, 
             diverg_vector,
             local_pres_vector;

// Global, local part of the pres array (stores the xs)
float* local_pres;
float* local_pres0;
float* pres;

// Global, local part of the diverg array (stores the bs)
float* local_diverg;
float* diverg;

// Function to create and commit MPI datatypes
void create_types() {
    //Vector to be used in the distribution of the divergence to the difference processors: diverg_vector
    MPI_Type_vector(local_height, local_width, local_width*dims[1]+2, MPI_FLOAT, &diverg_vector);
    MPI_Type_commit(&diverg_vector);

    //Vector to be used in border exchange for exchanging rows: border_row_t
    MPI_Type_contiguous(local_width, MPI_FLOAT, &border_row_t);
    MPI_Type_commit(&border_row_t);

    //Vector to be used in border exchange for exchanging columns: border_col_t
    MPI_Type_vector(local_height, 1, local_width, MPI_FLOAT, &border_col_t);
    MPI_Type_commit(&border_col_t);

    //Vector to be used for sending local_pres arrays in gather_pres() function
    MPI_Type_vector(local_height, local_width, local_width + 2, MPI_FLOAT, &local_pres_vector);
    MPI_Type_commit(&local_pres_vector);
}

int main (int argc, char **argv) {
    // Reading command line arguments
    iterations = 100;
    imageSize = 512;
    if(argc == 3){
        iterations = atoi(argv[1]);
        imageSize = atoi(argv[2]);
    }

    // MPI initialization, getting rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Creating cartesian communicator
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm );
    MPI_Cart_coords( cart_comm, rank, 2, coords );

    // Finding neighbours processes
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);


    // Determining size of local subdomain
    local_height = imageSize/dims[0];
    local_width = imageSize/dims[1];

    // Creating and commiting MPI datatypes for message passing
    create_types();

    // Allocating memory for local arrays
    local_pres = (float*)malloc(sizeof(float)*(local_width + 2*border)*(local_height+2*border));
    local_pres0 = (float*)malloc(sizeof(float)*(local_width + 2*border)*(local_height+2*border));
    local_diverg = (float*)malloc(sizeof(float)*local_width*local_height);

    // Initializing the CFD computation, only one process should do this.
    if(rank == 0){
        initFluid( &config, imageSize, imageSize);
        pres = config.pres;
        diverg = config.div;

        imageBuffer = (unsigned char*)malloc(sizeof(unsigned char)*imageSize*imageSize);
    }

    // Solving the CFD equations, one iteration for each timestep.
    // These are not the same iterations used in the Jacobi solver.
    // The solveFluid function call the Jacobi solver, wich runs for 
    // 100 iterations for each of these iterations.
    for(int i = 0; i < iterations; i++){
        solveFluid(&config);
    }

    // Converting the density to an image and writing it to file.
    if(rank == 0){
        densityToColor(imageBuffer, config.dens, config.N);
        write_bmp(imageBuffer, imageSize, imageSize);

        // Free fluid simulation memory
        freeFluid( &config );
    }
    
    // Finalize
    MPI_Finalize();
}
