#include <mpi.h>

// Global variables from main.c
extern int imageSize;
extern int rank, size,
    dims[2],
    coords[2],
    periods[2],
    north, south, east, west,
    local_height, local_width,
    width, height, border;

extern MPI_Comm cart_comm;

extern MPI_Datatype border_row_t,
                    border_col_t,
                    diverg_vector,
                    local_pres_vector;

extern float* local_pres;
extern float* local_pres0;
extern float* pres;
extern float* local_diverg;
extern float* diverg;
