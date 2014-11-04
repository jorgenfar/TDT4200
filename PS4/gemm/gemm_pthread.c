#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

float* A;
float* B;
float* C;
int m;
int n; 
int k;
float alpha = -2;
float beta = 1;
int nThreads = 1;

float rando(){
    return ((float)rand())/((float)RAND_MAX);
}


float* create_random_matrix(int m, int n){
    float* A = ( float*)malloc(sizeof( float)*m*n);

    for(int i = 0; i < m*n; i++){
        A[i] = rando();
    }

    return A;
}


void print_matrix( float* A, int m, int n){

    int max_size = 10;
    if(m > max_size || n > max_size){
        printf("WARNING: matrix too large, only printing part of it\n");
        m = max_size;
        n = max_size;
    }

    for(int y = 0; y < m; y++){
        for(int x = 0; x < n; x++){
            printf("%.4f  ", A[y*n + x]);
        }
        printf("\n");
    }
    printf("\n");
}

    void* worker(void* rank); //Thread function 

int main(int argc, char** argv){

    // Number of threads to 

    // Matrix sizes
    m = 2;
    n = 2;
    k = 2;

    // Reading command line arguments
    if(argc != 5){
        printf("useage: gemm nThreads m n k\n");
        exit(-1);
    }
    else{
        nThreads = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }

    // Initializing matrices
    A = create_random_matrix(m,k);
    B = create_random_matrix(k,n);
    C = create_random_matrix(m,n);

    // Performing computation
    long thread;
    pthread_t* thread_handles = (pthread_t*)malloc(nThreads * sizeof(pthread_t));

    for (thread = 0; thread < nThreads; thread++) {
        pthread_create(&thread_handles[thread], NULL, worker, (void*)thread);
    }

    for (thread = 0; thread< nThreads; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

    free(thread_handles);

    // Printing result
    // print_matrix(A, m,k);
    // print_matrix(B, k,n);
    // print_matrix(C, m,n);
}

void* worker(void* rank) {
    long my_rank = (long)rank;
    int startRow = my_rank * m/nThreads;
    int endRow = (my_rank + 1) * (m/nThreads) - 1;

    for (int x = startRow; x <= endRow; x++){
        for(int y = 0; y < m; y++){
            C[y*n + x] *= beta;
            for(int z = 0; z < k; z++){
                C[y*n + x] += alpha*A[y*k+z]*B[z*n + x];
            }
        }
    }
}
