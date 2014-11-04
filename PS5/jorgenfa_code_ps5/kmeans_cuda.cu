#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>

// Type for points
typedef struct{
    float x;    // x coordinate
    float y;    // y coordinate
    int cluster; // Cluster this point belongs to
} Point;

// Type for centroids
typedef struct{
    float x;    // x coordinate
    float y;    // y coordinate
    int nPoints; // Number of points in this cluster
} Centroid;

// Global variables
int nPoints;   // Number of points
int nClusters; // Number of clusters/centroids

Point* points;       // Array containig all points
Centroid* centroids; // Array containing all centroids


// Reading command line arguments
void parse_args(int argc, char** argv){
    if(argc != 3){
        printf("Useage: kmeans nClusters nPoints\n");
        exit(-1);
    }
    nClusters = atoi(argv[1]);
    nPoints = atoi(argv[2]);
}


// Create random point
Point create_random_point(){
    Point p;
    p.x = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.y = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.cluster = rand() % nClusters;
    return p;
}


// Create random centroid
Centroid create_random_centroid(){
    Centroid p;
    p.x = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.y = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
    p.nPoints = 0;
    return p;
}


// Initialize random data
// Points will be uniformly distributed
void init_data(){
    points = (Point*)malloc(sizeof(Point)*nPoints);
    for(int i = 0; i < nPoints; i++){
        points[i] = create_random_point();
        if(i < nClusters){
            points[i].cluster = i;
        }
    }

    centroids = (Centroid*)malloc(sizeof(Centroid)*nClusters);
    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }
}

// Initialize random data
// Points will be placed in circular clusters 
void init_clustered_data(){
    float diameter = 500.0/sqrt(nClusters);

    centroids = (Centroid*)malloc(sizeof(Centroid)*nClusters);
    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }

    points = (Point*)malloc(sizeof(Point)*nPoints);
    for(int i = 0; i < nPoints; i++){
        points[i] = create_random_point();
        if(i < nClusters){
            points[i].cluster = i;
        }
    }

    for(int i = 0; i < nPoints; i++){
        int c = points[i].cluster;
        points[i].x = centroids[c].x + ((float)rand() / (float)RAND_MAX) * diameter - (diameter/2);
        points[i].y = centroids[c].y + ((float)rand() / (float)RAND_MAX) * diameter - (diameter/2);
        points[i].cluster = rand() % nClusters;
    }

    for(int i = 0; i < nClusters; i++){
        centroids[i] = create_random_centroid();
    }
}


// Print all points and centroids to standard output
void print_data(){
    for(int i = 0; i < nPoints; i++){
        printf("%f\t%f\t%d\t\n", points[i].x, points[i].y, points[i].cluster);
    }
    printf("\n\n");
    for(int i = 0; i < nClusters; i++){
        printf("%f\t%f\t%d\t\n", centroids[i].x, centroids[i].y, i);
    }
}

// Print all points and centroids to a file
// File name will be based on input argument
// Can be used to print result after each iteration
void print_data_to_file(int i){
    char filename[15];
    sprintf(filename, "%04d.dat", i);
    FILE* f = fopen(filename, "w+");

    for(int i = 0; i < nPoints; i++){
        fprintf(f, "%f\t%f\t%d\t\n", points[i].x, points[i].y, points[i].cluster);
    }
    fprintf(f,"\n\n");
    for(int i = 0; i < nClusters; i++){
        fprintf(f,"%f\t%f\t%d\t\n", centroids[i].x, centroids[i].y, i);
    }

    fclose(f);
}


// Computing distance between point and centroid
float distance(Point a, Centroid b){
    float dx = a.x - b.x;
    float dy = a.y - b.y;

    return sqrt(dx*dx + dy*dy);
}


//Kernel to compute new centroid positions. One thread is created for each cluster. 
__global__ void new_centroids(Point* points, Centroid* centroids, int nPoints, int nClusters, float rand_x, float rand_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //Check that the amount of threads don't exceed the amount of points
    if (i < nClusters) {
        //Reset centroid position
        centroids[i].x = 0.0;
        centroids[i].y = 0.0;
        centroids[i].nPoints= 0;
        for(int j = 0; j < nPoints; j++){
            if (i == points[j].cluster) {
            centroids[i].x += points[j].x;
            centroids[i].y += points[j].y;
            centroids[i].nPoints++;
            }
        }
        //If a cluster lost all of its points, give it a new position
        if(centroids[i].nPoints == 0){
            Centroid p;
            p.x = rand_x;
            p.y = rand_y;
            p.nPoints = 0;
            centroids[i] = p;
        }
        else{
            //Divide the coordinates of each cluster by its amount of points. 
            centroids[i].x /= centroids[i].nPoints;
            centroids[i].y /= centroids[i].nPoints;
        }
    }
}

//Kernel to reassign points to the correct clusters. One thread is created for each point.
__global__ void reassign_points(Point* points, Centroid* centroids, int nPoints, int nClusters, int* updated) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //Check that the amount of threads don't exceed the amount of points
    if (i < nPoints) {
        float bestDistance = DBL_MAX;
        int bestCluster = -1;
        //For each cluster j...
        for(int j = 0; j < nClusters; j++){
            //Check distance between point i and cluster j
            float dx = points[i].x - centroids[j].x;
            float dy = points[i].y - centroids[j].y;
            float d = sqrt(dx*dx + dy*dy);
            //If distance to cluster j is smaller than distance measured so far to all other clusters 
            if(d < bestDistance){
                bestDistance = d;
                bestCluster = j;
            }
        }

        // If one point got reassigned to a new cluster, we have to do another iteration
        if(bestCluster != points[i].cluster){
            points[i].cluster = bestCluster;
            *updated = 1;
        }
        points[i].cluster = bestCluster;
    }
}

int main(int argc, char** argv){
    parse_args(argc, argv);

    // Create random data, either function can be used.
    //init_clustered_data();
    init_data();

    //Allocate memory on device for points
    Point* points_device;
    cudaMalloc((Point**)&points_device, sizeof(Point)*nPoints);

    //Allocate memory on device for centroids
    Centroid* centroids_device;
    cudaMalloc((Centroid**)&centroids_device, sizeof(Centroid)*nClusters);

    //Allocate memory on device for "updated" variable
    int* updated_device;
    cudaMalloc((int**)&updated_device, sizeof(int));

    //Copy points and centroids to device
    cudaMemcpy(points_device, points, sizeof(Point)*nPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_device, centroids, sizeof(Centroid)*nClusters, cudaMemcpyHostToDevice);

    // Iterate until no points are updated  
    int updated = 1;

    while(updated){
        updated = 0;
        float rand_x = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;
        float rand_y = ((float)rand() / (float)RAND_MAX) * 1000.0 - 500.0;

        int block_size = 256;
        //Calculate new centroid positions
        int grid_size_step_one = (nClusters/block_size) + 1; //This addition of 1 means I don't assume input as powers of 2
        new_centroids<<<grid_size_step_one, block_size>>>(points_device, centroids_device, nPoints, nClusters, rand_x, rand_y);

        //Reassign points to closest centroid
        int grid_size_step_two = (nPoints/block_size) + 1; //This addition of 1 means I don't assume input as powers of 2. 

        //Updated variable is copied to device before each iteration of reassign points...
        cudaMemcpy(updated_device, &updated, sizeof(int), cudaMemcpyHostToDevice);
        reassign_points<<<grid_size_step_two, block_size>>>(points_device, centroids_device, nPoints, nClusters, updated_device);
        //...And updated is copied back to host after each iteration
        cudaMemcpy(&updated, updated_device, sizeof(int), cudaMemcpyDeviceToHost);
    }

    //Copy points and centroids back to host                    
    cudaMemcpy(points, points_device, sizeof(Point)*nPoints, cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, centroids_device, sizeof(Centroid)*nClusters, cudaMemcpyDeviceToHost);

    // print_data();
}
