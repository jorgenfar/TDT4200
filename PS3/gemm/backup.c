#include <complex.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>

     void intrinsic_multiply(complex float* result, complex float a, complex float b, complex float c, complex float d) {
        //This function multiplies a*c and b*d

        //Align Ar -> Ai -> Br -> Bi and Cr -> Ci -> Br -> Bi in memory by using two arrays.
        float arr1[4];
        float arr2[4];

        arr1[0] = crealf(a);
        arr1[1] = cimagf(a);
        arr1[2] = crealf(b);
        arr1[3] = cimagf(b);
        arr2[0] = crealf(c);
        arr2[1] = cimagf(c);
        arr2[2] = crealf(d);
        arr2[3] = cimagf(d);

        //Create register x, y, t1, t2
        __m128 x, y, t1, t2; 
        //Load A & B and C & D into 128-bit registers called x and y using the aligned load command.
        x = _mm_load_ps(arr1);
        y = _mm_load_ps(arr2); 
        //Create register t1 and set to (a.r, a.r, b.r, b.r);
        t1 = _mm_moveldup_ps(x);
        //Create register t2 and set to t1*y (a.r * c.r, a.r*c.i, b.r*d.r, b.r*d.i)
        t2 = _mm_mul_ps(t1, y);
        //Sets y to (c.i, c.r, d.i, d.r)
        y = _mm_shuffle_ps(y, y, 0xb1);
        //Sets t1 to (a.i, a.i, b.i, b.i) 
        t1 = _mm_movehdup_ps(x);
        //Sets t1 to t1*y (a.i*c.i, a.i*c.r, b.i*d.i, d.i*d.r)
        t1 = _mm_mul_ps(t1, y);
        //Sets x to a*c and b*d(a.r*c.r-a.i*c.i, a.r*c.i+a.i*c.r, b.r*d.r-b.i*d.i, b.r*d.i+b.i*d.r)
        x = _mm_addsub_ps(t2, t1);
        //Store values from 128-bits register as an array of 4 floats
        __attribute__ ((aligned (16))) 
        float res[4];            
        _mm_store_ps(&res[0], x);
        //Create two complex numbers to return
        complex float outAC;
        complex float outBD;
        //Set values of output numbers to values in res array
        __real__ outAC = res[0];
        __imag__ outAC = res[1];
        __real__ outBD = res[2];
        __imag__ outBD = res[3];

        result[0] = outAC;
        result[1] = outBD;
     }

void gemm(complex float* A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        int k,
        complex float alpha,
        complex float beta){

    
        for(int x = 0; x < n; x += 1){
            for(int y = 0; y < m; y += 1){  

            }
        }


    complex float result[2]; //(complex float*)malloc(sizeof(complex float)*2);
        for(int x = 0; x < n; x += 1){
        for(int y = 0; y < m; y += 1){
            C[y*n + x] *= beta;
            for(int z = 0; z < k; z += 2){ 
                intrinsic_multiply(result, A[y*k + z], A[y*k + (z+1)], B[z*n + x], B[(z+1)*n + x]);
                C[y*n + x] += alpha*result[0];
                C[y*n + x] += alpha*result[1];
            }
        }
    }

}






