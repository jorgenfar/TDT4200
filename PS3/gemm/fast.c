#include <complex.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>

     void intrinsic_multiply(complex float* out, complex float* a, complex float* b, complex float* c, complex float* d) {
        //This function multiplies a*c and b*d

        //Create register x, y, t1, t2
        __m128 x, y, t1, t2; 
        //Load A & B and C & D into 128-bit registers called x and y using the aligned load command.
        x = _mm_load_ps((float*)a);
        y = _mm_load_ps((float*)c); 
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
        //Store values from 128-bits register in the out variable
        _mm_store_ps((float*)out, x);
     }

void gemm(complex float* A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        int k,
        complex float alpha,
        complex float beta){

        complex float* transposed = (complex float*)malloc(sizeof(complex float)*n*k);
        for(int z = 0; z < k; z++)
        {
            for(int x = 0; x < n; x++){  
                transposed[x*k + z] = B[z*n + x];
            } 
        }
        complex float* temp;
        temp = B;
        B = transposed;
        free(temp);

        complex float result[2]; 
        for(int x = 0; x < n; x++){
            for(int y = 0; y < m; y++){
             C[y*n + x] *= beta;
                for(int z = 0; z < k; z += 2){ 
                   intrinsic_multiply(result, &A[y*k + z], &A[y*k + (z+1)], &B[x*k + z], &B[x*k + (x+1)]);
                    C[y*n + x] += alpha*result[0];
                    C[y*n + x] += alpha*result[1];
                }
            }
        }

    }






