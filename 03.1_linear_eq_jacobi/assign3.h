/* *****************************************************************************
 *
 *  @file    assign3.h
 *  @brief   common functions used in assignment 3
 *  
 *  @author  Pattreeya Tanisaro
 *
 * **************************************************************************** */

#ifndef ASSIGN3_H
#define ASSIGN3_H


#include <math.h>                       // for fabs(..)
#include <stdio.h> 	                   // import of the definitions of the C IO library
#include "commonc.h"                    // self-defined common c utility func

#define   ROOT_NODE       0


/*	**************************************************************
 * @func: print_usage
 * @desc: print out how to use this program
 * @params:
 * ======== 
 *	prog [in] => char* as execultable
 *
 *	************************************************************ */ 

void print_usage(char *prog)
{
	 printf("\nSolving Ax=b using Jacobi Iteration\n");
	 printf("----------------------------\n");
	 printf("Usage: %s [Square Matrix A] [Vector b] [tolerance] [Output x]\n",prog );
	 printf("Exmaples: %s MatrixA_8x8  VectorB_8x1 0.001 X_8x1.dat \n\n\n",prog);
}
 

/*	**************************************************************
 * @func: print_matrix
 * @desc: print out matrix A size mxn
 * @params:    
 * ========    
 *	 m [in] -> number of rows
 *  n [in] -> number of columns
 *  A [in] -> matrix A 
 *
 *	************************************************************ */ 
void  print_matrix(int m, int n, double** A)
{
    int   i,j;

    printf("Matrix: size = %d\n",n);
    for(i=0; i<m; ++i) {
        for(j=0; j<n; ++j) {
            printf("[%f] ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n-------------------\n");
}
/*	**************************************************************
 * @func:   print_vector
 * @desc:   print out vector file B, size nx1
 * @params:
 * ======== 
 * n [in] -> nx1 size of B
 *	B [in] -> vector B of size nx1
 *
 *	************************************************************ */ 
void print_vector(int n, double* B)
{
    int j;
    
    printf("Vector: size = %d\n",n);
    for(j=0; j<n; ++j) {
        printf("[%f]\n",  B[j]);
    }
    printf("------------------\n");
}


/*	**************************************************************
 * @func:  jacobi
 * @desc:  Jacobi method to solve Ax = b
 * @params:
 * ======== 
 * rank      [in] ->	rank of the processes
 * rows      [in] ->	number of rows
 * n         [in] ->	size n of matrix nxn
 * A_blk_vec [in] ->	sub vector A of size (rows*n) x 1
 *                   and its memory will be free!
 * B         [in] ->	vector B as pointer
 * eps       [in] ->	tolerance
 * 
 * @return:  X as pointer to double, don't forget to free the memory!
 *
 *	************************************************************ */ 

double* jacobi(int rank, int rows, int n, double* A_blk_vec, 
               double* B, double eps )
{
    double*  X;                       // X as answer to the solution, owned by all procs
    double*  X_prev;                  // previous X, owned by all procs
    double*  X_tmp;                   // temporary X to update X, owned by all procs
    double** A_blk;                   // block vector A, rows x n owned by all procs
    int      i,j;                     // miscellous variables
    int      nblk;                    // block = rows x rank
    int      nstep;                   // number of calculation steps
    double   sum_err;                 // summation of error for tolerance
    double   sum;                     // temporary summation
    static  int MAX_CALC_STEP = 10000; // maximum calculation steps
    
    // =====================================================
    // Allocate memory for calculation
    // =====================================================
    X = malloc( sizeof(double) *n );
    memset(X, 0, sizeof(double)*n );

    X_prev = malloc( sizeof(double) *n );
    memset(X_prev, 0, sizeof(double)*n );

    X_tmp = malloc( sizeof(double) *n );


    // =====================================================
    // to make it easy, we map A_blk_vec 1d to A_blk 2d
    // =====================================================
    A_blk = malloc( sizeof(double *) * rows);
    nstep = 0;

    for( i=0; i<rows; ++i ) {
        A_blk[i] = malloc( sizeof(double)*n );
        memcpy( A_blk[i], &A_blk_vec[i*n], n*sizeof(double) );
    }

    MPI_Barrier( MPI_COMM_WORLD );

    // ==================================================================
    // Jacobi to solving x from Ax = b
    // x[i] = ( b[i] - ( sum(a[i][j]*x[j]) <-> i!= j ) )/a[i][j]
    //
    // ==================================================================
    nblk = rank*rows;
    nstep = 0;

    while (1) {  
       ++nstep; 
     
       for(i=0; i<rows; ++i) { 
           sum = 0.0;
           for(j=0; j<n; ++j) {
               if ( i+nblk != j ) {  // Take care! This is a block multiplication
                   sum += A_blk[i][j]*X_prev[j];
               }
           }       
           X[nblk+i] = B[nblk+i] - sum;

           if ( fabs( A_blk[i][nblk+i] ) < eps/1000  ) {
               if (rank == 0) {
                   printf("Error: X[%d] at step %d devided by zeros!",nblk+i,nstep);
                   MPI_Abort( MPI_COMM_WORLD, 1); 
               }
           }

           X[nblk+i] = X[nblk+i]/A_blk[i][nblk+i];
       }      

       // ==================================================
       // update X to all processes
       // ==================================================
       MPI_Allgather( &(X[nblk]), rows, MPI_DOUBLE, 
                      X_tmp, rows, MPI_DOUBLE, MPI_COMM_WORLD);
       
       memcpy(&X[0], &X_tmp[0], n*sizeof(double)); 
       //print_vector(n, X);
       
       // ==================================================
       // calculate the sum of the error less than tolerance, 
       // then X converges
       // ==================================================

       sum_err = 0;
       for(j=0; j<n; ++j) {
           sum_err += fabs( X[j] - X_prev[j] );
           X_prev[j] = X[j] ;
       }

       if (sum_err < eps || nstep > MAX_CALC_STEP ) {
           break;
       }
   } // end while ===================================================
   
    
    // ==============================================================
    // clean up
    // ==============================================================
    free( X_prev );
    free( X_tmp );
    free( A_blk_vec );

    for(i=0; i<rows; ++i) {
        free( A_blk[i] );
    }
    free( A_blk );

    return X;
}

#endif 

//////////////////////////////EOF//////////////////////////////////
