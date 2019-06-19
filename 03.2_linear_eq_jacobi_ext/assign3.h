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
 * @func: print_usage_extent
 * @desc: print out how to use this program for Assignment 3 task 2
 * @params:
 * ======== 
 *	prog [in] => char* as execultable
 *
 *	************************************************************ */ 

void print_usage_extent(char *prog)
{
	 printf("\nSolving Ax=b using Jacobi Iteration\n");
	 printf("-------------------------------\n");
	 printf("Usage: %s [Matrix A] [Vector b] [Tolerance] [Output file for x] [Task]\n\n",
           prog );
    printf("Task   'a' is reading A by rows using blocking collective operation");
    printf(" and reading b using non-blocking, non-collective opearion\n");
    printf("Task   'b' is reading A by columns using split collective operation");
    printf(" and reading b using split collective opearion with shared file pointer\n\n");

	 printf("Exmaples: %s MatrixA_8x8  VectorB_8x1 0.001 Output_a.dat a\n\n\n",prog);
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
 * @desc:  Jacobi method to solve Ax = b using data reading by rows
 * @params:
 * ======== 
 * rank      [in] ->	rank of the processes
 * rows      [in] ->	number of rows
 * n         [in] ->	size n of matrix nxn
 * A_blk_vec [in] ->	sub vector A of size (rows*n) x 1
 *                   and its memory will be free!
 * B         [in] ->	vector B as pointer, the memory will be
 *                   free by the calling function
 * eps       [in] ->	tolerance
 * 
 * @return:  X as pointer to double, don't forget to free its memory!
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

    // ===============================================================
    // Jacobi to solving x from Ax = b
    // x[i] = ( b[i] - ( sum(a[i][j]*x[j]) <-> i!= j ) )/a[i][j]
    //
    // ===============================================================
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
           if ( nstep > MAX_CALC_STEP ) {
               if ( rank == ROOT_NODE ) {
                   printf("Warning: The solution reaches the maximum step!\n");
                   printf("The system seems to be diverge!\n");
               }
           }
           break;
       }
   } // end while ====================================================
   
    
    // ===============================================================
    // clean up
    // ===============================================================
    free( X_prev );
    free( X_tmp );
    free( A_blk_vec );

    for(i=0; i<rows; ++i) {
        free( A_blk[i] );
    }
    free( A_blk );

    return X;
}



/*	**************************************************************
 * @func:  jacobi_by_column
 * @desc:  Jacobi method to solve Ax = b whereas A is reading as column
 * @params:
 * ======== 
 * rank      [in] ->	rank of the processes
 * rows      [in] ->	number of rows
 * n         [in] ->	size n of matrix nxn
 * A_blk_vec [in] ->	sub vector A of size (rows*n) x 1
 *                   and its memory will be free here!
 * B         [in] ->	vector B as pointer, 
 *                   B will be free by the calling function
 * eps       [in] ->	tolerance
 * 
 * @return:  X as pointer to double, don't forget to free the memory of X!
 *
 *	************************************************************ */ 

double* jacobi_by_column(int rank, int cols, int n, double* A_col_blk, 
                         double* B_blk, double eps )
{
    double*  X;                       // X as answer to the solution, data owned by all procs
    double*  X_prev;                  // previous X, data owned by all procs
    double*  X_tmp;                   // temporary X to update X, data owned by all procs
    double** A_blk;                   // block vector A, [n]x[cols] owned by all procs
    int      i,j;                     // miscellous variables
    int      nblk;                    // block = cols x rank
    int      nstep;                   // number of calculation steps
    double   sum_err;                 // summation of error for tolerance
    double*  sum;                     // used to store the sum of A and X
    double*  sum_tmp;                 // temp sum used as receive buf
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
    // To simplify the calculation, we map A_col_blk 1d to A_blk 2d
    // =====================================================
    A_blk = malloc( sizeof(double *) * n);
    nstep = 0;

    for( i=0; i<n; ++i ) {
        A_blk[i] = malloc( sizeof(double)*cols );
        memcpy( A_blk[i], &A_col_blk[i*cols], cols*sizeof(double) );
    }

    // ===============================================================
    //  ** Extra for reading by column **
    //  We save "partial" sum for each process
    // ===============================================================
    sum = malloc( sizeof(double) * n);

    sum_tmp = malloc( sizeof(double) * n);
    memset(sum_tmp, 0, sizeof(double) * n);

    // ===============================================================
    // Jacobi to solving x from Ax = b
    // x[i] = ( b[i] - ( sum(a[i][j]*x[j]) <-> i!= j ) )/a[i][j]
    //
    // ===============================================================
    nblk = rank*cols;
    nstep = 0;

    while (1) {  
       ++nstep; 
     
       // =======================================================
       //  n x cols = A_blk size for each proc, we calculate 
       //  partial sum of each row
       // =======================================================
       for(i=0; i<n; ++i) { 
           sum[i] = 0;
           for(j=0; j<cols; ++j) {
               if ( i != j+nblk ) {  
                   sum[i] += A_blk[i][j]*X_prev[j+nblk]; 
               }
           }       
       }
       
       // =======================================================
       //  For each process, til now we get only partial sum.
       //  Next, we have to collect "sum" from all processes
       //  to make "all sum" from all procs for matrix-vector multiplication"
       // =======================================================

       for (i = 0; i < n; i++ ) {
           MPI_Allreduce( &sum[i], &sum_tmp[i], 1, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD);
       }
       
       memcpy( sum, sum_tmp, n*sizeof(double) );

       // =======================================================
       //   We now calculate x (part) with B_blk and sum all
       // =======================================================

       for(i=0; i<cols; ++i) { 
           X[nblk+i] = B_blk[i] - sum[i+nblk];
       
           if ( fabs( A_blk[i+nblk][i] ) < eps/1000  ) { // prevent divided by zero!
               if (rank == 0) {
                   printf("Error: X[%d] at step %d devided by zeros!",nblk+i,nstep);
                   MPI_Abort( MPI_COMM_WORLD, 1); 
               }
           }

           X[nblk+i] = X[nblk+i]/A_blk[i+nblk][i];
       }      

       // ==================================================
       // update X to all processes
       // ==================================================
       MPI_Allgather( &(X[nblk]), cols, MPI_DOUBLE, 
                      X_tmp, cols, MPI_DOUBLE, MPI_COMM_WORLD);
       
       memcpy(&X[0], &X_tmp[0], n*sizeof(double)); 
       
       // ==================================================
       // calculate the sum of the error if it is less than 
       // tolerance, then X converges.
       // ==================================================

       sum_err = 0;
       for(j=0; j<n; ++j) {
           sum_err += fabs( X[j] - X_prev[j] );
           X_prev[j] = X[j] ;
       }

       if (sum_err < eps || nstep > MAX_CALC_STEP ) {
           if ( nstep > MAX_CALC_STEP ) {
               if ( rank == ROOT_NODE ) {
                   printf("Warning: The solution reaches the maximum step!\n");
                   printf("The system seems to be diverge!\n");
               }
           }
           break;
       }

   } // end while ====================================================
   
    MPI_Barrier( MPI_COMM_WORLD );

    // ===============================================================
    // Clean up heap
    // ===============================================================
    for(i=0; i<n; ++i) {
        free( A_blk[i] );
    }

    free( sum_tmp );
    free( sum );
    free( A_blk );
    free( X_tmp );
    free( X_prev );
    free( A_col_blk );
    
    return X;
}

#endif // ASSIGN3_H

//////////////////////////////EOF//////////////////////////////////
