/* *****************************************************************************
 * Program:    jacobi.c                                                  
 * Author:     Pattreeya Tanisaro                    
 * Description:  Solving linear system using jacobi method 
 * ===========  Ax = Dx + (L+U)x = b
 *              x[i]  = (b[i] - sum(A[i][j]*x_prev[j]))/a[i][i]
 *              The solution converges if the distance between vector x(k)
 *              and x(k+1) is small enough.
 *	   
 * Parameters:
 * ==========
 * Input:        argv[1] => Matrix A
 *               argv[2] => Vector b
 *               argv[3] => tolerance of error (esp)
 * 
 * Output:        x  => solution of solving Ax=b
 *                   
 * *************************************************************************** */ 

#include "mpi.h" 	                // import of the MPI definitions
#include <stdio.h> 	                // import of the definitions of the C IO library
#include <string.h>                     // import of the definitions of the string operations
#include <unistd.h>	                // standard unix io library definitions and declarations
#include <errno.h>	                // system error numbers
#include <sys/time.h>	                // speicial system time functions for c
#include <stdlib.h>                     // avoid warning from malloc/free
#include "commonc.h"                    // definition of common c programming
#include "assign3.h"                    // functions for assignment 3


/* **************************************************************
 * @func:   open_n_read_vector
 * @desc:   Open and read vector from file
 * @params:
 * ========  
 * @return  B as pointer to double. 
 *          The caller owns the pointer and have responsibility to
 *          delete it.
 * ************************************************************ */ 

double* open_n_read_vector(char* filename, int n, double* B)
{
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fhB;
    int        err;
    char       buf[100];
    int        len;
    int        eclass;
    int        i;
    int        _n;
    

    // ======================================================
    // open file to read
    // ======================================================
    err = MPI_File_open( MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhB );
    
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    MPI_File_set_view( fhB, 0, MPI_DOUBLE, MPI_DOUBLE, 
                             "native", MPI_INFO_NULL);
    // ======================================================
    // get vector size and compare with the size of matrix
    // ======================================================
    MPI_File_get_size( fhB, &fsize);
    _n = (int)( fsize/sizeof(double) );
    if ( _n != n) {
        printf("Error matrix size[%dx%d] and vector size [%dx1]: not match%s\n", 
               n,n,_n);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    // ======================================================
    // read operation 
    // ======================================================

    err = MPI_File_read( fhB, B, n, MPI_DOUBLE, &status );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from reading file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhB );

    return B;
}


/* **************************************************************
 * @func:    open_n_read_matrix_1d
 * @desc:    Open and read matrix A as 1-D
 * @params:
 * ========
 * filename [in] -> matrix file A 
 * n        [in] -> int from matrix A nxn
 *
 * @return   pointer to double (1d vector) from  matrix file.
 *           The calling function has the responsibility to 
 *           free the memory!
 * ************************************************************ */ 

double* open_n_read_matrix_1d(char* filename,  int* n)
{
    int        err;
    int        i,j;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fhA; 
    char       buf[100];
    int        len;
    int        eclass;
    double*    _A;
    int        _n;

    // ====================================================== 
    // file to read
    // ======================================================

    err = MPI_File_open( MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhA );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    err = MPI_File_set_view( fhA, 0, MPI_DOUBLE, MPI_DOUBLE, 
                             "native", MPI_INFO_NULL);

    // ======================================================
    // get matrix size and  check if it is a square matrix
    // ======================================================

    err = MPI_File_get_size( fhA, &fsize);
    _n = (int)sqrt( (float)((int)fsize/sizeof(double)) );
    if ( (int)((int)fsize/sizeof(double))%_n  !=  0 ) {
        printf("Error: given matrix is not a square matrix!\n");
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }
    // ======================================================
    // read operation is much slower than memcpy
    // we try to read it once with _A_tmp and split the data to A later
    // ======================================================

    _A = malloc( sizeof(double) * _n * _n);
    err = MPI_File_read( fhA, _A, _n*_n, MPI_DOUBLE, &status );

    *n  = _n;

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhA );
    return _A;
}


/* **************************************************************
 * @func:    write_vector
 * @desc:    Write vector X to filename
 * @params:
 * ======== 
 * filename [in] -> matrix file A 
 * n        [in] -> int from matrix A nxn	
 * X        [in] -> answer of Ax=b as pointer to double 
 *
 * ************************************************************ */ 

void write_vector(char* filename, int n, double*  X)
{
    int        err;
    MPI_File   fh;
    MPI_Status status;
    char       buf[100];
    int        len;
    int        eclass;

    err = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDWR | MPI_MODE_CREATE,
                        MPI_INFO_NULL, &fh);
    if (err) {
        MPI_Error_string(err, buf, &len);
        printf("Error from open file to write %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }
    MPI_File_write(fh, X, n, MPI_DOUBLE, &status );
    MPI_File_sync( fh ); // force flush
    MPI_File_close( &fh );   
}

/* **************************************************************
 * @func:  main 
 * @desc:     
 * ====== 
 * - Open and read matrix A
 * - Open and read vector b 
 * - Scatter block A to each process
 * - Solve Ax=b using Jacobi method
 * - Write answer to file
 *	- Print out information
 *
 * ************************************************************ */ 
int main( int argc, char* argv[] ) 
{ 

    int      rank; 			      // rank of the process
    int      nprocs;                          // number of processes
    int      i,j;                             // misc variables
    int      err;                             // error handling
    double   eps;                             // tolerant
    int      n;                               // nxn Matrix and nx1 Vector
    int      rows;                            // stripped rows
    double*  A_vec;                           // matrix A in 1-D, owned by ROOT_NODE
    double*  A_blk_vec;                       // block matrix A as vector (1-D)
    double*  B;                               // vector b with size nx1 
    double*  X;                               // vector x with size nx1


    MPI_Init(&argc, &argv);		 		                        
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );             
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

   
    // ==============================================================
    // check valid inputs (passing arguments)
    // ==============================================================

   if ( argc != 5) {
       if ( rank == ROOT_NODE ) {
           print_usage( argv[0] );
       }
       MPI_Finalize();		
       return 0;	
   }
   eps = atof(argv[3]);
   
    // ==============================================================
    //   Open A and get size n and A as vector
    // ==============================================================
   if ( rank == ROOT_NODE ) {
       A_vec = open_n_read_matrix_1d(argv[1], &n);
   }
   MPI_Bcast( &n, 1, MPI_INT, ROOT_NODE, MPI_COMM_WORLD );
   if ( n%nprocs != 0 ) {
       printf("Error: n%nprocs is not 0!");
       MPI_Finalize();		
       return 0;	
   }
   rows = (int)(n/nprocs);
   //printf("n = %d rows = %d\n",n, rows);

   // ==============================================================
   //   Open vector file B and get B
   // ==============================================================
   B  = malloc( sizeof(double) * n);
   if ( rank == ROOT_NODE ) {
       B = open_n_read_vector(argv[2], n, B);
       //print_vector(n, B);
   }
   MPI_Bcast(B, n, MPI_DOUBLE, ROOT_NODE, MPI_COMM_WORLD );

   // ==============================================================
   //   Split A into a small block with size rows x n and 
   //   spread A from root node to other processes = MPI_Scatter A with rows x n
   // ==============================================================

   A_blk_vec = malloc( sizeof(double) * rows * n );

   MPI_Scatter ( A_vec,  rows * n, MPI_DOUBLE, 
                 A_blk_vec, rows * n, MPI_DOUBLE, ROOT_NODE, MPI_COMM_WORLD ); 

   // ==============================================================
   //  Solving Jacobi method to get X
   // ==============================================================

   X = jacobi(rank, rows, n, A_blk_vec, B, eps );
   

   // ==============================================================
   //   write answer, X, to file 
   // ==============================================================

   if ( rank == ROOT_NODE ) {
       write_vector(argv[4], n, X);
   }
   
   // ==============================================================
   //   print out information to stdout
   // ==============================================================
   if ( rank == ROOT_NODE ) {
       printf("====================================================\n" );
       printf("Solving a linear system with Jacobi Method\n");
       printf("=====================================================\n");
       printf("\nInput:\n");
       printf("=======\n");
       printf("Matrix file A:            %s\n", argv[1]);
       printf("Vector file b:            %s\n", argv[2]);
       printf("Tolerance:                %f\n", eps);
       printf("Output vector file x:     %s\n", argv[4]);
       printf("\nOutput:\n");
       printf("=======\n");
       printf("Vector size:             %dx1\n", n);
       for(j=0; j<n; ++j) {
           printf("X[%d]   =   [%.8lf]\n",  j, X[j]);
       }
       printf("\n");
       printf("====================================================\n\n" );
   }


   // ==============================================================
   //   clean up
   // ==============================================================
   if ( rank == ROOT_NODE ) {
       free ( B );
       free ( A_vec );
   }
   free( X );

   MPI_Finalize();		
   return 0;	
}

////////////////////////////////END OF FILE///////////////////////////
