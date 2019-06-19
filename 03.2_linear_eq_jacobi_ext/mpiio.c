/* *****************************************************************************
 * Program:    mpiio.c                                                  
 * Author:     Pattreeya Tanisaro                    
 * Description:  Solving linear system using jacobi method 
 * ===========              Ax = Dx + (L+U)x = b
 *                       x[i]  = (b[i] - sum(A[i][j]*x_prev[j]))/a[i][i]
 *               The solution converges if the distance between vector x(k)
 *               and x(k+1) is small enough.
 *	   
 * Parameters:
 * ==========
 *               
 * Input:        argv[1] => Matrix A file
 *               argv[2] => Vector b file
 *               argv[3] => tolerance of error (esp)
 *               argv[4] => Vector x file as output file 
 *               argv[5] => Task 2 a or b
 * 
 * Output:        x  => solution of solving Ax=b 
 *                   => also as output file as specify in arg[4]
 *                   
 * *************************************************************************** */ 

#include "mpi.h" 	                // import of the MPI definitions
#include <stdio.h> 	                // import of the definitions of the C IO lib
#include <string.h>                 // import of the definitions of the string op
#include <unistd.h>	                // standard unix io library 
#include <errno.h>	                // system error numbers
#include <sys/time.h>	            // speicial system time functions for c
#include <stdlib.h>                 // avoid warning from malloc/free
#include "commonc.h"                // definition of common c programming
#include "assign3.h"                // functions for assignment 3




/* **************************************************************
 * @func:   read_vector_nblkncoll_individual
 * @desc:   Open and read vector from file using non-blocking
 *          collective with individual file pointer
 *
 * @note:   task (a)
 * @params:
 * ========
 * filename [in] -> matrix file A 
 * n        [in] -> int getting from reading matrix A
 * B        [in] -> pointer which allocated memory for vector 
 *
 * @return  B as pointer to double. 
 *          The caller owns the pointer and have responsibility to
 *          delete it.
 * ************************************************************ */ 

double* read_vector_nblkncoll_individual(char* filename, int n, double* B)
{
    MPI_Offset fsize;
    MPI_Status status;
    MPI_Request request;
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
    err = MPI_File_open( MPI_COMM_WORLD /*MPI_COMM_SELF*/, filename, 
                         MPI_MODE_RDONLY, MPI_INFO_NULL, &fhB );
    
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
    // read operation : non-blocking, non-collective, 
    //                  individual file pointer
    // ======================================================

    err = MPI_File_iread( fhB, B, n, MPI_DOUBLE, &request );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from reading file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }
    MPI_Wait(&request, &status);

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhB );

    return B;
}


/* **************************************************************
 * @func:    read_row_matrix_blkcol_shared
 * @desc:    Open and read matrix A by row as 1-D using block 
 *           collective operation with shared file pointer
 *           
 * @note:   task (a)
 * @params:
 * ========
 * filename [in] -> matrix file A 
 * procs    [in] -> number of processes
 * n        [out] -> size from square matrix A, n x n
 *
 * @return   pointer to double (1d vector) from  matrix file.
 *           The calling function has the responsibility to 
 *           free the memory!
 * ************************************************************ */ 

double* read_row_matrix_blkcol_shared(char* filename, int procs, int* n )
{
    int        err;
    int        i,j;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fhA; 
    char       buf[100];
    int        len;
    int        eclass;
    double*    A;
    int        _n;

    // ====================================================== 
    //  open file to read
    // ======================================================

    err = MPI_File_open( MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhA );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    // view to read block by rows for each process
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

    A = malloc( sizeof(double) * _n * _n/procs);
    err = MPI_File_read_ordered( fhA, A, _n*_n/procs, MPI_DOUBLE, &status );

    *n  = _n;

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhA );
    return A;
}


/* **************************************************************
 * @func:   read_vector_splitcol_shared
 * @desc:   Open and read vector from file using split collective
 *          with share file pointer
 *
 * @note:   task (b)
 * @params:
 * ========
 * filename [in] -> vector file B
 * n        [in] -> int from matrix A nxn
 * rank     [in] -> rank of the calling process
 * procs    [in] -> number of processes
 * B        [in] -> pointer which allocated memory size n
 *
 * @return  B as pointer to double. 
 *          The caller owns the pointer and have responsibility to
 *          delete it.
 * ************************************************************ */ 

double* read_vector_splitcol_shared(char* filename, int n, int rank, int procs, double* B)
{
    MPI_Offset fsize;
    MPI_Status status;
    MPI_Request request;
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
    err = MPI_File_open( MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhB );
    
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

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
    //  set column view for reading vector by part from each
    //  process
    // ======================================================
    
    MPI_Datatype  newtype;
    MPI_Type_contiguous( _n/procs, MPI_DOUBLE, &newtype );
    MPI_Type_commit( &newtype );
    //MPI_Offset disp = rank*sizeof(double)*_n/procs;
    //printf("R[%d] disp of vector =%d\n",rank, (int)disp);
    err = MPI_File_set_view( fhB, 0, MPI_DOUBLE, newtype, 
                             "native", MPI_INFO_NULL);

    // ======================================================
    // read operation using split collective with shared file ptr
    // ======================================================

    err = MPI_File_read_ordered_begin( fhB, B, _n/procs,  MPI_DOUBLE );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from reading file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }
    MPI_File_read_ordered_end( fhB, B, &status);

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhB );

    return B;
}


/* **************************************************************
 * @func:    read_col_matrix_splitcol_individual
 * @desc:    Open and read matrix A by row as 1-D using split 
 *           collective operation with individual file pointer
 *         
 * @note:   task (b)  
 * @params:
 * ========
 * filename [in] -> matrix file A 
 * procs    [in] -> number of processes
 * rank     [in] -> rank of the calling process
 * n        [out] -> int from matrix A nxn
 *
 * @return   pointer to double (1d vector) from  matrix file.
 *           The calling function has the responsibility to 
 *           free the memory!
 * ************************************************************ */ 

double*  read_col_matrix_splitcol_individual(char* filename, int procs, int rank, int* n )
{
    int        err;
    int        i,j;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fhA; 
    char       buf[100];
    int        len;
    int        eclass;
    double*    A;
    int        _n;

    // ====================================================== 
    // file to read
    // ======================================================

    err = MPI_File_open( /*MPI_COMM_WORLD*/ MPI_COMM_SELF, filename, 
                         MPI_MODE_RDONLY, MPI_INFO_NULL, &fhA );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

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
    //  set column vector view for reading by column
    // ======================================================
    
    MPI_Datatype  newtype;
    MPI_Type_vector(_n, _n/procs, _n, MPI_DOUBLE, &newtype);
    MPI_Type_commit(&newtype);
    MPI_Offset disp = rank*sizeof(double)*_n/procs;
    err = MPI_File_set_view( fhA, disp, MPI_DOUBLE, newtype, 
                             "native", MPI_INFO_NULL);


    // ======================================================
    // read operation is much slower than memcpy
    // we try to read it once with _A_tmp and split the data to A later
    //
    // split collective op, individual fptr
    // begin calls : collective over the group of processes
    // end calls   : collective over the group of processes
    // ======================================================

    A = malloc( sizeof(double) * _n * _n/procs);
    
    err = MPI_File_read_all_begin( fhA, A, _n*_n/procs,  MPI_DOUBLE );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from reading file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    MPI_File_read_all_end( fhA, A, &status);

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhA );

    *n  = _n;
    return A;
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
 * - Solve Ax=b using Jacobi method
 * - Write answer to file
 *	- Print out information
 *
 * ************************************************************ */ 
int main( int argc, char* argv[] ) 
{ 

    int      rank; 			      // rank of the process
    int      nprocs;              // number of processes
    int      i,j;                 // misc variables
    int      err;                 // error handling
    double   eps;                 // tolerant
    int      n;                   // nxn Matrix and nx1 Vector
    int      rows;                // stripped rows or stripped column
    double*  A_vec;               // matrix A in 1-D
    double*  A_blk_vec;           // block matrix A as vector (1-D)
    double*  B;                   // vector b with size nx1 
    double*  X;                   // vector x with size nx1
    char     selector;            // task selector a or b


	MPI_Init(&argc, &argv);		 		                        
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs );             
	MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

   
	// ================================================================
	//   Check for valid inputs (passing arguments)
	// ================================================================

   if ( argc != 6) {
       if ( rank == ROOT_NODE ) {
           print_usage_extent( argv[0] );
       }
       MPI_Finalize();		
       return 0;	
   }
   eps      = atof(argv[3]);
   selector = argv[5][0];

   if ( (selector != 'a') && (selector != 'b') ) {
       if ( rank == ROOT_NODE ) {
           printf("Warning: No task %c which you select!\n",selector);
           printf("Please select only a or b for your task\n");
       }
       MPI_Finalize();		
       return 0;	
   }


	// ================================================================
	//   Open matrix A, get size n and read A as a block vector
   	//   A_vec => n/nprocs x n = rows x n is allocated inside read
   	//   function.
	// ================================================================

   switch( selector )
   {
     case 'a':
       // =======================================================
       // Reading matrix file by rows using blocking collective
       // operation with shared file pointers.
       // =======================================================

       	A_vec = read_row_matrix_blkcol_shared(argv[1], nprocs, &n);  

#if DEBUG_READING       
       	for(j=0; j<(n/nprocs)*n; ++j) {
           printf("R[%d] A[%d][%f]\n", rank,  j, A_vec[j]);           
       	}
#endif
       	break;

     case 'b':
       // =======================================================
       // Reading matrix file by column using split collective 
       // operation with individual file pointers
       // =======================================================

       	A_vec = read_col_matrix_splitcol_individual( argv[1], nprocs, rank, &n);

#if DEBUG_READING       
       	for(j=0; j<(n/nprocs)*n; ++j) {
            printf("R[%d] A[%d][%f]\n", rank,  j, A_vec[j]);           
       	}
#endif
       	break;

     default:
       	break;
   }

   rows = (int)(n/nprocs);

   // ================================================================
   //  Check inputs again after knowing the size of matrix
   // ================================================================
   if ( n%nprocs != 0 || nprocs > n ) {
       free( A_vec );
       if ( rank == ROOT_NODE ) {
           printf("\nWarning: invalid inputs! n = %d, number of process = %d\n\n",
                  n,nprocs);
       } 
       MPI_Finalize();		
       return 0;	
   }

   // ================================================================
   //   Open vector file B and get B
   // ================================================================
   switch( selector )
   {
   case 'a':
       // =======================================================
       // Reading vector using non-blocking, non-collective with
       // individual file pointer.
       // Reading B with size n will be owned by each process
       // =======================================================
       B  = malloc( sizeof(double) * n);
       B  = read_vector_nblkncoll_individual(argv[2], n, B);

#if DEBUG_READING       
       for(i=0; i<n; i++) {
           printf("R[%d] b[%d][%f]\n",rank,i, B[i]);
       }
#endif
       break;

   case 'b':
       // =======================================================
       // Reading vector using split collective operation
       // with shared file pointer.
       // Reading B with size n/nprocs for each process
       // =======================================================
       B  = malloc( sizeof(double) * n/nprocs );
       B  = read_vector_splitcol_shared( argv[2], n, rank, nprocs, B);

#if DEBUG_READING       
       for(i=0; i<n/nprocs; i++) {
           printf("R[%d] b[%d][%f]\n",rank,i, B[i]);
       } 
#endif
       break;

   default:
       break;
   }


   // ==============================================================
   //  Solving Jacobi method to get X
   //  A_vec is deleleted inside jacobi func, after no used.
   //  B     is deleleted in main where it has been allocated
   //  X     is a return vector and will be deleted in main
   // ==============================================================

   switch( selector )
   {
   case 'a':
       X = jacobi(rank, rows, n, A_vec, B, eps );
       break;

   case 'b':
       X = jacobi_by_column(rank, rows, n, A_vec, B, eps);
       break;

   default:
       break;
   }
       
   // ==============================================================
   //   Write answer, X, to file 
   // ==============================================================

   if ( rank == ROOT_NODE ) {
       write_vector(argv[4], n, X);
   }
   
   // ==============================================================
   //   Print out information to stdout
   // ==============================================================
   if ( rank == ROOT_NODE ) {
       printf("====================================================\n" );
       printf("Solving a linear system with Jacobi Method\n");
       printf("=====================================================\n");
       printf("\nInput:\n");
       printf("=======\n");
       printf("Task:                     %s\n", argv[5]);
       printf("Number of processors:     %d\n", nprocs);
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
  
   free( B );
   free( X );

   MPI_Finalize();		
   return 0;	
}

////////////////////////////////END OF FILE///////////////////////////
