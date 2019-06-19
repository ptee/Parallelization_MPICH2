/* *****************************************************************************
 * Program:    cannonmmul.c                                                  
 * Author:     Pattreeya Tanisaro                   
 * Description:  Matrix multiplication AxB = C using Cannon's algorithm
 * ===========   
 *              a) using MPI_Sendrecv_replace
 *              b) using persistent communication
 *	   
 * Parameters:
 * ==========                
 * Input:        argv[1] => [in]  Matrix file A
 *               argv[2] => [in]  Matrix file B
 *               argv[3] => [out] Matrix file C
 *               argv[4] => [in]  Task option whether a or b
 *                   
 * *************************************************************************** */ 

#include "mpi.h" 	                // import of the MPI definitions
#include <stdio.h> 	                // import of the definitions of the C IO lib
#include <string.h>                     // import of the definitions of the string op
#include <unistd.h>	                // standard unix io library 
#include <errno.h>	                // system error numbers
#include <sys/time.h>	                // speicial system time functions for c
#include <stdlib.h>                     // avoid warning from malloc/free
#include "iocommon.h"                   // common funcs for image manipulation


#define VALID_FUNC_SELECTOR(sel) ((sel=='a')||(sel=='b'))? true: false

/* **************************************************************
 * @func: print_usage
 * @desc: print out how to use this program
 * @params:
 * ======== 
 *	prog [in] => char* as execultable
 *
 * ************************************************************ */ 

void print_matrixmultiply_usage(char *prog)
{
	 printf("\nMatrix multiplication A*B = C using Cannon algorithm\n");
	 printf("------------------------------------------------------\n");
	 printf("Usage:     %s   [Matrix A]  [Matrix B]   [Matrix C]  \n",prog );
    	 printf("Comm_Method: 'a' is to use MPI_Sendrecv_replace\n");
    	 printf("Comm_Method: 'b' is to use MPI persistent communication\n");
	 printf("Exmaples:  %s    A_16x16     B_16x16     C_16x16.out  \n\n\n",prog);
}


/* **************************************************************
 * @func:  multiply_matrix
 * @desc:  matrix multiplication C=A*B (for square matrix) 
 * @params:
 * ========
 * n [in]  = matrix size 
 * A [in]  = matrix A with size nxn
 * C [out] = A*B
 *         
 * ************************************************************ */ 

void multiply_matrix( int n, double* A, double* B, double* C )
{
    int i,j,k;

    for ( i=0; i < n; i++ ) 
    {
        for ( j=0; j < n; j++ ) 
        {
            for ( k=0; k < n; k++ )
            {
                C[i*n+j] += A[i*n+k]*B[k*n+j]; // c = a*b
            }
        }
    }

}


/* **************************************************************
 * @func:  calc_displ
 * @desc:  calculate displacemnet of each process
 * @params:
 * ========
 *
 * @return displacement of the calling processes as integer
 *         
 * ************************************************************ */ 

int calc_displ( MPI_Comm g_comm, int n, int nlocal )
{
    int   rank;
    int   displ;
    int   coords[DIMENSIONS];

    MPI_Comm_rank( g_comm, &rank); 
    MPI_Cart_coords( g_comm, rank, DIMENSIONS, coords);
    displ = (n*nlocal*coords[0]) + (nlocal*coords[1]);
    //printf("R[%d] displ = %d\n", rank, displ);

    return displ;
}


/* **************************************************************
 * @func:  skew_A
 * @desc:  intitailization matrix A for Cannon's algorithm
 * @params:
 * ========
 *         
 * ************************************************************ */ 

void skew_A(double* A, int* coords, MPI_Comm g_comm, int nlocal )
{
    int   coords_src[DIMENSIONS];          // coordinate of the source rank
    int   coords_dest[DIMENSIONS];         // coordinate of the destination rank
    int   tag = 0;
    MPI_Status status;
    int   dest_rank, src_rank;

    coords_dest[0] = coords[0];
    coords_dest[1] = coords[1] - coords[0];
    MPI_Cart_rank( g_comm, coords_dest, &dest_rank);
    coords_src[0] = coords[0];
    coords_src[1] = coords[1] + coords[0];
    MPI_Cart_rank( g_comm, coords_src, &src_rank);

    /*
      printf("A: R[%d] coords(%d,%d) From R[%d] Co-src(%d,%d) To R[%d] Co-dest(%d,%d)\n",
             g_rank, coords[0], coords[1], 
             src_rank, coords_src[0], coords_src[1],
             dest_rank, coords_dest[0], coords_dest[1] );
    */

    MPI_Sendrecv_replace( A, nlocal*nlocal, MPI_DOUBLE, dest_rank, tag, 
                          src_rank, tag, g_comm, &status);

}


/* **************************************************************
 * @func:  skew_B
 * @desc:  intitailization matrix B for Cannon's algorithm
 * @params:
 * ========
 *         
 * ************************************************************ */ 

void skew_B(double* B, int* coords, MPI_Comm g_comm, int nlocal )
{
    int   coords_src[DIMENSIONS];          // coordinate of the source rank
    int   coords_dest[DIMENSIONS];         // coordinate of the destination rank
    int   tag = 0;
    MPI_Status status;
    int   dest_rank, src_rank;


    coords_dest[0] = coords[0] - coords[1];
    coords_dest[1] = coords[1];
    MPI_Cart_rank( g_comm, coords_dest, &dest_rank);
    coords_src[0] = coords[0] + coords[1];
    coords_src[1] = coords[1];
    MPI_Cart_rank( g_comm, coords_src, &src_rank);

    /*
       printf("B: R[%d] coords(%d,%d) From R[%d] Co-src(%d,%d) To R[%d] Co-dest(%d,%d)\n",
              g_rank, coords[0], coords[1], 
              src_rank, coords_src[0], coords_src[1],
              dest_rank, coords_dest[0], coords_dest[1] );
    */

    MPI_Sendrecv_replace( B, nlocal*nlocal, MPI_DOUBLE, dest_rank, tag, 
                          src_rank, tag, g_comm, &status);

}


/* **************************************************************
 * @func:  sendrecv_xchange
 * @desc:  Using MPI_Sendrecv_replace to exchange data of A between 
 *         neighbour ranks
 * @params:
 * ========
 *         
 * ************************************************************ */ 

void sendrecv_xchange( double* A, double* B, MPI_Comm g_comm, int nlocal )
{
    int dest_rank, src_rank;
    int tag = 0;
    MPI_Status status;

    // shift matrix A to the left by one
    MPI_Cart_shift( g_comm, 1, 1,  &dest_rank, &src_rank);

    //printf(" R[%d] left_R[%d] right_R[%d]\n", g_rank, dest_rank, src_rank );

    MPI_Sendrecv_replace( A, nlocal*nlocal, MPI_DOUBLE, dest_rank, tag, 
                          src_rank, tag, g_comm, &status);
       
    // shfit matrix B up by one
    MPI_Cart_shift( g_comm, 0, 1,  &dest_rank, &src_rank);

    //printf(" R[%d] up_R[%d] down_R[%d]\n", g_rank, dest_rank, src_rank );

    MPI_Sendrecv_replace( B, nlocal*nlocal, MPI_DOUBLE, dest_rank, tag, 
                          src_rank, tag, g_comm, &status);
}

/* **************************************************************
 * @func: init_comm_horizontal_xchange
 * @desc: initialize persistent communication of horizontal neighbours
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void init_comm_horizontal_xchange( double* A, int nlocal,  MPI_Comm g_comm,
                                   MPI_Request *reqs )
{
    int      dest_rank, src_rank, rank;       // destination, source and this rank 
    int      tag0 = 0;                        // sending/receivng tag     
    MPI_Status status;                        // send/receive status
    int      i, j;                            // loop variables


    //MPI_Comm_rank( g_comm, &rank); 

    // shift left by 1
    MPI_Cart_shift( g_comm, 1, 1,  &dest_rank, &src_rank);
    //printf(" R[%d] left_R[%d] right_R[%d]\n", rank, dest_rank, src_rank );

    MPI_Send_init (A, nlocal*nlocal, MPI_DOUBLE, dest_rank, tag0, 
                   g_comm, &reqs[0]);
    MPI_Recv_init (A, nlocal*nlocal, MPI_DOUBLE, src_rank, tag0, 
                   g_comm, &reqs[1]);
 
}


/* **************************************************************
 * @func:  init_comm_vertical_xchange
 * @desc:  initialize persistent communication of vertical neighbours
 * @params:
 * ========
 *         
 * ************************************************************ */ 

void init_comm_vertical_xchange( double* B, int nlocal, MPI_Comm g_comm,
                                 MPI_Request *reqs )
{
    int      dest_rank, src_rank, rank;       // destination, source and this rank 
    int      tag0 = 0;                        // sending/receivng tag     
    MPI_Status status;                        // send/receive status
    int      i, j;                            // loop variables
    
    //MPI_Comm_rank( g_comm, &rank); 

    // shfit matrix B up by one
    MPI_Cart_shift( g_comm, 0, 1,  &dest_rank, &src_rank);
    //printf(" R[%d] up_R[%d] down_R[%d]\n", rank, dest_rank, src_rank );

    MPI_Send_init (B, nlocal*nlocal, MPI_DOUBLE, dest_rank, tag0, 
                   g_comm, &reqs[2]);
    MPI_Recv_init (B, nlocal*nlocal, MPI_DOUBLE, src_rank, tag0, 
                   g_comm, &reqs[3]);

}


/* **************************************************************
 * @func: start_comm
 * @desc: start persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void start_comm( MPI_Request* reqs )
{
 
    // start for both A and B
    MPI_Startall (4, reqs);   

}


/* **************************************************************
 * @func: wait_reqs
 * @desc: wait for requests in persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void wait_reqs( MPI_Request* reqs )
{

    MPI_Status   status[MAX_NUM_REQS];

    MPI_Waitall (4, reqs, status);   

}


/* **************************************************************
 * @func: free_reqs
 * @desc: free requests in persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void free_reqs( MPI_Request* reqs )
{
    int i;

    for (i=0; i< 4; i++) {
        MPI_Request_free( &reqs[i] );
    }
}

/* **************************************************************
 * @func:  read_sync
 * @desc:  ready data (double) from file synchronously
 * @params:
 * ========
 *
 * @return pointer to the data, (here refers to a matrix), the caller
 *         must take responsibility to free the memory
 * 
 * ************************************************************ */ 

double* read_sync( char* filename, MPI_Comm g_comm, int nlocal, int n )
{
    int        err;
    int        i,j;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fh; 
    char       buf[100];
    int        len;
    int        eclass;
    double*    A;
    int        displ;
    MPI_Datatype  vectype;


    displ = calc_displ( g_comm, n, nlocal );


    // ====================================================== 
    // Open file 
    // ======================================================   
    err = MPI_File_open(  MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    A = malloc( nlocal*nlocal* sizeof(double) );
    memset(A, 0x0, nlocal*nlocal*sizeof(double));
    
    // ======================================================
    //  set view for reading vector by part from each process
    // ======================================================
    
    MPI_Type_vector( nlocal, nlocal, n, MPI_DOUBLE, &vectype );
    MPI_Type_commit( &vectype );
    MPI_Offset offset = sizeof(double)*displ;
    err = MPI_File_set_view( fh, offset, MPI_DOUBLE, vectype, 
                             "native", MPI_INFO_NULL);

    // ======================================================
    // read operation from predefined set_view
    // ======================================================
    MPI_File_read( fh, A, nlocal*nlocal, MPI_DOUBLE, &status);

    //print_vector( rank, A, nlocal*nlocal, 
    //              0, nlocal*nlocal, DOUBLE_TYPE );

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fh );
    return A;
}


/* **************************************************************
 * @func:  write_sync
 * @desc:  write partial data (1-D vector) from all processes to file synchronously
 * @params:
 * ========
 *
 * @return none, memory occupied by A pointer is free inside!
 *         
 * ************************************************************ */ 

void write_sync( char* filename, double* A, int nlocal, int n, int g_comm )
{
    int        err;
    MPI_File   fh;
    MPI_Status status;
    MPI_Datatype  vectype;
    char       buf[100];
    int        len;
    int        eclass;
    int        i,j;
    int        g_rank;
    int        displ;
    
    displ = calc_displ( g_comm, n, nlocal );

    err = MPI_File_open( g_comm , filename, 
                         MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh );
    if (err) {
        MPI_Error_string(err, buf, &len);
        printf("Error from open file to write %d: %s\n", eclass, buf);
        MPI_Abort( g_comm, 1); 
    }

    MPI_File_set_atomicity( fh, true );
    
    
    // ======================================================
    //  set view for writing vector by part from each process
    // ======================================================
    
    MPI_Type_vector( nlocal, nlocal, n, MPI_DOUBLE, &vectype );
    MPI_Type_commit( &vectype );
    MPI_Offset offset = sizeof(double)*displ;
    err = MPI_File_set_view( fh, offset, MPI_DOUBLE, vectype, 
                             "native", MPI_INFO_NULL);

    // ======================================================
    // read operation from predefined set_view
    // ======================================================
    MPI_File_write( fh, A, nlocal*nlocal, MPI_DOUBLE, &status);

    MPI_File_sync( fh ); // force flush
    MPI_Barrier( g_comm );
    
    // =====================================================
    //  Clean up
    // =====================================================
    MPI_File_close( &fh );   
    
    free( A );

}


/* **************************************************************
 * @func:  main 
 * @desc:     
 * ====== 
 * - Create grid topology according to number of given processes 
 * - Open and read matrix A and B synchronously from files 
 * - Initialize matrix A and B from Cannon's algorithm
 * - Muliply matrices and shift data in matrices which required
 *   for Cannon's algorithm
 * - Write C=A*B from all processes to file synchronously
 * - Print out necessary information to stdout
 * - Clean up 
 *
 * ************************************************************ */ 
int main( int argc, char* argv[] ) 
{ 

    int      rank; 			      // rank of the process
    int      g_rank;                          // grid rank
    int      nprocs;                          // number of processes
    int      i,j;                             // misc variables
    double*  A;                               // partial matrix A from reading
    double*  B;                               // partial matrix B from reading
    double*  C;                               // partial matrix C from A*B
    int      n;                               // size of matrix n x n
    int      nlocal;                          // partial matrix size = n/dims[0]
    int      sp;                              // sp = sqrt(nprocs)
    int      g_wrap[2] = {WRAP, WRAP};        // grid topology is wrapped or periodic
    int      dims[DIMENSIONS] = {0,0};        // grid dimension
    int      coords[DIMENSIONS];              // coordinate of this rank
    MPI_Comm g_comm;                          // grid communication 
    ulong    total_seq = 0;                   // time used in sequential portion
    ulong    total_comm = 0;                  // time used in communication
    ulong    total_par = 0;                   // time used in parallel algorithm
    struct timeval start_time, stop_time;     // time struct to start and stop timer
    char     func_c;                          // function selector
    MPI_Request reqs[MAX_NUM_REQS];           // requests used in persistent communication
    MPI_Status status;                        // send/receive status
    

    MPI_Init(&argc, &argv);		 		                        
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );             
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

   // ===========================================================
   //  Check valid inputs (passing arguments)
   // ===========================================================
   if ( argc != 5 ) {
       if ( rank == ROOT_NODE ) {
           print_matrixmultiply_usage( argv[0] );
       }
       MPI_Finalize();		
       return 0;	
   }
   sp = (int)floor(sqrt(nprocs));
   if( sp*sp != nprocs ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: Number of processes %d is not square!",nprocs);
       }
       MPI_Finalize();		
       return 0;	
   }
   func_c = argv[4][0];
   if ( ! VALID_FUNC_SELECTOR( func_c) ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: Option %c does not exist!",func_c);
           print_matrixmultiply_usage( argv[0] );
       }
       MPI_Finalize();		
       return 0;	
   }

   // ===========================================================
   //  Compute size of grids and create its topology
   // ===========================================================

   dims[0] = dims[1] = (int)sqrt(nprocs);
   MPI_Cart_create( MPI_COMM_WORLD, DIMENSIONS, dims, g_wrap, true, &g_comm );
   MPI_Comm_rank( g_comm, &g_rank); 
   MPI_Cart_coords( g_comm, g_rank, DIMENSIONS, coords);
   //printf("R[%d] sp = %d dims = %d x %d \n", g_rank, sp, dims[0], dims[1]);

   // start timer
   MPI_Barrier( g_comm );
   gettimeofday( &start_time, NULL );

   n = get_size( argv[1] );
   if ( n != get_size ( argv[2] ) ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: matrix A (%d) and B(%d) are not equal size!",n, get_size ( argv[2] ));
       }
       MPI_Finalize();		
       return 0;	
   }
   n = (int)sqrt( (float)((int)n/sizeof(double)) );
   if ( n%sp != 0 ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: matrix with size %d not divisible by sqrt of nprocs %d!",n,sp );
       }
       MPI_Finalize();		
       return 0;	
   }
   nlocal = n/dims[0];
   //printf("n=%d, nlocal = %d\n",n, nlocal);

   // stop timer
   MPI_Barrier( g_comm );
   gettimeofday( &stop_time, NULL );
   total_seq += timeval2msec(stop_time) - timeval2msec(start_time);

   // ===========================================================
   //   Open and read square matrix A and B
   // ===========================================================

   // start timer
   MPI_Barrier( g_comm );
   gettimeofday( &start_time, NULL );

   A = read_sync( argv[1], g_comm, nlocal, n );
   B = read_sync( argv[2], g_comm, nlocal, n );
   C = malloc( nlocal*nlocal*sizeof(double) );
   memset( C, 0, nlocal*nlocal*sizeof(double) );

   // stop timer
   MPI_Barrier( g_comm );
   gettimeofday( &stop_time, NULL );
   total_par += timeval2msec(stop_time) - timeval2msec(start_time);

   // ===========================================================
   // Init Cannon's alorithm, skew A and B
   // ===========================================================
   // start timer
   // *** approximation of communication time****
   MPI_Barrier( g_comm );
   gettimeofday( &start_time, NULL );

   // skew A, shift left  
   if ( coords[0] != 0 && (nprocs > 1) )
   {
       skew_A( A, coords, g_comm, nlocal );

   }

   // skew B, shift up
   if ( coords[1] != 0 && (nprocs > 1) )
   {

       skew_B( B, coords, g_comm, nlocal );

   }

   // stop timer
   MPI_Barrier( g_comm );
   gettimeofday( &stop_time, NULL );
   total_comm += timeval2msec(stop_time) - timeval2msec(start_time);				

   
   // ===========================================================
   // Main matrix multiplication of Cannon's algorithm
   // ===========================================================

   // init persistent communication 
   if ( (func_c == 'b')  &&  (nprocs > 1) ) {
       // start timer
        MPI_Barrier( g_comm );
        gettimeofday( &start_time, NULL ); 
       
       init_comm_horizontal_xchange( A, nlocal, g_comm, reqs );
       init_comm_vertical_xchange( B,  nlocal, g_comm, reqs );
       
       // stop timer
       MPI_Barrier( g_comm );
       gettimeofday( &stop_time, NULL );
       total_comm += timeval2msec(stop_time) - timeval2msec(start_time);  
   }

   // main computation
   for ( i = 0; i < sp; i++ )
   {
       
       // start timer
       MPI_Barrier( g_comm );
       gettimeofday( &start_time, NULL );

       multiply_matrix( nlocal, A, B, C );
       
       // stop timer
       MPI_Barrier( g_comm );
       gettimeofday( &stop_time, NULL );
       total_par += timeval2msec(stop_time) - timeval2msec(start_time);				

       if( nprocs == 1 ) break;

       // start timer 
       MPI_Barrier( g_comm );
       gettimeofday( &start_time, NULL );

       if ( func_c == 'a' ) {

           sendrecv_xchange( A, B, g_comm, nlocal );

       }
       else {
           // start persistent communication
           start_comm( reqs );         
           
           // can do some other works here...
           
           // wait for requests finish
           wait_reqs( reqs ); 
           
       }

       // stop timer
       MPI_Barrier( g_comm );
       gettimeofday( &stop_time, NULL );
       total_comm += timeval2msec(stop_time) - timeval2msec(start_time);				
      
   }

   // ===========================================================
   //  write output matrix C to file and free C 
   // ===========================================================

   // start timer
   MPI_Barrier( g_comm );
   gettimeofday( &start_time, NULL );

   write_sync( argv[3], C, nlocal, n, g_comm );

   // stop timer
   MPI_Barrier( g_comm );
   gettimeofday( &stop_time, NULL );
   total_par += timeval2msec(stop_time) - timeval2msec(start_time);				

   // ======================================================
   // write information to screen
   // ======================================================
   if ( rank == ROOT_NODE ) {
       printf("====================================================\n" );
       printf("Matrix multiplcation using Cannon's Algorithm C=A*B \n");
       printf("=====================================================\n");
       printf("A:                %s\n", argv[1]);
       printf("B:                %s\n", argv[2]);
       printf("C:                %s\n", argv[3]);
       printf("Select comm:               %c\n", func_c);
       printf("Matrix size:                [%d x %d]\n",n, n);
       printf("No. of processes:                %d\n", nprocs);
       printf("total sequential part:     %ld msec\n", total_seq);
       printf("total parallel part:       %ld msec\n", total_par);
       printf("total communication:       %ld msec\n", total_comm);
       printf("=====================================================\n");
   }


   // ======================================================
   // clean up
   // ======================================================
   if ( func_c == 'b' && nprocs > 1 ) {
       free_reqs( reqs );
   }

   MPI_Comm_free( &g_comm );
   free ( A );
   free ( B );

   MPI_Finalize();		
   return 0;	

}

//////////////////////////////////END OF FILE///////////////////////////
