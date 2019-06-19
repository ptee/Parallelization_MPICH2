/* *****************************************************************************
 * Program:    imagefilter.c                                                  
 * Author:     Pattreeya Tanisaro                    
 * Description:  "Distributed Image Processing Filter (for gray image)"
 * ===========   This program will blur an image file by using parallelization.
 *               Each processor receives part of image only,
 *               
 *               result[y][x] = sum_v(sum_u(F[u][v]*source[y+v-k][x+u-k]))
 *               
 *               k=2; Filter matrix F = (2k+1)x(2k+1)  
 *	   
 * Parameters:
 * ==========                
 * Input:        argv[1] => Image file to blur as input
 *               argv[2] => m as filter strength (how many filter is applied)
 *               argv[3] => Output file after blurring
 * 
 * Output:        x  => solution of solving Ax=b 
 *                   => also as output file as specify in arg[3]
 *                   
 * *************************************************************************** */ 

#include "mpi.h" 	                // import of the MPI definitions
#include <stdio.h> 	                // import of the definitions of the C IO lib
#include <string.h>                     // import of the definitions of the string op
#include <unistd.h>	                // standard unix io library 
#include <errno.h>	                // system error numbers
#include <sys/time.h>	                // speicial system time functions for c
#include <stdlib.h>                     // avoid warning from malloc/free
#include "cimage.h"                     // common funcs for image manipulation


/* **************************************************************
 * @func: exchange_rows
 * @desc: exchange data row-wise in each partial image
 * @params:
 * ======== 
 * @note: This function is not used here but this function is similar
 *        to the exchange data using persistent communication.
 *
 * ************************************************************ */ 
void exchange_rows( uchar** A_part_in, int rank, int rows, int w, int nprocs )
{
    int      tag0 = 0;                        // sending tag     
    int      tag1 = 1;                        // receving tag
    uchar*   top_send;                        // top send buffer
    uchar*   bottom_send;                     // bottom send buffer
    uchar*   top_recv;                        // top received buffer       
    uchar*   bottom_recv;                     // bottom received buffer
    MPI_Status status;                        // send/receive status
    int      i, j;                            // loop variables

    for ( i=0; i< K; i++) {
        top_send    = &(A_part_in[i+K][K]);
        top_recv    = &(A_part_in[i][K]);
        bottom_send = &(A_part_in[i+rows][K]);
        bottom_recv = &(A_part_in[i+rows+K][K]);

        // first node P0 <-> P1
        if ( rank == 0 ) {
            MPI_Sendrecv( bottom_send, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                          bottom_recv, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                          MPI_COMM_WORLD, &status);               
        }

        // last node P(N-1) <-> P(N-2)
        else if ( rank == nprocs - 1 ) {        
            MPI_Sendrecv( top_send, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                          top_recv, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                          MPI_COMM_WORLD, &status);
        }

        // middle nodes
        else {
            // send    top    part of the matrix from current to previous rank
            // receive bottom part of the matrix from current to next rank
            
            MPI_Sendrecv( top_send, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                          top_recv, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                          MPI_COMM_WORLD, &status);
            MPI_Sendrecv( bottom_send, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                          bottom_recv, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                          MPI_COMM_WORLD, &status);
        }  

   } // end for


}

/* **************************************************************
 * @func: init_comm
 * @desc: initialize persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void init_comm( uchar** A_part_in, int rows, int w, int nprocs,
                MPI_Request *send_reqs, MPI_Request *recv_reqs )
{
    int      rank;
    int      tag0 = 0;                        // sending/receivng tag     
    uchar*   top_send;                        // top send buffer
    uchar*   bottom_send;                     // bottom send buffer
    uchar*   top_recv;                        // top received buffer       
    uchar*   bottom_recv;                     // bottom received buffer
    MPI_Status status;                        // send/receive status
    int      i, j;                            // loop variables

    if ( nprocs == 1 ) 
        return;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 
    //   ---------------------------------------------- 
    //     ex                                     ex
    //     ex ----------------------------------- ex
    //        ----------------------------------- 
    //        -----------------------------------
    //        -----------------------------------
    //        -----------------------------------
    //        -----------------------------------
    //     ex ----------------------------------- ex
    //     ex                                     ex
    //   ----------------------------------------------

    // we need the loop of K here because the data block has the size = w+K
    // for each line. we cannot copy it at once
    //
    for ( i=0; i< K; i++) {
        top_send    = &(A_part_in[i+K][K]);
        top_recv    = &(A_part_in[i][K]);
        bottom_send = &(A_part_in[i+rows][K]);
        bottom_recv = &(A_part_in[i+rows+K][K]);

        // first node P0 <-> P1
        if ( rank == 0 ) {
            MPI_Send_init (bottom_send, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                           MPI_COMM_WORLD, &recv_reqs[i]);
            MPI_Recv_init (bottom_recv, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                           MPI_COMM_WORLD, &send_reqs[i]);
        }

        // last node P(N-1) <-> P(N-2)
        else if ( rank == nprocs - 1 ) {        
            MPI_Send_init (top_send, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                           MPI_COMM_WORLD, &recv_reqs[i]);
            MPI_Recv_init (top_recv, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                           MPI_COMM_WORLD, &send_reqs[i]);
        }

        // middle nodes
        else {
            // send    top    part of the matrix from current to previous rank
            // receive bottom part of the matrix from current to next rank

            MPI_Send_init (top_send, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                           MPI_COMM_WORLD, &recv_reqs[i]);
            MPI_Recv_init (top_recv, w, MPI_UNSIGNED_CHAR, rank-1, tag0, 
                           MPI_COMM_WORLD, &send_reqs[i]);


            MPI_Send_init (bottom_send, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                           MPI_COMM_WORLD, &recv_reqs[K+i]);
            MPI_Recv_init (bottom_recv, w, MPI_UNSIGNED_CHAR, rank+1, tag0, 
                           MPI_COMM_WORLD, &send_reqs[K+i]);
        }
    } // end for
}


/* **************************************************************
 * @func: start_comm
 * @desc: start persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void start_comm( int nprocs, MPI_Request* send_reqs, MPI_Request* recv_reqs )
{
    int    rank;

    if ( nprocs == 1 )
        return;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

    // first node P0, only the bottom part will be send+recv to P1
    if ( rank == ROOT_NODE ) {
        MPI_Startall (K, send_reqs);   
        MPI_Startall (K, recv_reqs);   
    }
    
    // last node P(N-1) <-> P(N-2)
    else if ( rank == nprocs - 1 ) {        
        MPI_Startall (K, send_reqs);   
        MPI_Startall (K, recv_reqs);   
    }

    // middle nodes
    else {
        MPI_Startall (2*K, send_reqs);   
        MPI_Startall (2*K, recv_reqs);   
    }        
}


/* **************************************************************
 * @func: wait_reqs
 * @desc: wait for requests in persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void wait_reqs( int nprocs,  MPI_Request* send_reqs, MPI_Request* recv_reqs )
{
    int           rank;
    MPI_Status   send_status[MAX_NUM_REQS];
    MPI_Status   recv_status[MAX_NUM_REQS];

    if ( nprocs == 1 )
        return;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

    // for first node, P0 <-> P1 only bottom part of P0
    if ( rank == ROOT_NODE ) {
        MPI_Waitall (K, send_reqs, send_status);   
        MPI_Waitall (K, recv_reqs, recv_status);   
    }
    
    // last node P(N-1) <-> P(N-2)
    else if ( rank == nprocs - 1 ) {        
        MPI_Waitall (K, send_reqs, send_status);   
        MPI_Waitall (K, recv_reqs, recv_status);   
    }

    // middle nodes
    else {
        MPI_Waitall (2*K, send_reqs, send_status);   
        MPI_Waitall (2*K, recv_reqs, recv_status);   
    }        
}


/* **************************************************************
 * @func: free_request
 * @desc: free requests using in persistent communication
 * @params:
 * ======== 
 *
 * ************************************************************ */ 
void free_request(int nprocs,  MPI_Request* reqs )
{
    int   i;
    int   rank; 
    if ( nprocs == 1 )
        return;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

    if ( rank == ROOT_NODE ||  rank == nprocs - 1) {
        for (i=0; i< K; i++) {
            MPI_Request_free( &reqs[i] );
        }
    }
    else  {        
        for (i=0; i< 2*K; i++) {
            MPI_Request_free( &reqs[i] );
        }
    }

}

/* **************************************************************
 * @func:  main 
 * @desc:     
 * ====== 
 * - Open and read image A
 * - Split block image A to other processes using MPI_Scatter
 * - Map 1-D block to 2-D block (size + 2*K )
 * - Blur each partial image using filter F m times
 *    - send/receive edge part of the image to each process
 *    - call blurring func for each partial image
 *    - do it m times
 * - Re-compose partial images into one filnal image
 * - Write image to file
 * - Print out infos to stdout
 * - Clean up
 * ************************************************************ */ 
int main( int argc, char* argv[] ) 
{ 

    int      rank; 			      // rank of the process
    int      nprocs;                          // number of processes
    int      i,j;                             // misc variables
    int      rows;                            // stripped rows = H/nprocs
    uchar*   A_vec;                           // complete A in 1-D (reading)
    uchar*   A_part_vec;                      // part of image A in 1-D (temporary)
    uchar**  A_part_in;                       // part of image(2-D) as input to func
    uchar**  A_part_out;                      // part of image(2-D) as output from func
    int      w;                               // width of the image by calculation
    int      H;                               // height of the image given by user
    int      m;                               // number of applying filter to image
    MPI_Request send_reqs[MAX_NUM_REQS];      // requests used in persistent communication
    MPI_Request recv_reqs[MAX_NUM_REQS];      // requests used in persistent communication


    MPI_Init(&argc, &argv);		 		                        
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );             
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

   
    // ===========================================================
    //  Check valid inputs (passing arguments)
    // ===========================================================

   if ( argc != 5 ) {
       if ( rank == ROOT_NODE ) {
           print_usage( argv[0] );
       }
       MPI_Finalize();		
       return 0;	
   }
   m = atoi(argv[3]);
   if ( ! VALID_NUM_FILTERS(m) ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: Applying filter, %d times is out of range!\n",m);
       }
       MPI_Finalize();		
       return 0;	
   }
   H = atoi( argv[2] );
   if ( H%nprocs != 0 ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: The height of image is fixed with %d pixels!\n", H);
           printf("It cannot be equally divided by %d processors!", nprocs);
       }
       MPI_Finalize();		
       return 0;	
   }

   // ===========================================================
   //   Open image and get width w and read image A (1-D)
   //   from ROOT_NODE
   // ===========================================================
   if ( rank == ROOT_NODE ) {
       A_vec = open_n_read_image(argv[1], H, &w);
   }

   MPI_Bcast( &w, 1, MPI_INT, ROOT_NODE, MPI_COMM_WORLD );
   rows = (int)(H/nprocs);
   #if DEBUG
   printf("m= %d wxh= (%d,%d) -> (rowsxw) = %dx%d and K = %d\n",
          m, w, H,  rows, w, K);
   #endif
   // ===========================================================
   //   Scatter 1-d array from root-node to othernode as part
   // of image.  This step, we need to map data from MPI_Scatter.
   // ===========================================================

   A_part_vec = malloc( sizeof(uchar) * (rows) * (w) );
   MPI_Scatter ( A_vec,      rows * w, MPI_UNSIGNED_CHAR, 
                 A_part_vec, rows * w, MPI_UNSIGNED_CHAR, ROOT_NODE, MPI_COMM_WORLD ); 

   // ===========================================================
   // Each node map  1-d array to 2-d with plus size + 2*K array (part image)
   // Map each line of A_part_vec (rowsxw) to A_part_in (rows+2K)x(w+2K)
   // We leave the eage of partial image of K to zero
   //    ---------------------------------
   //    | K ------------------------  K |
   //    |   |XXXXXXXXXXXXXXXXXXXXXXX|   |
   //    | K |XXXXXXXXXXXXXXXXXXXXXXX| K |
   //    |_______________________________|
   //    
   // ===========================================================
   
   
   A_part_in = malloc( sizeof(uchar *) * (rows+2*K) );
   for( i=0; i< (rows+2*K); i++ ) {
       A_part_in[i] = malloc( sizeof(uchar) * (w+2*K) );
       memset( A_part_in[i], 0, (w+2*K)*sizeof(uchar));
       if ( i >= K && i < rows+K ) {
           memcpy( &(A_part_in[i][K]), &A_part_vec[(i-K)*w], w*sizeof(uchar) );
       }
   }
   //print_matrix(rank, A_part_in, rows+(2*K), w+(2*K), 0, rows+K, 0, w+(2*K));
    
      
   // ===========================================================
   //  Init communication persistent 
   // ===========================================================

   init_comm( A_part_in, rows, w, nprocs, send_reqs, recv_reqs );

   
   // =====================================================================
   //  Main filtering/blurring image "m" times
   // =====================================================================
  
   while (m > 0 ) {
       
       // =======================================================
       //  Exchange the data top <-> bottom of each block or partial image
       //      A_part_in with persistent communication
       //
       //  The exchange will be repeated for each blurring,
       //   Number of requests = per requests
       // =======================================================
           
       // start persistent communication
       start_comm( nprocs, send_reqs, recv_reqs );      
       
       // can do some other works here...
           
       // wait for requests finish
       wait_reqs( nprocs,  send_reqs,  recv_reqs ); 
      
       // =======================================================
       //  Blurring image m times
       // =======================================================

       A_part_out = blurring( A_part_in, rows+2*K, w+2*K );
       --m;

       // If there is still next iteration , we copy output image to be next input
       // using old memory and free unused memory.
       // Every call of "blurring" will allocate new memory for
       // output image (A_part_out)
       if ( m > 0 ) {
           for( i=0; i<rows+2*K; i++ ) {
               memcpy( &(A_part_in[i][0]), &(A_part_out[i][0]), (w+2*K)*sizeof(uchar) );
               free( A_part_out[i] );
           }
           free( A_part_out );
       }

   }

   
   // =====================================================================
   //   Re-compose the partial images into one final image with
   //  original size, reuse A_vec which we already have previously.
   // =====================================================================
   
   for( j=K; j<rows+K; j++ ) {
       memcpy( &(A_part_vec[(j-K)*w]), &(A_part_out[j][K]), w*sizeof(uchar) );
   }
   
   MPI_Gather( A_part_vec, rows*w, MPI_UNSIGNED_CHAR, 
               A_vec,      rows*w, MPI_UNSIGNED_CHAR, ROOT_NODE, MPI_COMM_WORLD );


   // ======================================================
   //  Write image as 1-D vector to output file
   // ======================================================
   if ( rank == ROOT_NODE ) {
       write_image( argv[4], A_vec, w, H );
   }

   // ======================================================
   // write information to screen
   // ======================================================
   if ( rank == ROOT_NODE ) {
       printf("====================================================\n" );
       printf("Distributed Image Processing: Blurring Image\n");
       printf("=====================================================\n");
       printf("Image file (i/p):                %s\n", argv[1]);
       printf("Image size:                [%d x %d]\n",w, H);
       printf("No. of processes:                %d\n",nprocs);
       printf("No. of applying filter:          %d\n", atoi(argv[3]));
       printf("========\n");
       printf("Blured image (o/p):              %s\n", argv[4]);
       printf("=====================================================\n");
   }

   // ======================================================
   //  Clean up
   // ======================================================
   free_request( nprocs, send_reqs );
   free_request( nprocs, recv_reqs );


   for( i=0; i<rows+2*K; i++ ) {
       free( A_part_in[i] );
       free( A_part_out[i] );
   }
   free( A_part_in );
   free( A_part_out );
    
   if ( rank == ROOT_NODE ) {
       free( A_vec );
   }
   

   MPI_Finalize();		
   return 0;	

}

//////////////////////////////////END OF FILE///////////////////////////
