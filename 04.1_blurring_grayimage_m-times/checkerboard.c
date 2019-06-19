/* *****************************************************************************
 * Program:    checkerboard.c                                                  
 * Author:     Pattreeya Tanisaro                    
 * Description:  "Distributed Image Processing Filter (for gray image) base on
 * ===========    grid topology"
 * 	This program will blur an image file by using parallelization.
 *              Each processor receives part of image only,
 *               
 *              result[y][x] = sum_v(sum_u(F[u][v]*source[y+v-k][x+u-k]))
 *               
 *              k=2; Filter matrix F = (2k+1)x(2k+1)  
 *	   
 * Parameters:
 * ==========                
 * Input: argv[1] => Image file
 *        argv[2] => m as filter strength
 *        argv[3] => Output file after filtering
 * 
 * Output:     x  => solution of solving Ax=b 
 *                => also as output file as specify in arg[3]
 *                   
 * *************************************************************************** */ 

#include "mpi.h" 	                   // import of the MPI definitions
#include <stdio.h> 	                   // import of the definitions of the C IO lib
#include <string.h>                    // import of the definitions of the string op
#include <unistd.h>	                   // standard unix io library 
#include <errno.h>	                   // system error numbers
#include <sys/time.h>	               // speicial system time functions for c
#include <stdlib.h>                    // avoid warning from malloc/free
#include "cimage.h"                    // common funcs for image manipulation

#define DEBUG_COMM     0
#define DEBUG          0

/* **************************************************************
 * @func: valid_coords
 * @desc: check if the given coordinates are in range of the dimension
 * @params:
 * ======== 
 * @return true if the given coordinates are in range of the dimension
 *         false otherwise
 *
 * ************************************************************ */ 
bool valid_coords( int* test_coords, int* dims )
{
    if ( test_coords[0] < 0 || test_coords[0] >= dims[0] )
        return false;

    if ( test_coords[1] < 0 || test_coords[1] >= dims[1] )
        return false;

    return true;
}


/* **************************************************************
 * @func:  exchange_grid_data
 * @desc:  exchange the border of the (part) images in grid topology
 *         The exchange composes of 3 main parts:
 *         - left-right
 *         - up-down
 *         - corner (This part is trivial because K is very small (2 pixels)
 *           Therefore, the data size to be exchange is only 2x2 pixels.
 * @params:
 * ======== 
 *
 * ************************************************************ */ 

void exchange_grid_data( int g_rank, int g_comm, uchar** A_part_in, int w, int h, int* dims )
{
    MPI_Status status;                        // send/receive status
    int      tag0 = 0;                        // sending/receving tag     
    int      i, j;                            // loop variables
    int      prev_rank;                       // source rank from cart_shift
    int      next_rank;                       // destination rank from cart_shift
    int      up    = INVALID_RANK;            // direction up            
    int      dn    = INVALID_RANK;            // direction down
    int      left  = INVALID_RANK;            // direction left (displcement)
    int      right = INVALID_RANK;            // direction right (displacement)
    int      coord_old[DIMENSIONS];           // to store coordinates in finding corner
    int      coord_new[DIMENSIONS];           // to store coordinates in finding corner
    int      coord_new2[DIMENSIONS];          // to store coordinates in finding corner  

    // @note: About MPI_Sendrecv
    // The concept of using MPI_Sendrecv is to identify itself (rank)
    // who i am and define the other environments base on "whoami".
    // Otherwise, we will face the problem of deadlock.

    // =====================================================
    // exchange data between left-right neighbor processors
    // =====================================================
    MPI_Cart_shift( g_comm, 1, 1, &prev_rank, &next_rank );
    //printf("R[%d] dir(1,1) prev[%d] - next[%d]\n",g_rank, prev_rank, next_rank);
 
    // left-rim rank
    if ( prev_rank == INVALID_RANK ) {
        for ( i=0; i< h; i++) 
        {
            MPI_Sendrecv( &A_part_in[K+i][w],   K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[K+i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
        }
        right = next_rank;
    }
    // right-rim rank
    else if ( next_rank == INVALID_RANK ) {
        for ( i=0; i< h; i++) 
        {
            MPI_Sendrecv( &A_part_in[K+i][K], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[K+i][0],   K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
        }
        left = prev_rank; 
    }
    // middle: left <-> right
    else {
        for ( i=0; i< h; i++) 
        {
            MPI_Sendrecv( &A_part_in[K+i][w],   K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[K+i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);

            MPI_Sendrecv( &A_part_in[K+i][K], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[K+i][0],   K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
        }
        left = prev_rank; right = next_rank;
    }
    
    // =====================================================
    // exchange data between up-down neighbor processors
    // =====================================================
    MPI_Cart_shift( g_comm, 0, 1, &prev_rank, &next_rank );
    //printf("R[%d] dir(0,1) prev[%d] - next[%d]\n",g_rank, prev_rank, next_rank);

    // up-rim rank
    if ( prev_rank == INVALID_RANK ) {
        for ( i=0; i< K; i++) 
        {
            MPI_Sendrecv( &A_part_in[h+i][K],   w, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][K], w, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
        }
        dn = next_rank;
    }
    // down-rim rank
    else if ( next_rank == INVALID_RANK ) {
        for ( i=0; i< K; i++) 
        {
            MPI_Sendrecv( &A_part_in[K+i][K], w, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[i][K],   w, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
        }
        up = prev_rank;
    }
    // middle
    else {
        for ( i=0; i< K; i++) {
            MPI_Sendrecv( &A_part_in[K+i][K], w, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[i][K],   w, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
            
            MPI_Sendrecv( &A_part_in[h+i][K],   w, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][K], w, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
        }
        up = prev_rank; dn = next_rank;
    }

    // =====================================================
    //  exchange data between corners of the part images
    // This is a long process, however, at the end only 1 pixel
    // will be changed in convolution with filter K=2!
    // =====================================================
    
    // 1) top left corner
    if ( up == INVALID_RANK && left == INVALID_RANK && 
         dn != INVALID_RANK && right != INVALID_RANK )
    {
        MPI_Cart_coords(g_comm, dn, DIMENSIONS, coord_old);
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]+1;
        MPI_Cart_rank( g_comm, coord_new, &next_rank);
        //printf("top-left R[%d] <-> %d\n", g_rank, next_rank);

        for ( i=0; i< K; i++) 
        {
            MPI_Sendrecv( &A_part_in[h+i][w],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
        }
    }

    // 2) bottom right corner
    if ( up != INVALID_RANK && left != INVALID_RANK && 
         dn == INVALID_RANK && right == INVALID_RANK )
    {
        MPI_Cart_coords(g_comm, up, DIMENSIONS, coord_old);
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]-1;
        MPI_Cart_rank( g_comm, coord_new, &next_rank);
        //printf("bottom-right R[%d] <-> %d\n", g_rank, next_rank);

        for ( i=0; i< K; i++) 
        {
            MPI_Sendrecv( &A_part_in[K+i][K],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[i][0], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
        }
    }

    // 3) top-right corner
    if ( up == INVALID_RANK && left != INVALID_RANK && 
         dn != INVALID_RANK && right == INVALID_RANK )
    {
        MPI_Cart_coords(g_comm, dn, DIMENSIONS, coord_old);
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]-1;
        MPI_Cart_rank( g_comm, coord_new, &next_rank);
        //printf("top-right R[%d] <-> %d\n", g_rank, next_rank);

        for ( i=0; i< K; i++) 
        {
            MPI_Sendrecv( &A_part_in[h+i][K],   K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][0], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
        }
    }

    // 4) bottom-left corner
    if ( up != INVALID_RANK && left == INVALID_RANK && 
         dn == INVALID_RANK && right != INVALID_RANK )
    {
        MPI_Cart_coords(g_comm, up, DIMENSIONS, coord_old);
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]+1;
        MPI_Cart_rank( g_comm, coord_new, &next_rank);
        //printf("bottom-left R[%d] <-> %d\n", g_rank, next_rank);
        
        for ( i=0; i< K; i++) 
        {
            MPI_Sendrecv( &A_part_in[K+i][w],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
            
        }
    }


    // 5) (I am) mid-left  => sendrecv x 2 ( up + dn)
    //     allow for any mid-left+ (either the right-side can be "rim" or "not-rim")
    if ( up != INVALID_RANK && /*left != INVALID_RANK && */
         dn != INVALID_RANK && right != INVALID_RANK )
    {
        // communicate with right-up node (45 deg)
        MPI_Cart_coords(g_comm, up, DIMENSIONS, coord_old);
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]+1;
        MPI_Cart_rank( g_comm, coord_new, &prev_rank);
        // communicate with right-down node (-45 deg)
        MPI_Cart_coords(g_comm, dn, DIMENSIONS, coord_old);
        coord_new2[0] = coord_old[0];
        coord_new2[1] = coord_old[1]+1;
        MPI_Cart_rank( g_comm, coord_new2, &next_rank);
        //printf("mid-left[%d] <-> righ-up:%d & right-dn:%d\n", g_rank, prev_rank, next_rank);

        for ( i=0; i< K; i++) 
        {
            // to righte-up
            MPI_Sendrecv( &A_part_in[K+i][w], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[i][w+K], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
            // to right-down 
            MPI_Sendrecv( &A_part_in[h+i][w],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);            
        }
    }

    // 6) (I am) mid-right => sendrecv x 2 ( up + dn)
    //     allow for any mid-right+ (either the right-side can be "rim" or "not-rim")
    if ( up != INVALID_RANK && left != INVALID_RANK  && 
         dn != INVALID_RANK /*&& right != INVALID_RANK*/ )
    {
        // communicate with left-up node (135 deg)
        MPI_Cart_coords(g_comm, up, DIMENSIONS, coord_old);
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]-1;
        MPI_Cart_rank( g_comm, coord_new, &prev_rank);
        // communicate with left-down node (-45 deg)
        MPI_Cart_coords(g_comm, dn, DIMENSIONS, coord_old);
        coord_new2[0] = coord_old[0];
        coord_new2[1] = coord_old[1]-1;
        MPI_Cart_rank( g_comm, coord_new2, &next_rank);
        //printf("mid-right[%d] <-> left-up:%d & left-dn:%d\n", g_rank, prev_rank, next_rank);
        for ( i=0; i< K; i++) 
        {
            // to left-up
            MPI_Sendrecv( &A_part_in[K+i][K], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[i][0], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
            // to left-down 
            MPI_Sendrecv( &A_part_in[h+i][K],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][0], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);            
        }
    }

    
    // 7) (I am) mid-top => sendrecv x 2 ( left + right)
    if ( up == INVALID_RANK && left != INVALID_RANK  && 
         dn != INVALID_RANK && right != INVALID_RANK )
    {
        MPI_Cart_coords(g_comm, dn, DIMENSIONS, coord_old);
        // comm with left-dn (-135 deg)
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]-1;
        MPI_Cart_rank( g_comm, coord_new, &prev_rank);
        // comm with right-dn (-45 deg)
        coord_new2[0] = coord_old[0];
        coord_new2[1] = coord_old[1]+1;
        MPI_Cart_rank( g_comm, coord_new2, &next_rank);
        //printf("mid-top R[%d] <-> left:%d & right:%d\n", g_rank, prev_rank, next_rank);

        for ( i=0; i< K; i++) 
        {
            // right-dn
            MPI_Sendrecv( &A_part_in[h+i][w],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[h+K+i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
            // left-dn
            MPI_Sendrecv( &A_part_in[h+i][K],   K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[h+K+i][0], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
        }
    }


    // 8) (I am) mid-bottom => sendrecv x 2 ( left+right)
    if ( up != INVALID_RANK && left != INVALID_RANK  && 
         dn == INVALID_RANK &&  right != INVALID_RANK )
    {
        MPI_Cart_coords(g_comm, up, DIMENSIONS, coord_old);
        // comm with left-up (135 deg)
        coord_new[0] = coord_old[0];
        coord_new[1] = coord_old[1]-1;
        MPI_Cart_rank( g_comm, coord_new, &prev_rank);
        // comm with right-up (45 deg)
        coord_new2[0] = coord_old[0];
        coord_new2[1] = coord_old[1]+1;
        MPI_Cart_rank( g_comm, coord_new2, &next_rank);
        //printf("mid-bottom R[%d] <-> left:%d & right:%d\n", g_rank, prev_rank, next_rank);

        for ( i=0; i< K; i++) 
        {
            // right-up
            MPI_Sendrecv( &A_part_in[K+i][w],     K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          &A_part_in[i][w+K], K, MPI_UNSIGNED_CHAR, next_rank, tag0, 
                          g_comm, &status);
            // left-up
            MPI_Sendrecv( &A_part_in[K+i][K],   K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          &A_part_in[i][0], K, MPI_UNSIGNED_CHAR, prev_rank, tag0, 
                          g_comm, &status);
        }
    }


    MPI_Barrier( g_comm );
    #if DEBUG
    if ( g_rank == 5 ) 
    {
        print_matrix( g_rank, A_part_in, h+2*K, w+2*K, 0, h+2*K, 0, w+2*K); 
    }
    #endif
}


/* **************************************************************
 * @func: scatter
 * @desc: to calculate displacement and scatter data from root
 *        node to other nodes under the grid topology.
 *        This function should work similar to define data type
 *        as MPI_Type_vector combine with MPI_Scatterv
 * @params:
 * ======== 
 *  
 * @return partial image as 2-D array. The caller has the responsibility
 *         to free the space after use.
 * ************************************************************ */ 
uchar** scatter_image( int nprocs, int* g_coords, int g_comm, uchar* A_vec, 
                 int total_width, int h, int w, int* _displ /*out*/ )
{
    MPI_Status status;                        // send/receive status
    int     g_rank;                           // grid rank
    uchar** A_part_in;                        // partial image
    int     tag0=0;                           // send/receive tag
    int     i,j;                              // workaround variables
    int     displ;                            // displacement
    int     coll_displ[nprocs][h];            // collective displacement


    MPI_Comm_rank( g_comm, &g_rank); 

    // =====================================================
    // allocate memory for part image as 2-D data
    // =====================================================
    A_part_in = malloc( sizeof(uchar *) * (h+2*K) );
    for ( i=0; i< h+2*K; i++ )
    {
        A_part_in[i] = malloc( sizeof(uchar) * (w+2*K) );
        memset( A_part_in[i], 0, sizeof(uchar)*(w+2*K) );
    }

    // =====================================================
    // calculate displacement and scatter data from root
    // node to other nodes under the grid topology.
    // =====================================================

    if ( g_rank == 0 ) 
    {
        for( i=0; i< h; i++ )
        {  
            for (j=1; j< nprocs; j++) {
                displ = ( total_width * h * g_coords[j*DIMENSIONS+0] ) + 
                        ( w * g_coords[j*DIMENSIONS+1] ) +
                        ( total_width * i );
                coll_displ[j][i] = displ;
               #if 0 // DEBUG_COMM
                printf("R[%d] i[%d] coords(%d, %d) disp[%d] A[%d]\n",
                       j, i, g_coords[j*DIMENSIONS+0], g_coords[j*DIMENSIONS+1], 
                       displ, A_vec[displ]);
                #endif
                MPI_Send( &A_vec[displ], w, MPI_UNSIGNED_CHAR, j, tag0, g_comm);
            }
            displ = ( total_width * h * g_coords[0] ) + 
                    ( w * g_coords[1] ) +
                    ( total_width * i );
            coll_displ[0][i] = displ;
            #if 0 // DEBUG_COMM
            printf("R[%d] i[%d] coords(%d, %d) disp[%d] A[%d]\n",
                   g_rank, i, g_coords[j*DIMENSIONS+0], g_coords[j*DIMENSIONS+1], 
                   displ, A_vec[displ]);
           #endif
            memcpy( &A_part_in[i+K][K], &A_vec[displ], sizeof(uchar)*w );
        }
    }
    else
    {
        for( i=0; i< h; i++ )
        {  
            MPI_Recv( &A_part_in[i+K][K], w, MPI_UNSIGNED_CHAR, ROOT_NODE, 
                      tag0, g_comm, &status);
        }
    }
    
    // =====================================================
    // broadcast displ to each process to be used later in writing file
    // =====================================================
    MPI_Scatter ( &coll_displ[0][0], h, MPI_INT, 
                  _displ,             h, MPI_INT, ROOT_NODE, g_comm ); 
    #if DEBUG
    for ( i=0; i < h; i++ ) {
        printf("R[%d] i[%d] disp[%d]\n",g_rank,i, _displ[i]);
    }
    #endif
    
    return A_part_in;

}


/* **************************************************************
 * @func: write_image_sync
 * @desc: write part image to file synchronously from all processes
 *
 * @params:
 * ======== 
 *  
 * ************************************************************ */ 

void write_image_sync( char* filename, uchar** A_part_out, int* displ, int w, int h, 
                       int g_comm )
{
    int        err;
    MPI_File   fh;
    MPI_Status status;
    char       buf[100];
    int        len;
    int        eclass;
    int        i;
    int        g_rank;
    int        errcnt;

    err = MPI_File_open( g_comm , filename, 
                         MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh );
    if (err) {
        MPI_Error_string(err, buf, &len);
        printf("Error from open file to write %d: %s\n", eclass, buf);
        MPI_Abort( g_comm, 1); 
    }

    MPI_File_set_atomicity( fh, true );
    
    // =====================================================
    //  Writing data as 1-D block at once should be much 
    //  fater than writing data each row to file.
    // =====================================================

    for ( i=0; i<h; i++ ) {
        MPI_File_write_at_all( fh, displ[i], &A_part_out[i+K][K], w, 
                               MPI_UNSIGNED_CHAR, &status );
    }
    
    MPI_File_sync( fh ); // force flush
    MPI_Barrier( g_comm );
    
    // =====================================================
    //  Clean up
    // =====================================================
    MPI_File_close( &fh );   

}

/* **************************************************************
 * @func:  main 
 * @desc:     
 * ====== 
 * - Open and read image A
 * - Scatter block image A as part to other processes
 * - Blur each partial image using filter F m times
 * - Write part image from all processes to file 
 *
 * ************************************************************ */ 
int main( int argc, char* argv[] ) 
{
    MPI_Comm  g_comm;                         // grid topology
    int      rank; 			                  // rank of the process
    int      nprocs;                          // number of processes
    int      i;                               // iteration variable
    uchar*   A_vec;                           // image A in 1-D (reading)
    uchar**  A_part_in;                       // block image(2-D) as input to func
    uchar**  A_part_out;                      // block image(2-D) as output from func
    int      w;                               // width of each image block
    int      h;                               // height of each image block
    int      m;                               // number of applying filter to image
    int      g_rank;                          // grid rank to seperate it from world comm
    int      total_width;                     // total width of the image
    int      H;                               // total height of image, given by user
    int      dims[DIMENSIONS] = {0,0};        // dimension of grid topology
    int      wrap[DIMENSIONS] = {NOWRAP, NOWRAP}; // using no-wrap for this grid
    int      coords[DIMENSIONS];              // coordinates for this rank
    int*     displ;                           // displacement of each block in part image
    int      g_coords[DIMENSIONS*MAX_NUM_PROCS]; // collective coordinates


    MPI_Init(&argc, &argv);		 		                        
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );             
    MPI_Comm_rank( MPI_COMM_WORLD, &rank); 

    // ==============================================================
    //  Check valid inputs (passing arguments)
    // ==============================================================

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
   H = atoi(argv[2]);
   if ( H%2 != 0 ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: The height of image, %d, is not able to mod 2!\n", H);
       }
       MPI_Finalize();		
       return 0;	
   }

   // grid size possibilities: 2, 4, 6, 8, 12, 16..
   if ( nprocs%2 != 0 ) {
       if ( rank == ROOT_NODE ) {
           printf("Error: Number of processes given,%d, mod 2 != 0 !\n", nprocs);
       }
       MPI_Finalize();		
       return 0;	
   }
   
   // ==============================================================
   // Compute size of grids and create its topology
   // ==============================================================
   MPI_Dims_create( nprocs, DIMENSIONS, dims);
   MPI_Cart_create( MPI_COMM_WORLD, DIMENSIONS, dims, wrap, true, &g_comm );
   MPI_Comm_rank( g_comm, &g_rank); 
   MPI_Cart_coords( g_comm, g_rank, DIMENSIONS, coords);
   //printf("dims = %d x %d \n", dims[0], dims[1]);
   //printf("R[%d] coords(%d,%d)\n",g_rank, coords[0], coords[1]);

   // ==============================================================
   //   Open image and get width w and read image A (1-D)
   //   from cart_ROOT_NODE 
   // ==============================================================
   if ( g_rank == ROOT_NODE ) {
       A_vec = open_n_read_image(argv[1], H, &total_width);
   }
   MPI_Bcast( &total_width, 1, MPI_INT, ROOT_NODE, g_comm ); 

   // calculate w and h for each part image
   h = H/dims[0];
   w = total_width/dims[1];
   //printf("R[%d] height=%d, width=%d, dims[%d,%d]\n",g_rank, h, w, dims[0], dims[1]);

   // ==============================================================
   // collect the coords to calculate the displacement of each procs
   // ==============================================================
   MPI_Allgather( coords, DIMENSIONS, MPI_INT, 
                  g_coords, DIMENSIONS, MPI_INT,  g_comm);

   // ==============================================================
   //   Scatter 1-d array from root-node to othernode as part image.
   // ==============================================================
   // We want to reuse the calculated displacements again in writing
   // part of image to file. Therefore, the data positions using in
   // scatter_data must be stored and will be re-sued when recompose the
   // image during writing.
   displ = malloc( sizeof(int)*h ); 

   // Important part of this checkerboard is to scatter data from
   // root node to the other processes in grid topology.
   // This func call behaves as MPI_Scatterv with vector as data type.
   A_part_in = scatter_image( nprocs, g_coords, g_comm, A_vec, total_width, h, w, displ);
   

   // =====================================================================
   //  Main filtering/blurring image "m" times
   // =====================================================================
  
   while (m > 0 ) {
       
       // =======================================================
       //  Exchange the data for each block of the grid inside the call.
       //  The exchange process will be repeated for each blurring.
       // =======================================================
       exchange_grid_data( g_rank, g_comm, A_part_in, w, h, dims );    
       MPI_Barrier( g_comm );
       

       // =======================================================
       //  Blurring image m times
       // =======================================================

       A_part_out = blurring( A_part_in, h+2*K, w+2*K );
       --m;

       // If there is still next iteration , we copy output image to be next input
       // using old memory and free unused memory.
       // Every call of "blurring" will allocate new memory for
       // output image (A_part_out)
       if ( m > 0 ) {
           for( i=0; i<h+2*K; i++ ) {
               memcpy( &(A_part_in[i][0]), &(A_part_out[i][0]), (w+2*K)*sizeof(uchar) );
               free( A_part_out[i] );
           }
           free( A_part_out );
       }

   }

   // ======================================================
   //  Write partial image to file synchronously. 
   // ======================================================
   write_image_sync( argv[4],  A_part_in, displ, w, h, g_comm );

   if ( g_rank == ROOT_NODE ) {
       printf("====================================================\n" );
       printf("Distributed Image Processing: Blurring Image on Grid\n");
       printf("=====================================================\n");
		 printf("Image file (i/p):                 %s\n", argv[1]);
       printf("Image size:                [%d x %d]\n",total_width, H);
       printf("No. of processes:                 %d\n",nprocs);
       printf("Grid dimensions:            [%d x %d]\n", dims[0],dims[1]);
       printf("No. of applying filter:            %d\n", atoi(argv[3]));
       printf("--------\n");
       printf("Blured image (o/p):                %s\n", argv[4]);
       printf("=====================================================\n");
   }
   // ======================================================
   //  Clean up
   // ======================================================
   if ( rank == ROOT_NODE ) {
       free ( A_vec );
   }

   for( i=0; i<h+2*K; i++ ) {
       free( A_part_in[i] );
       free( A_part_out[i] );
   }
   free( A_part_in );
   free( A_part_out );
   free ( displ );

   MPI_Finalize();		
   return 0;	

}

////////////////////////////////END OF FILE///////////////////////////
