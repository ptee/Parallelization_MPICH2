/* *****************************************************************************
 *
 *  @file    cimage.h
 *  @brief   common image functions used in assignment 4
 *  
 *  @author  Pattreeya Tanisaro
 *
 * **************************************************************************** */

#ifndef CIMAGE_H
#define CIMAGE_H


#include <math.h>                       // for fabs(..)
#include <stdio.h> 	                   // import of the definitions of the C IO library
#include "commonc.h"                    // self-defined common c utility func
#include "iocommon.h"                   // mpiio common 

#define   K                  2          // filter size of blur filter (2k+1)x(2k+1)
#define   MAX_PIXEL_VALUE  255          // max of uchar is white pixel
#define   MIN_PIXEL_VALUE    0          // min of uchar is black pixel
#define   MAX_FILTERING    100          // maximum times of applying filters
#define   MIN_FILTERING      1          // minimum times of applying filters
#define   MAX_NUM_REQS     256          // maximum number of requests
#define   REQ_RECV           0          // request for receving start index
#define   REQ_SEND           K          // request for sending start index
#define   NOWRAP             0          // not periodic
#define   DIMENSIONS         2          // number of dimensions, we use 2-D grid
#define   INVALID_RANK      -1          // invalid rank receving from MPI_cart_shift


// macro to check number of multiplier of the blurring filter
#define   VALID_NUM_FILTERS(m) ((m<MIN_FILTERING)||(m>MAX_FILTERING))? false: true

// ==============================================================
//  Blur Filter with mask size 5x5 
// ==============================================================

static double  F[2*K+1][2*K+1] = { {0,              0,   0.027027,        0,        0 },
                                   {0,       0.054054,   0.10811,   0.054054,       0 },
                                   {0.027027, 0.10811,   0.24324,   0.10811,  0.027027},
                                   {0,       0.054054,   0.10811,   0.054054,       0 },
                                   {0,              0,   0.027027,         0,       0 } };

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
	 printf("\nImage Processing Filter to Blur Image\n");
    printf("---------------------------------------\n");
    printf("Usage: %s [GrayImageFile] [Height] [Repetition] [BlurImage]\n\n",prog);
    printf("GrayImageFile    : gray image with depth = 8 as input\n");
    printf("Height           : image height in pixel e.g. 960 as input\n");
	 printf("FilterRepetition : number of repetition to apply blurring filter\n");
    printf("BulrImage        : Output filename ( blurred image )\n");
    printf("Examples: %s ffm_1280x960.gray 960 3 out_1280x960.gray\n",prog);
	 printf("---------------------------------------\n\n");
}

/*	**************************************************************
 * @func:    open_n_read_image
 * @desc:    Open and read image as 1-d vector
 * @params:
 * ========
 * filename [in] -> image file name (gray image)
 * w        [out] -> as width of the image
 *
 * @return   pointer to uchar of gray image as 1-d vector.
 *           The calling function has the responsibility to 
 *           free the memory!
 *	************************************************************ */ 

uchar* open_n_read_image(char* filename, int H, int* w)
{
    int        err;
    int        i,j;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fhA; 
    char       buf[100];
    int        len;
    int        eclass;
    uchar*     A;
    int        _w;

    // ====================================================== 
    //  image file to read
    // ======================================================

    err = MPI_File_open( MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhA );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    err = MPI_File_set_view( fhA, 0, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, 
                             "native", MPI_INFO_NULL);

    // ======================================================
    // get matrix width
    // ======================================================

    err = MPI_File_get_size( fhA, &fsize);
    _w = (int)(fsize/sizeof(uchar))/H;
    
    // ======================================================
    // read image file
    // ======================================================

    A = malloc( sizeof(uchar) * _w * H);
    err = MPI_File_read( fhA, A, _w*H, MPI_UNSIGNED_CHAR, &status );

    *w  = _w;

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fhA );
    return A;
}


/*	**************************************************************
 * @func:    write_image
 * @desc:    Write data to file
 * @params:
 * ========
 * filename [in/out] -> image file name (gray image)
 * w        [in]     -> as width of the image
 * A        [in]     -> image
 * 
 *	************************************************************ */ 

void write_image( char* filename, uchar* A, int w, int h )
{
    int        err;
    MPI_File   fh;
    MPI_Status status;
    char       buf[100];
    int        len;
    int        eclass;
    int        j;

    err = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                        MPI_INFO_NULL, &fh);
    if (err) {
        MPI_Error_string(err, buf, &len);
        printf("Error from open file to write %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    
    err = MPI_File_set_view( fh, 0, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, 
                             "native", MPI_INFO_NULL);

    // =====================================================
    //  Writing data as 1-D block at once should be much 
    //  fater than writing data each row to file.
    // =====================================================
    printf("writing file...%s, wxh\n",filename, w,h);
    MPI_File_write(fh, A, w*h, MPI_UNSIGNED_CHAR, &status );
    MPI_File_sync( fh ); // force flush

    // =====================================================
    //  Clean up
    // =====================================================
    MPI_File_close( &fh );   
}


/*	**************************************************************
 * @func:    bluring
 * @desc:    blurring image using 5x5 filter
 * @params:
 * ========
 * A_part_in   [in] ->  2-D partial image as input,
 *                      The caller owns this pointer.
 * rank        [in] ->  processor rank
 * rows        [in] ->  rows or height of the partial image,
 *                      A_part_in 
 * w           [in] ->  image width
 *
 * @return   uchar**, the caller has to delete its meomory!!
 *
 *	************************************************************ */ 

uchar** blurring(uchar** A_part_in, int rows, int w)
{
    uchar** A_part_out;                        
    int     i,j;                           // variables for image interation
    int     v,u;                           // variables for filter
    float   sum;

    // =====================================================
    //  Allocate memory for filtered  for output (partial) image.
    //  This pointer is free by the caller function!
    // =====================================================

    A_part_out = malloc( sizeof(uchar *) * rows );
    for( j=0; j<rows; j++ ) {
        A_part_out[j] = malloc( sizeof(uchar) * w );
        memset( &(A_part_out[j][0]), 0, w*sizeof(uchar) );
    }


    // =====================================================
    //  Filtering 
    //         ~direction used in this algorithm~   
    // -------------------> <i> x-axis
    // |                <w>
    // |
    // | <rows>
    // | 
    // v  <H> = 960 fixed
    // <j> y-axis
    //
    // =====================================================

    for( j=K; j<rows-K; j++ ) // y-axis, start from K...rows-K, K is the border size
    {
        for( i=K; i<w-K; i++ ) // x-axis, start from K...w-K
        {
            sum = 0.0;

            // convolution A[y][x] = sum(sum( F[v][u]*A[y..][x..] ))
            for( v=-K; v<=K; v++ ) {
                for( u=-K; u<=K; u++ ) {
                    sum += F[v+K][u+K] * (A_part_in[j+v][i+u]);
                }
            }

            A_part_out[j][i] = ((float)floor(sum) > (float)MAX_PIXEL_VALUE)? \
                (uchar)MAX_PIXEL_VALUE : (uchar)floor(sum);

        } // end for x-axis

    } // end for y-axis

    return A_part_out;
}


#endif // CIMAGE_H

///////////////////////////////////EOF//////////////////////////////////
