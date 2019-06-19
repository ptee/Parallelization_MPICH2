/* *****************************************************************************
 *
 *  @file    iocommon.h
 *  @brief   common io operations used in assignment 4
 *  
 *  @author  Pattreeya Tanisaro
 *
 * **************************************************************************** */

#ifndef IOCOMMON_H
#define IOCOMMON_H


#include <math.h>                       // for fabs(..)
#include <stdio.h> 	                   // import of the definitions of the C IO library
#include "commonc.h"                    // self-defined common c utility func


#define   ROOT_NODE          0
#define   MAX_NUM_PROCS   1024



/*	**************************************************************
 * @func: print_matrix
 * @desc: print out matrix in range width in range (lower_w, upper_w) and 
 *                                  height in range (lower_h, upper_h)
 * @params:
 * ======== 
 *
 *	************************************************************ */ 

void print_matrix( int rank, uchar** matrix, int h, int w, 
                   int lower_limit_h, int upper_limit_h /*open bracket, not incl it */,
                   int lower_limit_w, int upper_limit_w /*open bracket, not incl it */  )
{
    int j,i;
    if (    upper_limit_w < 0 || upper_limit_w > w \
         || upper_limit_h < 0 || upper_limit_h > h \
         || lower_limit_w < 0 || lower_limit_w > w \
         || lower_limit_h < 0 || lower_limit_h > h \
         || lower_limit_w >= upper_limit_w          \
         || lower_limit_h >= upper_limit_h  )
        return;
    
    printf("hxw=%dx%d, h[%d,%d), w[%d,%d)\n", 
           h, w, lower_limit_h, upper_limit_h, lower_limit_w, upper_limit_w);

    for ( j=0; j<h; j++) {
        for ( i=0; i<w; i++) {
            if (    (j >= lower_limit_h ) && (j < upper_limit_h ) 
                 && (i >= lower_limit_w)  && (i < upper_limit_w ) 
                ) {
                printf("R[%d] A[%d][%d] = %d\n", rank, j, i, matrix[j][i]);
            }
        }
    } 
    printf("-------------- end print_matrix ------------\n");
}


/*	**************************************************************
 * @func: print_vector
 * @desc: print out matrix in range width in range (lower_w, upper_w) and 
 *                                  height in range (lower_h, upper_h)
 * @params:
 * ======== 
 *
 *	************************************************************ */ 

void print_vector( int rank, uchar* vector, int len, 
                   int lower_limit /*close*/, int upper_limit /*open*/ )
{
    int j,i;

    if ( (upper_limit < 0)   || 
         (upper_limit > len) || 
         (lower_limit >= upper_limit) )
        return;

    for ( j=0; j<len; j++) {
        if (    (j >= lower_limit ) && (j < upper_limit )  ) 
            printf("R[%d] V[%d] = %d\n", rank, j, vector[j]);
    } 
    printf("-------------- end print_vector ------------\n");
}


/*	**************************************************************
 * @func:    open_n_read_sqrmatrix_1d
 * @desc:    Open and read matrix A as 1-D
 * @params:
 * ========
 * filename [in] -> matrix file A 
 * n        [in] -> int from matrix A nxn
 *
 * @return   pointer to double (1d vector) from  matrix file.
 *           The calling function has the responsibility to 
 *           free the memory!
 *	************************************************************ */ 

double* open_n_read_sqrmatrix_1d(char* filename,  int* n)
{
    int        err;
    int        i,j;
    MPI_Offset fsize;
    MPI_Status status;
    MPI_File   fh; 
    char       buf[100];
    int        len;
    int        eclass;
    double*    _A;
    int        _n;

    // ====================================================== 
    // file to read
    // ======================================================

    err = MPI_File_open( MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh );
    if (err != MPI_SUCCESS ) { 
        MPI_Error_string(err, buf, &len);
        printf("Error from opening file %d: %s\n", eclass, buf);
        MPI_Abort( MPI_COMM_WORLD, 1); 
    }

    err = MPI_File_set_view( fh, 0, MPI_DOUBLE, MPI_DOUBLE, 
                             "native", MPI_INFO_NULL);

    // ======================================================
    // get matrix size and  check if it is a square matrix
    // ======================================================

    err = MPI_File_get_size( fh, &fsize);
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
    err = MPI_File_read( fh, _A, _n*_n, MPI_DOUBLE, &status );

    *n  = _n;

    // ======================================================
    // clean up
    // ======================================================
    MPI_File_close( &fh );
    return _A;
}


#endif // IOCOMMON_H

//////////////////////////////EOF//////////////////////////////////
