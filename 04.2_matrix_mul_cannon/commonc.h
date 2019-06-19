/* *****************************************************************************
 *
 *  @file    commonc.h
 *  @brief   common definitions and self-defined func using in C programming
 *           for compinging with MPI
 *  @author  Pattreeya Tanisaro
 *
 * **************************************************************************** */

#ifndef COMMONC_H
#define COMMONC_H

typedef int                    bool;           // quick fix 
typedef unsigned long int      ulong;          // for convenience
typedef unsigned int           uint;           // for convenience
typedef unsigned char          uchar;           // for convenience
#define false                  0               // quick fix for false
#define true                   1               // quick fix for true
#define MAX_MESG_BUFF          4096            // size of the buffer
#define MAX_SMALL_MESG_BUFF    256             
#define MAX_MIDDLE_MESG_BUFF   1024
#define MAX_LARGE_MESG_BUFF    4096
#define MAX_HUGE_MESG_BUFF     8192
#define MAX_VERY_SMALL_BUFF      64          


// !!! No operator on value allowed when passing to these macros !!!
#define MIN(a,b)    ((a)>(b)? (b):(a))           // get minimum
#define MAX(a,b)    ((a)>(b)? (a):(b))           // get maximum


/* **************************************************************
 * @func: timeval2msec 
 * @desc: convert struct timeval to ms(milliseconds)
 * @params: [in] tv: struct timeval
 * @return: unsign long int as millisecond
 *
 * ************************************************************ */

unsigned long int timeval2msec(struct timeval tv ) 
{
    return ((tv.tv_sec * 1000) + (tv.tv_usec / 1000));
}


/* **************************************************************
 * @func: timeval2microsec
 * @desc: convert struct timeval to micro sec
 * @params: [in] tv: struct timeval
 * @return: unsign long int as micro sec
 *
 * ************************************************************ */

unsigned long int timeval2microsec(struct timeval tv ) 
{
    return tv.tv_sec*1000000 + (tv.tv_usec);
}



#endif // COMMONC_H
////////////////////////////////END OF FILE/////////////////////////////////////
