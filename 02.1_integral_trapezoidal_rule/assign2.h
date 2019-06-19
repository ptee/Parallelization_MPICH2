/* *****************************************************************************
 *
 *  @file    assign2.h
 *  @brief   common functions used in assignment 2
 *  
 *  @author  Pattreeya Tanisaro
 *
 * **************************************************************************** */

#ifndef ASSIGN2_H
#define ASSIGN2_H


#include <math.h>                       // for log & pow
#include <stdio.h> 	                // import of the definitions of the C IO library
#include "commonc.h"                    // self-defined common c utility func


/* **************************************************************
 * @func: func func1 func2 ...
 * @desc: select function pointer to calculate various forms of function
 *        which given via the command-line
 * @params:
 * ======== 
 * x [in]  => double as input parameters
 *
 * @return: double as the output
 *
 * ************************************************************ */ 

typedef double (*ptFunc)(double);                  

double func1 (double x) { return 1/x; }                       // for select 'a'
double func2 (double x) { return pow(sin(x),2); }             // for select 'b' 

ptFunc func(char f_c )
{
	 ptFunc f = &func2;
	 switch (f_c)
	 {
	    case 'a': f = &func1; break;
	    case 'b': f = &func2; break;
	    default: break;
	 }
	 return f;

}

/* **************************************************************
 * @func: print_usage
 * @desc: print out how to use this program
 * @params:
 * ======== 
 *	prog [in] => char* as execultable
 *
 * ************************************************************ */ 

void print_usage(char *prog)
{
	 printf("\nComposite Trapezoidal Rule to integrate the function f(x)\n");
	 printf("----------------------------\n");
	 printf("Usage: %s [f(x)] [num of interval] [lower bound] [upper bound]\n",prog );
	 printf("[f(x)] => a for f(x) =  pow( 2, sin(x) )\n");
	 printf("[f(x)] => b for f(x) =  1/(x)\n");
	 printf("[f(x)] => c...z for f(x) =  1/(x)\n");
	 printf("Exmaples: %s  a  1000000  0.1  10000\n\n\n",prog);
}


/* **************************************************************
 * @func: check_valid_input
 * @desc: check if input arguments are valid to proceed
 * @params:
 * ======== 
 * f_c [in] => char as to select function f(x) 
 * N   [in] => int as number of interval
 * a   [in] => double as lower limit
 * b   [in] => double as upper limt
 * num_procs [in] => int as number of processes
 *
 * @return: true if all inputs are okay, false otherwise
 *
 * ************************************************************ */ 

bool check_valid_input(char f_c, int N, double a, double b, int num_procs)
{
	 bool ok;
	 ok = (b > a)? true: false;
	 ok = ok && ( (N > 0) && (N > num_procs) ) ? true: false;
	 ok = ok && ( isalpha(f_c) ) ? true: false;
	 return ok;
}


/* **************************************************************
 * @func: composite_trapezoidal_part
 * @desc: calculate part of composite trapezoidal rule
 *        sum_partial = h* sum[ f(a+kh) ]
 * @params:
 * ======== 
 * N   [in] => int as number of interval
 * a   [in] => double as lower limit
 * b   [in] => double as upper limt
 * num_procs [in] => int as number of processes
 * my_rank   [in] => int as rank of the process
 * ptFunc    [in] => function pointer to function we want to calculate
 *
 * @return: partial sum of composite trapezoidal as double
 *
 * ************************************************************ */ 

double composite_trapezoidal_part( int N, double a, double b, int num_procs, int my_rank, 
                                   double (*ptFunc)(double)  )
{
	 double h;
	 double sum = 0.0;
	 int    i;

	 h = (b-a)/N;
	 for ( i = my_rank+1; i <= N-1; i += num_procs ) {
	     sum += h*( (*ptFunc)(a+i*h) );
	 }
	 return sum;
}


#endif // ASSIGN2_H
////////////////////////////////END OF FILE/////////////////////////////////////
