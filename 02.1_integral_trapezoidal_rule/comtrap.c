/* *****************************************************************************
 * Program:    comtrap.c                                                  
 * Author:     Pattreeya Tanisaro                    
 * Description:  To calculate the integral of the function using Composite Trapezoidal Rule
 * ===========   and measure the cost for sequential, parallel and communication components.
 *
 * Parameters:
 * ==========
 * Input: argv[1] => a character to function selection
 *                   'a' to calc 1/x, 'b' to calc pow(sin(x), 2) and so on.
 *	  argv[2] => numbers of interval           
 *        argv[3] => lower bound
 *        argv[4] => upper bound
 *
 * Output: display => the integral of func f(x) in the the limit of [a,b]
 *                    with N interval.
 *                 => sequential and parallel and communication cost
 *	
 * *************************************************************************** */ 

#include "mpi.h" 	               // import of the MPI definitions
#include <stdio.h> 	               // import of the definitions of the C IO library
#include <string.h>                    // import of the definitions of the string operations
#include <unistd.h>	               // standard unix io library definitions and declarations
#include <errno.h>	               // system error numbers
#include <sys/time.h>	               // speicial system time functions for c
#include <stdlib.h>                    // avoid warning from malloc/free
#include <math.h>                      // for log & pow
#include <limits.h>                    // for INT_MIN, INT_MAX
#include "assign2.h"                   // common functions for assignment 2

#define TEST_DEBUG      0              // to test small set of data
#define ROOT_NODE       0              // define root-node using node 0
 

/* **************************************************************
 * @func: main
 * @desc: main
 * @params:
 * ======== 
 * Input: argv[1] => a char to select a function to perform the integratral
 *                   'a' to calc 1/x, 'b' to calc pow(sin(x), 2) and so on.
 *	  argv[2] => numbers of interval           
 *        argv[3] => lower bound
 *        argv[4] => upper bound
 *
 * ************************************************************ */ 

int main(int argc, char* argv[ ]) 
{ 

	 int    my_rank; 			            // rank of the process
	 int    num_procs;                                  // number of processes
	 int    i;                                          // misc variables
	 bool   ok;                                         // to verify the return value
	 double  sum, tmp_sum = 0, g_sum = 0;               // sum of (integral) calculation
	 MPI_Status status;                                 // received status
	 struct timeval calc_start, calc_stop;              // start and end time of calculation
	 struct timeval comm_start, comm_stop;              // start and end time of communication
	 int    sum_tag = 0;                                // tags for sending/receiving calc
	 double a, b;                                       // lower bound, upper bound
	 int    N;                                          // Intervals  
	 double h;                                          // equidistant h = (b-a)/N
	 char   f_c;                                        // input argument as function selection

	 
	MPI_Init(&argc, &argv);		 		                        
	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );             
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

	// ==============================================================
	// check valid inputs (passing arguments)
	// ==============================================================

	if ( argc == 5) {
		 f_c = argv[1][0];
		 N  = atoi(argv[2]);
		 a  = atof(argv[3]); 
		 b  = atof(argv[4]);
		 ok = check_valid_input(f_c, N, a, b, num_procs);
		 if ( !ok ) {
			  if ( my_rank == 0 ) {
					printf("\nError: Your inputs are invalid! Check your inputs again!\n\n");
			  }
			  MPI_Finalize();		
			  exit(1);
		 }
	}
	else {
		 if ( my_rank == 0 ) {
			  print_usage( argv[0] );
		 }
		 MPI_Finalize();		
		 return 0;
	}


	// ==============================================================
	// calculate partial sum of composite trapezoidal rule
	// ==============================================================
	
	// start timer for calculation
	MPI_Barrier( MPI_COMM_WORLD );
	gettimeofday( &calc_start, NULL );

	sum = composite_trapezoidal_part(N, a, b, num_procs, my_rank, func(f_c) ); 

	// stop timer for calculation
	MPI_Barrier( MPI_COMM_WORLD );
	gettimeofday( &calc_stop, NULL);
	

	// ==============================================================
	// To send partial sum from all other nodes to root node
	//
	//  <<_All_nodes_except_Root_Node>>  ==>   <<_Root_node_>> 
	//  Don't send & receive to its own node because it might cause deadlock!
	//
	// ==============================================================
	
	// start timer for communication
	gettimeofday( &comm_start, NULL);

	if (my_rank == ROOT_NODE) {
		 g_sum += sum;                                    
		 for ( i=1; i < num_procs; ++i ) {
			  MPI_Recv(&tmp_sum, 1, MPI_DOUBLE, i, sum_tag,  MPI_COMM_WORLD, &status);
			  g_sum += tmp_sum;
           	 #if TEST_DEBUG
			  printf("Recv[%d]->[%d] recv_sum: %f, total_sum: %f\n",my_rank,i, tmp_sum, g_sum);
           	 #endif
		 }                                
	}
	else {
			 MPI_Send( &sum, 1, MPI_DOUBLE, 0, sum_tag,  MPI_COMM_WORLD);
	}

	// stop timer for communication
	MPI_Barrier( MPI_COMM_WORLD );
	gettimeofday( &comm_stop, NULL);

	// ===================================================================
	// calculate the final output for the integral & print output summary
	// ===================================================================

	if ( my_rank == ROOT_NODE ) {

		 g_sum = ((b-a)/N)*( func(f_c)(a) + func(f_c)(b) )/2 + g_sum;

		 printf("\n------------------  Summary  -----------------\n");
		 printf("Input:\n");
		 printf("=======\n");
		 printf("Number of processes:                      %d\n", num_procs);
		 printf("Integrated func choice:                 [%c]\n", f_c);
		 printf("Integral range:                 [%.2f, %.2f]\n", a, b);
		 printf("Number of Intervals:                      %d\n", N);
		 printf("\nOutput:\n");
		 printf("=======\n");
		 printf("Integratal func[%c] yields:             %.6f\n", g_sum);
		 printf("Elapsed time in communication:           %ld   usec\n", 
				  timeval2microsec(comm_stop) - timeval2microsec(comm_start) );
		 printf("Elapsed time in calculation:             %ld   usec\n", 
				  timeval2microsec(calc_stop) - timeval2microsec(calc_start) );
		 printf("Total elapsed time (approx):             %ld   usec\n", 
				  timeval2microsec(comm_stop) - timeval2microsec(calc_start) );
		 printf("================================================\n\n" );
	}

	MPI_Finalize();		
	return 0;	
} 

///////////////////////////////////END OF FILE///////////////////////////////////
