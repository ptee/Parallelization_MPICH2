/* *****************************************************************************
 * Program:    binminmax.c                                                  
 * Author:     Pattreeya Tanisaro <pattreeya@gmail.com>                    
 * Date:       2.05.2011		              			        
 * Parameters:
 * ==========
 * Input: integer => Any random number as interger ( to be calculated in term 
 *	             of power of 2 in the program.
 *
 * Output: display => the max, min, average of the random set of integers,
 *                 => calculation and communication cost
 *                                                                   
 * Description:  To find the maximum, minimum, average of a random set of intergers
 * ===========   and the cost of calculation and communication.
 *	
 * *************************************************************************** */ 


#include "mpi.h" 	            // import of the MPI definitions
#include <stdio.h> 	            // import of the definitions of the C IO library
#include <string.h>                 // import of the definitions of the string operations
#include <unistd.h>	            // standard unix io library definitions and declarations
#include <errno.h>	            // system error numbers
#include <sys/time.h>	            // speicial system time functions for c
#include <stdlib.h>                 // avoid warning from malloc/free
#include <math.h>                   // for log & pow
#include <limits.h>                 // for INT_MIN, INT_MAX
#include "commonc.h"                // self-defined common c utility func


#define TEST_DEBUG      0           // to test small set of data, n should be less than 8
#define MY_RAND_MAX     10000       // set max value for random number instead of default


/* **************************************************************
 * @func: calculation
 * @desc: algorithm to find min, max and sum of all values
 * @params:  [in] my_rank: rank of the process used to generate 
 *                         the seed on same machine.
 *           ***  The seed with time alone does not work properly 
 *                with multiple of nodes on same machine!!
 *                Therefore we need to generate random values using
 *                current time [on machine] combine with rank.
 *           ***    
 *           [in] dsize:   numbers of random numbers for
 *                         each process
 *           [out] min:    minimum
 *           [out] max:    maximum
 *           [out] sum:    sum of all numbers
 *           [out] buf:    buffer used to debug
 * @return: true if no error
 *          false if error found, the calling func will handle the exit
 *
 * ************************************************************ */ 

bool calculation(int my_rank, int dsize, int* min, int* max, ulong* sum
                 #if TEST_DEBUG
		    , char* buf 
                 #endif
                )
{
  #if TEST_DEBUG
	 char   data[MAX_SMALL_MESG_BUFF+1];              // temporary buffer
	 char   _buf[MAX_MESG_BUFF+1] = {0};              // buffer for string output
  #endif
	 int    *numbers;                                 // rand numbers in each process
	 int    i, _min, _max;                            // temporary variables
	 ulong  _sum;                                     // sum of output

	// each process will have random numbers with dsize
	numbers =  (int *)  malloc( dsize * sizeof(int) );
	if ( numbers == NULL ) {
		 return false;
	}
	memset(numbers, 0, dsize);
	srand( time(NULL) + my_rank );    
	_min = INT_MAX; _max = INT_MIN; _sum = 0;
	for ( i=0; i < dsize; ++i ) {
		 numbers[i]  =  rand() % MY_RAND_MAX + 1;  
		 _sum  +=  numbers[i];
		 _min   =  MIN(numbers[i], _min);
		 _max   =  MAX(numbers[i], _max);
  #if TEST_DEBUG
		 sprintf( data, "[%d]", numbers[i] );
		 if ( strnlen(_buf) <  MAX_MESG_BUFF - strnlen(data)- 1 ) // to prevent buffer overflow
			  strcat( _buf, data );                               
  #endif
	}

	free( numbers );
	*sum = _sum; *min = _min; *max = _max;
  #if TEST_DEBUG
	strcpy( buf, _buf );
  #endif

	return true;
}

/* **************************************************************
 * @func: main
 * @desc: main
 * @params:  [in] my_rank: rank of the process
 *           [in] dsize:   numbers of random numbers for
 *                         each process
 *           [out] min:    minimum
 *           [out] max:    maximum
 *           [out] sum:    sum of all numbers
 *           [out] buf:    buffer used to debug
 *
 * ************************************************************ */ 

int main(int argc, char* argv[ ]) 
{ 

	 int    my_rank; 					   // rank of the process
	 int    num_procs;                                         // number of processes
	 char   *c, proc_name[MPI_MAX_PROCESSOR_NAME+1] = {0}; 	   // hostname  
	 char   buf[MAX_MESG_BUFF+1] = {0};                        // Debug buffer
	 int    dsize;                                     // each process's data(random number) size
	 int    i, j, k;                                   // misc variables
	 bool   ok;                                        // to verify the return value
	 int    d;                                         // depth of the binary tree
	 ulong  sum, tmp_sum;                              // sum as output 
	 int    min, max;                                  // min, max as output
	 int    g_min = INT_MAX, g_max = INT_MIN;          // global min, global max
	 int    _min = INT_MAX, _max = INT_MIN;            // tempoary min, max
	 int    rand_num;                                  // random number 2^m, as m given from the prog
	 MPI_Status status;                                // received status
	 struct timeval calc_start, calc_stop;             // start and end time of calculation
	 struct timeval comm_start, comm_stop;             // start and end time of communication
	 int   sum_tag = 0, min_tag = 10, max_tag = 20, buf_tag = 50; // tag for communications
	 int   partner;                                    // partner of this rank
	 
	MPI_Init(&argc, &argv);		 		                        
	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );             
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

	if ( (c=strchr(proc_name,'.'))  !=  NULL) *c = '\0'; 	   

	// ==============================================================
	// check for first argument as random number given to the program
	// ==============================================================
	if ( argc == 2) {
		 rand_num  =  pow( 2, atoi(argv[1]) );	
		 if ( rand_num%num_procs != 0 ) {
			  printf("\n\nErr: Number of random number cannot be divided equally by number of processes!");
			  printf("\n   Number of random number: %d, number of process: %d!\n\n", rand_num, num_procs);
			  MPI_Finalize();		
			  exit(1);
		 }
	}
	else {
		 printf("\n\nErr:%s must follow by a random number which will be powered of 2\n\n",argv[0]);
		 MPI_Finalize();		
		 exit(1);
	}

	// slide the random number equally for each process
	dsize = rand_num/num_procs;
	
	// ==============================================================
	// To calculate min, max, sum the value of the process & make time
	// ==============================================================

	// start timer for calculation
	MPI_Barrier( MPI_COMM_WORLD );
	gettimeofday( &calc_start, NULL );
	
	ok = calculation( my_rank, dsize, &min, &max, &sum
              #if TEST_DEBUG
				  , buf
              #endif
		         );
	if ( ! ok ) {
		 MPI_Finalize();		
		 exit(1);
	}

	
	// stop timer for calculation
	MPI_Barrier( MPI_COMM_WORLD );
	gettimeofday( &calc_stop, NULL);

	// ==============================================================
	// To calculate the binary tree => [partner] = [me] xor [1<<d-1]
	// while d is the depth of the tree .
	// =============================================================

	// start timer for communication
	gettimeofday( &comm_start, NULL);

   #if TEST_DEBUG
	printf("my_rank: %d, sum: %ld, min: %d, max:%d\n", my_rank, sum, min, max);
   #endif

	d = (int)(log(num_procs)/log(2));
	for ( j = d-1 ; j >= 0; --j )               // each b-tree depth, we reduce the communication
	{     
		 if ( my_rank < pow(2,(j+1)) ) {         // filter for each depth to prevent the re-send of nodes

			  partner = my_rank^(1<<(j));         // calculate the partner of this node in b-tree

			  #if TEST_DEBUG
			    printf("[%d] my_rank: %d , partner = %d\n",j, my_rank, partner);
			  #endif

				 if ( (my_rank & partner) == my_rank ) { 
					  #if TEST_DEBUG
					  printf("[recv] my_rank: %d received from: %d\n", my_rank, partner);
					  #endif
					  MPI_Recv(&tmp_sum, 1, MPI_UNSIGNED_LONG, partner,  sum_tag,  MPI_COMM_WORLD, &status);
					  sum += tmp_sum;
					  MPI_Recv(&_min,   1, MPI_INT, partner, min_tag,  MPI_COMM_WORLD, &status);
					  g_min = MIN( _min, g_min);
					  MPI_Recv(&_max,   1, MPI_INT, partner, max_tag,  MPI_COMM_WORLD, &status);
					  g_max = MAX( g_max, _max);
					  
			  }
			  else {
               #if TEST_DEBUG
					printf("[send] my_rank: %d sends to: %d, sum = %ld\n", my_rank, partner, sum);
               #endif
					MPI_Send(&sum, 1, MPI_UNSIGNED_LONG, partner ,  sum_tag,  MPI_COMM_WORLD); 
					min = MIN( g_min, min );
					MPI_Send(&min, 1, MPI_INT, partner, min_tag,  MPI_COMM_WORLD); 
					max = MAX( g_max, max );
					MPI_Send(&max, 1, MPI_INT, partner, max_tag,  MPI_COMM_WORLD); 

			  }
		 } // end if
	} // end for

	
	// stop timer for communication
	MPI_Barrier( MPI_COMM_WORLD );
	gettimeofday( &comm_stop, NULL);

	
	// ============================
	// DEBUG output
	// ============================
#if TEST_DEBUG 
	if ( my_rank == 0 ) {
		 printf("from %d -> %s\n", my_rank, buf); 
		 for ( i = 1; i < num_procs; ++i ) {
			  MPI_Recv( &buf, MAX_MESG_BUFF+1, MPI_CHAR, i, buf_tag,  MPI_COMM_WORLD, &status); 
			  printf("from %d -> %s\n", i, buf); 
		 }
	}
	else {
		 MPI_Send( &buf, MAX_MESG_BUFF+1, MPI_CHAR, 0, buf_tag,  MPI_COMM_WORLD);
		 
	}
#endif


	// ======================
	// print output summary
	// ======================
	if ( my_rank == 0 ) {

		 printf("\n------------------  Summary  -----------------\n");
		 printf("Input:\n");
		 printf("=======\n");
		 printf("Total random numbers:          %d\n",rand_num);
		 printf("Number of processes:           %d\n",num_procs);
		 printf("Random numbers on one process: %d\n", dsize);
		 printf("Number values in range:        [0,%d]\n", MY_RAND_MAX);
		 printf("\nOutput:\n");
		 printf("=======\n");
		 printf("Sum from all processes:        %ld\n", sum);
		 printf("Average from all processes:    %.4lf\n", (double)sum/rand_num);
		 printf("Global maximum:                %d\n", g_max );
		 printf("Global minimum:                %d\n", g_min );
		 printf("Elapsed time in communication: %ld   usec\n", 
				  timeval2microsec(comm_stop) - timeval2microsec(comm_start) );
		 printf("Elapsed time in calculation:   %ld   usec\n", 
				  timeval2microsec(calc_stop) - timeval2microsec(calc_start) );
		 printf("Total elapsed time (approx):   %ld   usec\n", 
				  timeval2microsec(comm_stop) - timeval2microsec(calc_start) );
		 printf("================================================\n\n" );
	}

	MPI_Finalize();		
	return 0;	
} 

///////////////////////////////////END OF FILE///////////////////////////////////
