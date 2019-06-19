
/* *****************************************************************************
 * Program:     butterfly.c                                                  
 * Author:      Pattreeya Tanisaro                
 * Description: Using Butterfly algoritm as point-to-point communication 
 *              for blocking and non-blocking send and receive.
 *              Similar to Alltoall communication to send partial output of integral calculation.
 *              Here, we using Composite Trapezoidal Rule to compute the integral
 * ===========  then measure the cost for sequential, parallel and communication components.
 *
 * Parameters:
 * ==========
 * Input: argv[1] => a character to function selection
 *	      argv[2] => numbers of interval           
 *        argv[3] => lower bound
 *        argv[4] => upper bound
 *        argv[5] => communication type
 *
 * Output: display => the integral of func f(x) in the the limit of [a,b]
 *                    with N interval.
 *                 => sequential and parallel and communication cost
 *	
 * *************************************************************************** */ 

#include "mpi.h" 	                // import of the MPI definitions
#include <stdio.h> 	                // import of the definitions of the C IO library
#include <string.h>                     // import of the definitions of the string operations
#include <unistd.h>	                // standard unix io library definitions and declarations
#include <errno.h>	                // system error numbers
#include <sys/time.h>	                // speicial system time functions for c
#include <stdlib.h>                     // avoid warning from malloc/free
#include <math.h>                       // for log & pow
#include <limits.h>                     // for INT_MIN, INT_MAX
#include "commonc.h"                    // self-defined common c utility func
#include "assign2.h"                    // common functions for assignment 2

#define TEST_DEBUG          0           // to test the algorithms

#define ARGS_NUM            6           // argument number
#define COMM_SEL_NUM        6           // number of communication types support by this programm
#define BSEND_BUF_SIZE   1024           // size of the buffer for bsend as num of elements
#define REQ_NUM           256           // number of requests 

// ---------------------------------------------------------------------
// global communication operations => Blocking & Non-blocking -> 6 types
// ---------------------------------------------------------------------
char g_comm_ops[COMM_SEL_NUM][32] = {"Send", "Bsend", "Ssend", "Isend", "Ibsend", "Issend"};


/* **************************************************************
 * @func: print_usage_6
 * @desc: print the usage and example for calling butterfly
 *  
 * @params:
 * ======== 
 * prog [in] => name of the executable
 * @return: none
 *
 * ************************************************************ */ 

void print_usage_6(char *prog)
{
	 printf("\nComposite Trapezoidal Rule to integrate function f(x)\n");
	 printf("by using various communication operations\n");
	 printf("----------------------------\n");
	 printf("Usage: %s [f(x)] [num of interval] [lower bound] [upper bound] [comm_op]\n",prog );
	 printf("[f(x)] => a for f(x) =  pow( 2, sin(x) )\n");
	 printf("[f(x)] => b for f(x) =  1/(x)\n");
	 printf("[f(x)] => c...z for f(x) is not yet defined\n");
	 printf("[comm_op] => Send   for Standard blocking");
	 printf("[comm_op] => Ssend  for Synchronous blocking");
	 printf("[comm_op] => Bsend  for Buffered blocking");
	 printf("[comm_op] => Isend  for Standard non-blocking");
	 printf("[comm_op] => Issend for Sychronous non-block");
	 printf("[comm_op] => Ibend  for buffered non-block");
	 printf("Exmaples: %s  a  1000000  0.1  10000 Send\n\n\n",prog);
}


/* **************************************************************
 * @func: Send
 * @desc: Blocking Standard send and recevie operations
 *  
 * @params:
 * ======== 
 * d   [in] => number of depth of the communication equal to log(p)/log(2)
 * sum [in] => integral of the current process
 * @return: double as the overall sum from all processes
 *
 * ************************************************************ */ 

double Send(int d, double sum)
{
    int    j, partner, my_rank;              // ranks
    int    sum_tag = 0;	                     // tag
    double recv_sum;                         // temporary receving sum
    MPI_Status status;                       // received status
    int    bit_mask;                         // bit mask to calculate the partner to send/recv

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

    // butterfly network
    for ( j = d-1 ; j >= 0; --j )            // each comm depth 
    {     
        bit_mask = 1<<j;
		partner  = my_rank^(bit_mask);        // calculate the partner of this node as in b-tree

        if ( my_rank & bit_mask ) {
            MPI_Send(&sum,      1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD); 
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &status);
        }
        else {
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, partner, sum_tag, MPI_COMM_WORLD, &status);
            MPI_Ssend(&sum,     1, MPI_DOUBLE, partner, sum_tag, MPI_COMM_WORLD); 

        }
        sum += recv_sum;		 

        #if TEST_DEBUG
		printf("[%d] rank[%d] total sum %.6f\n", j, my_rank, sum); 
        #endif
    }

    return sum;
}


/* **************************************************************
 * @func: Ssend
 * @desc: Blocking Synchronous send and recevie operations
 *  
 * @params:
 * ======== 
 * d   [in] => number of depth of the communication equal to log(p)/log(2)
 * sum [in] => integral of the current process
 * @return: double as the overall sum from all processes
 *
 * ************************************************************ */ 

double Ssend(int d, double sum)
{
    int    j, partner, my_rank;              // ranks
    int    sum_tag = 0;	                     // tag
    double recv_sum;                         // temporary receving sum
    MPI_Status status;                       // received status
    int    bit_mask;                         // bit mask to calculate the partner


    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

	 // butterfly network
    for ( j = d-1 ; j >= 0; --j )            // each comm depth 
    {     
        bit_mask = 1<<j;
		partner  = my_rank^bit_mask;         // calculate the partner of this node as in b-tree

        // prevent dead-lock by obviously masking the sender & receiver
        if ( my_rank & bit_mask )
        { 
            MPI_Ssend(&sum,     1, MPI_DOUBLE, partner, sum_tag, MPI_COMM_WORLD); 
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, partner, sum_tag, MPI_COMM_WORLD, &status);
        }
        else
        {
            MPI_Recv(&recv_sum, 1, MPI_DOUBLE, partner, sum_tag, MPI_COMM_WORLD, &status);
            MPI_Ssend(&sum,     1, MPI_DOUBLE, partner, sum_tag, MPI_COMM_WORLD); 
        }
                
		sum += recv_sum;

        #if TEST_DEBUG
		printf("[%d] rank[%d] total sum %.6f\n", j, my_rank, sum); 
        #endif
    }

    return sum;
}


/* **************************************************************
 * @func: Bsend
 * @desc: Blocking Buffered send and recevie operations
 *  
 * @params:
 * ======== 
 * d   [in] => number of depth of the communication equal to log(p)/log(2)
 * sum [in] => integral of the current process
 * @return: double as the overall sum from all processes
 *
 * ************************************************************ */ 

double Bsend(int d, double sum)
{
    int    j, partner, my_rank;              // ranks
    int    sum_tag = 0;	                     // tag
    double recv_sum;                         // temporary receving sum
    MPI_Status status;                       // received status
    int    bsize = MPI_BSEND_OVERHEAD + sizeof(int)*BSEND_BUF_SIZE; // buffer size, make big to avoid overflow
    int    buff[bsize];                      // buffer for Bsend attach
    int    bit_mask;                         // bit_mask

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

    // attach the buffer
    MPI_Buffer_attach( buff, bsize);

    // butterfly network
    for ( j = d-1 ; j >= 0; --j )            // each comm depth 
    {     
        bit_mask = 1<<j;
        partner  = my_rank^(bit_mask);       // calculate the partner of this node as in b-tree
		
        MPI_Bsend(&sum,  1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD); 
		MPI_Recv(&recv_sum, 1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &status);
		sum += recv_sum;

        #if TEST_DEBUG
		printf("[%d] rank[%d] total sum %.6f\n", j, my_rank, sum); 
        #endif
    }

    // detach the buffer
    MPI_Buffer_detach( &(*buff), &bsize );

    return sum;
}


/* **************************************************************
 * @func: Isend
 * @desc: Non-Blocking Standard send and recevie operations
 *  
 * @params:
 * ======== 
 * d   [in] => number of depth of the communication equal to log(p)/log(2)
 * sum [in] => integral of the current process
 * @return: double as the overall sum from all processes
 *
 * ************************************************************ */ 

double Isend(int d, double sum)
{
    int    j, partner, my_rank;              // ranks
    int    sum_tag = 0;	                     // tag
    double recv_sum;                         // temporary receving sum
    MPI_Status  status[REQ_NUM];             // received status
    MPI_Request request[REQ_NUM];            // requests for wait as array
    int   bit_mask;                          // bit_mask

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

    // butterfly network
    for ( j = d-1 ; j >= 0; --j )            // each comm depth 
    {     
        bit_mask = 1<<j;
        partner = my_rank^bit_mask;          // calculate the partner of this node as in b-tree
		  
        
        // initialize ...and return 
        // my rank sends with req0, partner will recevie with req1 
        MPI_Isend(&sum,      1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &request[0]); 
        MPI_Irecv(&recv_sum, 1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &request[1]);

        // do other work if any...
        
        // filter to wait for completion; the order of wait must be in order
        // somhow similar to when we mask the b-tree..to select partner or my_rank's work
        if ( my_rank & bit_mask ) 
        {
            MPI_Wait( &request[0], &status[0] );
            MPI_Wait( &request[1], &status[1] );
        }
        else // swap oder of the req
        {
            MPI_Wait( &request[1], &status[1] );
            MPI_Wait( &request[0], &status[0] );
        }

        sum += recv_sum;
		  
        #if TEST_DEBUG
        printf("[%d] rank[%d] total sum %.6f\n", j, my_rank, sum); 
        #endif
    }

    return sum;
}


/* **************************************************************
 * @func: Issend
 * @desc: Non-Blocking Synchronous send and recevie operations
 *  
 * @params:
 * ======== 
 * d   [in] => number of depth of the communication equal to log(p)/log(2)
 * sum [in] => integral of the current process
 * @return: double as the overall sum from all processes
 *
 * ************************************************************ */ 

double Issend(int d, double sum)
{
    int    j, partner, my_rank;              // ranks
    int    sum_tag = 0;	                     // tag
    double recv_sum;                         // temporary receving sum
    MPI_Status  status[REQ_NUM];             // received status
    MPI_Request request[REQ_NUM];            // requests for wait as array
    int   bit_mask;                          // bit_mask


    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

    // butterfly network
    for ( j = d-1 ; j >= 0; --j )            // each comm depth 
    {     
        bit_mask = 1<<j;
		partner = my_rank^bit_mask;          // calculate the partner of this node as in b-tree
		  
        
        // initialize ...
		MPI_Issend(&sum,     1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &request[0]); 
		MPI_Irecv(&recv_sum, 1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &request[1]);

        // do work if any

        
        // filter to wait for completion; the order of wait must be in order
        // somhow similar to when we mask the b-tree..to select partner or my_rank's work

        if ( my_rank & bit_mask ) {
            MPI_Wait( &request[0], &status[0] );
            MPI_Wait( &request[1], &status[1] );
        }
        else // swap oder of the req
        {
            MPI_Wait( &request[1], &status[1] );
            MPI_Wait( &request[0], &status[0] );
        }
		  
        sum += recv_sum;
		  
        #if TEST_DEBUG
		printf("[%d] rank[%d] total sum %.6f\n", j, my_rank, sum); 
        #endif
	 }

    return sum;
}


/* **************************************************************
 * @func: Ibsend
 * @desc: Non-Blocking Buffered send and recevie operations
 *  
 * @params:
 * ======== 
 * d   [in] => number of depth of the communication equal to log(p)/log(2)
 * sum [in] => integral of the current process
 * @return: double as the overall sum from all processes
 * 
 * ************************************************************ */ 

double Ibsend(int d, double sum)
{
    int    j, partner, my_rank;              // ranks
    int    sum_tag = 0;	                     // tag
    double recv_sum;                         // temporary receving sum
    MPI_Status  status[REQ_NUM];             // received status
    MPI_Request request[REQ_NUM];            // requests for wait as array
    int    bit_mask;                         // bit_mask
    int    bsize = MPI_BSEND_OVERHEAD + sizeof(int)*BSEND_BUF_SIZE; // buffer size, make big to avoid overflow
    int    buff[bsize];                      // buffer for Bsend attach
	 

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

    // attach the buffer
    MPI_Buffer_attach( buff, bsize);


    // butterfly network
    for ( j = d-1 ; j >= 0; --j )            // each comm depth 
    {     
        bit_mask = 1<<j;
		partner = my_rank^bit_mask;          // calculate the partner of this node as in b-tree
        
        // initialize ...
		MPI_Ibsend(&sum,     1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &request[0]); 
		MPI_Irecv(&recv_sum, 1, MPI_DOUBLE, partner,  sum_tag,  MPI_COMM_WORLD, &request[1]);

        // do work

        
        // filter to wait for completion; the order of wait must be in order
        // somehow filtering similar to when we mask the b-tree..to select partner or my_rank's work
        if ( my_rank & bit_mask ) {
            MPI_Wait( &request[0], &status[0] );
            MPI_Wait( &request[1], &status[1] );
        }
        else // we swap the order of wait...
        {
            MPI_Wait( &request[1], &status[1] );
            MPI_Wait( &request[0], &status[0] );
        }

        sum += recv_sum;

        #if TEST_DEBUG
		printf("[%d] rank[%d] total sum %.6f\n", j, my_rank, sum); 
        #endif

    }
    
    // detach the buffer
    MPI_Buffer_detach( &(*buff), &bsize );

    return sum;
}



/* **************************************************************
 * @func: check_comm_input
 * @desc: check the input if communication selector in our choices
 *  
 * @params:
 * ======== 
 * f_comm [in] => input argument which specifiy the type of comm operations
 *
 * @return: int as the index of the communication type
 *
 * ************************************************************ */ 

int check_comm_input(char* f_comm)
{
    int  i, found = -1;
    for( i=0; i< COMM_SEL_NUM; ++i ) {
        // found matching string, we return the index to the communication type
		if ( !strcmp(f_comm, g_comm_ops[i]) ) { 
		found = i;
		break;
		}
    }

    return found;
}

/* **************************************************************
 * @func: main
 * @desc: main
 * @params:
 * ======== 
 * Input: argv[1] => function to integrate in term of c function! e.g.
 *                   1/x, pow(sin(x), 2) and so on.
 *	  argv[2] => numbers of interval           
 *        argv[3] => lower bound
 *        argv[4] => upper bound
 *        argv[5] => communication type
 *
 * ************************************************************ */ 

int main(int argc, char* argv[]) 
{ 
    int    my_rank; 					                // rank of the process
    int    num_procs;                                  // number of processes
    int    d;                                          // depth of the structure in butterfly
    bool   ok;                                         // to verify the return value
    double sum = 0;                                    // sum of (integral) calculation
    struct timeval calc_start, calc_stop;              // start and end time of calculation
    struct timeval comm_start, comm_stop;              // start and end time of communication
	double a, b;                                       // lower bound, upper bound
	int    N;                                          // Intervals  
	double h;                                          // equidistant h = (b-a)/N
	char   f_c;                                        // input argument as function selection
	char   f_comm[MAX_VERY_SMALL_BUFF]={0};            // last argument for communication op
	int    comm_selector;                              // communication op selector
	 
	MPI_Init(&argc, &argv);		 		                        
	MPI_Comm_size( MPI_COMM_WORLD, &num_procs );             
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	               

	// ==============================================================
	// check valid inputs (passing arguments)
	// ==============================================================

	 if ( argc == ARGS_NUM ) {
		  f_c = argv[1][0];
		  N  = atoi(argv[2]);
		  a  = atof(argv[3]); 
		  b  = atof(argv[4]);
		  strncpy(f_comm, argv[5], strlen(argv[5])); 
		  ok = check_valid_input(f_c, N, a, b, num_procs);
		  comm_selector = check_comm_input(f_comm);
		  ok = ok && ( comm_selector != -1);
		  if ( !ok ) {
				if ( my_rank == ROOT_NODE ) {
					 printf("\nError: Your inputs are invalid! Check your inputs again!\n\n");
				}
				MPI_Finalize();		
				exit(1);
		  }
	 }
	 else {
		  if ( my_rank == ROOT_NODE ) {
				print_usage_6( argv[0] );
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
	 //
	 // To send partial sum from each nodes to all other nodes similar to all-to-all
	 // communication, this algorithm is called "Butterfly" algorithm.
	 // 
	 // ==============================================================
	
	 // start timer for communication
	 gettimeofday( &comm_start, NULL);

	 d = (int)(log(num_procs)/log(2));           // the depth of the Butterfly network
	
    #if TEST_DEBUG
	 printf("rank[%d] initial sum %.6f\n", my_rank, sum); 
    #endif

	 switch (comm_selector)
	 {
    	case 1:
        	sum = Bsend(d, sum); 
        	break;
    	case 2:
        	sum = Ssend(d, sum); 
        	break;
    	case 3:
        	sum = Isend(d, sum); 
        	break;
    	case 4:
        	sum = Ibsend(d, sum); 
        	break;
    	case 5:
        	sum = Issend(d, sum); 
        	break;
    	default:
        	sum = Send(d, sum); 
	 }
         
	 // stop timer for communication
	 MPI_Barrier( MPI_COMM_WORLD );
	 gettimeofday( &comm_stop, NULL);

	 // ===================================================================
	 // Calculate the final output for the integral & print output summary
	 // 
	 //     S(f,h) = h*( f(a) + f(b) )/2 + partial_sum_from_all_nodes
	 //
	 // ===================================================================

	 sum = ((b-a)/N)*( func(f_c)(a) + func(f_c)(b) )/2 + sum;
	
     #if TEST_DEBUG
	 printf("rank[%d] total sum %.6f\n", my_rank, sum); 
    #endif

	 if ( my_rank == ROOT_NODE ) {

		  printf("\n------------------  Summary  -----------------\n");
		  printf("Input:\n");
		  printf("=======\n");
		  printf("Number of processes:                      %d\n", num_procs);
		  printf("Integrated func:                         [%c]\n", f_c);
		  printf("Number of Intervals:                      %d\n", N);
		  printf("Integral range:                  [%.2f, %.2f]\n", a, b);
		  printf("P-t-P communication:                      %s\n", f_comm);	 
		  printf("\nOutput:\n");
		  printf("=======\n");
		  printf("Integratal func[%c] yields:             %.6f\n", f_c, sum); 
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
