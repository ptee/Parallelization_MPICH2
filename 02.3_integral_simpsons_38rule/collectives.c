/* *****************************************************************************
 * Program:    collectives.c                                                  
 * Author:     Pattreeya Tanisaro                    
 * Description:  compute integral using Simpson 3/8 rule using collective communication
 * ===========   
 *
 * Parameters:
 * ==========
 * Input: argv[1] => function a or b 
 *	  argv[2] => numbers of interval           
 *        argv[3] => lower bound
 *        argv[4] => upper bound
 *        argv[5] => using MPI_Reduce() to compute the result -> "Reduce"
 *                =>       MPI_Alltoall() to compute the result -> "Alltoall"
 *        argv[6] => optional to select show communication and calculation time 
 *                   for each rank. 
 *                   If this parameter not specified, set to 'n' (no), 
 *                   else any param inputs -> 'y' (yes)
 *
 * Output: display => the integral of func f(x) in the the limit of [a,b]
 *                    with N interval.
 *                 => calculation and communication cost
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
#include "assign2.h"                    // common functions for assignment 2

#define DEBUG_COMM      0               // to test small set of data
#define DEBUG_CALC      0               // to test small set of data
#define COMM_SEL_NUM    2               // number of communication types support by this programm
#define MAX_PROCESSES   2048            // max number of processors
#define NBUF            2               // to store lower and upper limit for each subinterval
#define TIMED_COUNT     3               // number of timed count

// --------------------------------------------------------------------------------
// collective communication operations 
// Using a) MPI_Reduce or b) MPI_Alltoall to send the result for complete integral
// --------------------------------------------------------------------------------
char g_comm_ops[COMM_SEL_NUM][32] = {"Reduce", "Alltoall"};



/* **************************************************************
 * @func: simpson38
 * @desc: integral using Simpson 3/8's rule
 * @params:
 * ======== 
 * a     [in] => double as lower limit
 * h     [in] => stepsize
 * start [in] => start of subinterval
 * end   [in] => end of subinterval
 * ptFunc[in] => function pointer to function we want to calculate
 *
 * @return: result of integral or sum as double
 *
 * ************************************************************ */ 

double simpson38(double a, double h,  int start, int end, double (*ptFunc)(double))
{
    double sum = 0;
    int   k;
    
    for ( k = start; k <= end; ++k ) {
        sum +=   (*ptFunc)(a +(3*k-3)*h) +  3*(*ptFunc)(a+ (3*k-2)*h) + 
               3*(*ptFunc)(a +(3*k-1)*h) +    (*ptFunc)(a+ 3*k*h);
    }
    
    return 3*h*sum/8;
}


/* **************************************************************
 * @func: check_comm_input
 * @desc: check the input if communication selector in our choices
 *  
 * @params:
 * ======== 
 * f_comm [in] => input argument which specifiy the type of comm operations
 *
 * @return: the index of the communication type as int
 *          "Reduce" return 0, "Alltoall" return 1, if not found = -1
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
 * @func: alltoall
 * @desc: Use MPI_Alltoallto send all results of the computations
 *        to all processes and complete integral
 * @params:
 * ======== 
 * my_rank   [in] => current rank
 * send_ata  [in] => buffer 
 * num_procs [in] => number of process
 *
 * @return: sum from all processes
 *          
 * ************************************************************ */ 

double alltoall(int my_rank, double* send_ata, int num_procs )
{
    double g_sum = 0;                                  // all sum
    int    i;                                          // running variable
    double recv_ata[MAX_PROCESSES];                    // recv buf for Alltoall

    MPI_Alltoall( &send_ata[0], 1, MPI_DOUBLE, 
                  &recv_ata[0], 1, MPI_DOUBLE, MPI_COMM_WORLD );

    for (i=0; i < num_procs; ++i) {
        g_sum += recv_ata[i];
        #if DEBUG_CALC
        printf("[%d]  sum[%.7f]\n",i, recv_ata[i]);
        #endif
    }

    return g_sum;
}      

/* **************************************************************
 * @func: subinterval
 * @desc: calculate start and end interval index for each rank
 * @params:
 * ======== 
 * N         [in] => number of all intervals
 * num_procs [in] => number of processes
 * sendbuf    [out] => buffer contains start and end time
 *          
 * ************************************************************ */ 

void subinterval(long N, int num_procs, int* sendbuf)
{
    int i,j;
    int step;
    int reminder;

    reminder = N%num_procs;
    step = (int)floor(N/num_procs);
    if (reminder == 0) {
        for (i=0, j=0; i < num_procs*NBUF; i+=NBUF,++j) {
            sendbuf[i]   = step*j + 1;                   // start of subinterval limit 
            sendbuf[i+1] = sendbuf[i] + (step-1);        // end  of subinterval limit  
        }
    }
    else {
        sendbuf[0]  = 1;                                 // start of subinterval limit 
        sendbuf[1]  = sendbuf[0] + step;                 // end  of subinterval limit  
        for (i=2, j=0; i < num_procs*NBUF; i+=NBUF,++j) {
            --reminder;
            sendbuf[i]   = sendbuf[i-1] + 1;             
            if ( reminder > 0 )
                sendbuf[i+1] = sendbuf[i] + step;        
            else
                sendbuf[i+1] = sendbuf[i] + step-1;      
        }        
    }

#if DEBUG_CALC
    for (i=0; i < num_procs*NBUF; i+=NBUF) {
        printf("[%d]  start:[%d]  end:[%d]\n",i, sendbuf[i], sendbuf[i+1]);
    }
#endif
    
}
/* **************************************************************
 * @func: main
 * @desc: main
 * @params:
 * ======== 
 *	Input: argv[1] => a char to select a function to perform the integratral
 *                   'a' to calc 1/x, 'b' to calc pow(sin(x), 2) and so on.
 *	       argv[2] => numbers of interval           
 *        argv[3] => lower bound
 *        argv[4] => upper bound
 *        argv[5] => using MPI_Reduce() to compute the result -> "Reduce"
 *                =>       MPI_Alltoall() to compute the result -> "Alltoall"
 *        argv[6] => optional to select show communication and calculation time 
 *                   for each rank. 
 *                   If this parameter not specified, set to 'n' (no), 
 *                   else any param inputs -> 'y' (yes)
 *
 * ************************************************************ */ 

int main(int argc, char* argv[ ]) 
{ 

    int    my_rank; 				       // rank of the process
    int    num_procs;                                  // number of processes
    int    i,j;                                        // misc variables
    bool   ok;                                         // to verify the return value
    double sum, g_sum=0;                               // sum of calculation
    double a, b;                                       // lower bound, upper bound
    long   N;                                          // Intervals  
    double h;                                          // equidistant h = (b-a)/N
    char   f_c;                                        // input argument as function selection
    char   f_comm[MAX_VERY_SMALL_BUFF]={0};            // last argument for communication op
    int    comm_selector;                              // communication op selector
    int    step;                                       // number of steps
    int    sendbuf[NBUF*MAX_PROCESSES];                // send buffer, size = 2 x max_proc
    int    recvbuf[NBUF*MAX_PROCESSES];                // recv buffer, size = 2 x max_proc
    struct timeval time_start, time_stop;              // start and end time of calculation
    uint   timed[TIMED_COUNT];                         // timed: 1-scatter 2-calc 3-alltotall
    uint   *rtimed;                                    // receivd: 1-scatter 2-calc 3-alltotall
    double send_ata[MAX_PROCESSES]={0};                // send buf for Alltoall
    uint   avg_calc_time=0, avg_comm_time=0;           // average calculation and communication time
    char   show_time_debug;                            // show time debug? default for argc = 6->"no"

    MPI_Init(&argc, &argv);		 		                        
    MPI_Comm_size( MPI_COMM_WORLD, &num_procs );             
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank); 	               

    // ==============================================================
    // check valid inputs (passing arguments)
    // ==============================================================

    if ( argc == 6 || argc == 7) {
        f_c = argv[1][0];
	N  = atol(argv[2]);
	a  = atof(argv[3]); 
	b  = atof(argv[4]);
        strncpy(f_comm, argv[5], strlen(argv[5])); 
	ok = check_valid_input(f_c, N, a, b, num_procs);
        comm_selector = check_comm_input(f_comm);
        ok = ok && ( comm_selector != -1);
        if ( !ok ) {
	    if ( my_rank == ROOT_NODE ) {
	        printf("\nInput error: One of your given might be wrong!\n\n");
	    }
            MPI_Finalize();
            return 0;
	  }
          // additional parameter (not mandatory)
          // if argument 7 exists, set show time 'y'/yes, if not specify -> 'n'/no
          show_time_debug = (argc == 7)? 'y':'n'; 
      }
      else {
          if ( my_rank == ROOT_NODE ) {
		print_usage( argv[0] );
	   }
	   MPI_Finalize();		
	   return 0;
    }

    // ==============================================================
    // calculate subinterval limits in ROOT_NODE
    // ==============================================================
    h = (b-a)/(3*N);

    if ( my_rank == ROOT_NODE ) {
        subinterval(N, num_procs, sendbuf);
    }

    // ==============================================================
    // MPI_Scatter to send subinterval limits as sendbuf to the other processes
    // ==============================================================
    gettimeofday( &time_start, NULL);

    MPI_Scatter(&sendbuf[0], NBUF, MPI_INT, 
                &recvbuf[0], NBUF, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);

    gettimeofday( &time_stop, NULL);
    timed[0] = (uint)(timeval2microsec(time_stop) - timeval2microsec(time_start));

    // ==============================================================
    // Compute Simpson 3/8's Rule
    // ==============================================================
    gettimeofday( &time_start, NULL);

    sum =  simpson38(a, h,  recvbuf[0], recvbuf[1], func(f_c));   

    gettimeofday( &time_stop, NULL);
    timed[1] = (uint)(timeval2microsec(time_stop) - timeval2microsec(time_start));
   
   #if DEBUG_CALC
    printf("R[%d]  sum = %.7f in range[%d,%d]\n",my_rank, sum,recvbuf[0],recvbuf[1]);
   #endif

    // ==============================================================
    // (a) Using MPI_Reduce, passing argument = Reduce
    // (b) Using MPI_Alltoall, passing argument = Alltoall
    // ==============================================================
    gettimeofday( &time_start, NULL);

    switch( comm_selector )
    {
   	case 0:
       	    MPI_Reduce(&sum, &g_sum, 1, MPI_DOUBLE, MPI_SUM, ROOT_NODE, MPI_COMM_WORLD);
       	break;
   	case 1:
       	// make a copy of sum to all units(blocks) in send_ata to  send out
       	// each copy of sum of other processes
       	    for (i=0; i < num_procs; ++i) {
                send_ata[i] = sum;
             }
       	     g_sum = alltoall( my_rank, send_ata, num_procs);
       	break;
   	default:
       	break;
    }

    gettimeofday( &time_stop, NULL);
    timed[2] = (uint)(timeval2microsec(time_stop) - timeval2microsec(time_start));

   #if DEBUG_CALC
     printf("Collective[%d] sum = %.5f\n",my_rank, g_sum);
   #endif
   #if DEBUG_COMM
     printf("[%d] T0 = %d\n",my_rank, timed[0]);
     printf("[%d] T1 = %d\n",my_rank, timed[1]);
     printf("[%d] T2 = %d\n",my_rank, timed[2]);
   #endif

    // ===================================================================
    // MPI_Gather: All processes send time measurement to ROOT_NODE
    // ===================================================================
   if ( my_rank == ROOT_NODE ) {
       rtimed = (uint *)malloc(TIMED_COUNT*MAX_PROCESSES*sizeof(uint));
   }


   MPI_Gather( &timed[0], TIMED_COUNT, MPI_UNSIGNED,
               rtimed,    TIMED_COUNT, MPI_UNSIGNED, ROOT_NODE, MPI_COMM_WORLD );

    // ===================================================================
    // Print out output from ROOT_NODE
    // ===================================================================
    if ( my_rank == ROOT_NODE ) {

        printf("\nComposite Simpson3/8 using collective communication\n");
       	printf("=====================================================\n");
	printf("\nInput:\n");
	printf("=======\n");
	printf("Number of processes:                      %d\n", num_procs);
	printf("Integrated func choice:                  [%c]\n", f_c);
	printf("Integral range:                      [%.2f, %.2f]\n", a, b);
	printf("Number of Intervals:                      %d\n", N);
        printf("Communication operation:                  %s\n", f_comm);
	printf("\nOutput:\n");
	printf("=======\n");
	printf("Integratal func[%c] yields:                %.6f\n", g_sum);
        printf("\nTimed\n");
      
        for (j=0; j < num_procs; ++j) {
            if ( show_time_debug == 'y') {
                printf("Rank: [%d]\n",j);
                #if DEBUG_COMM
                 printf("MPI_Scatter takes      :                  %d usec\n", rtimed[TIMED_COUNT*j]);
                 printf("Sends all results takes:                  %d usec\n", rtimed[TIMED_COUNT*j+2]);
                #endif
                printf("Total communication t  :                  %d usec\n", 
                      	rtimed[TIMED_COUNT*j+2]+rtimed[TIMED_COUNT*j]);
                printf("Total calculation t    :                  %d usec\n", rtimed[TIMED_COUNT*j+1]);
           }
           avg_calc_time += rtimed[TIMED_COUNT*j+1];
           avg_comm_time += rtimed[TIMED_COUNT*j] + rtimed[TIMED_COUNT*j+2];
       }
       printf("\n=======\n");
       printf("Average communication time :               %d usec\n",
              (uint)floor(avg_comm_time/num_procs));
       printf("Average calculation time   :               %d usec\n", 
              (uint)floor(avg_calc_time/num_procs));
       printf("Average total time         :               %d usec\n",
              (uint)floor(avg_calc_time+avg_comm_time)/num_procs);
       printf("====================================================\n\n" );
    }

 
    if ( my_rank == ROOT_NODE ) {
        free( rtimed );
    }
    MPI_Finalize();		
    return 0;	
} 


///////////////////////////////////END OF FILE///////////////////////////////////
