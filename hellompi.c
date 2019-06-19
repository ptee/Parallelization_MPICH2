/* ***********************************************************************
 * Program: hellompi.c                                                  
 * Author:  Pattreeya Tanisaro                    
 * Task:    Test the MPICH2 environment                                                           
 * Parameters: no	                                                
 * Environment variables: no                                            
 *                                                                     
 * Description: hellompi is a simple MPI program which
 *              the root process print the output   
 *              from all other processes.
 *              For each node, where the program is running,	       
 *              the hostname will be printed.                           
 *                                                                      
 * ***********************************************************************/ 

#include "mpi.h" 	      	// import of the MPI definitions
#include <stdio.h> 	        // import of the definitions of the C IO library
#include <string.h>         	// import of the definitions of the string operations
#include <unistd.h>	        // standard unix io library definitions and declarations
#include <errno.h>	        // system error numbers
#include <sys/time.h>	    	// speicial system time functions for c

// =====================================================
// root node take the messages from other node to print
// =====================================================

void master()
{
	 int ntasks, rank, tag;
	 char message[MPI_MAX_PROCESSOR_NAME];
	 MPI_Status status;

	 MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
	 for (rank=1; rank < ntasks; ++rank) {
		  tag = 0;
		  MPI_Recv(&message, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_ANY_SOURCE, tag,  MPI_COMM_WORLD, &status);
		  printf("%s\n",message);
	 }
}


// ========================================================================
// each other node other than root will send its hello message to root node
// ========================================================================

void slave(char *proc_name, int rank)
{
	 char send_buf[MPI_MAX_PROCESSOR_NAME+1];

	 sprintf(send_buf,"hello from %03d %s",rank, proc_name);
	 MPI_Send( &send_buf, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, 0,  MPI_COMM_WORLD);

}

// ========================================================================
// @main print out hello message from all nodes
// ========================================================================

int main(int argc, char* argv[ ]) 
{ 

	int namelen;					// length of name
	int my_rank; 					// rank of the process
	char *c, proc_name[MPI_MAX_PROCESSOR_NAME+1]; 	// hostname  

	MPI_Init(&argc, &argv);		 		// initializing of MPI-Interface
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 	//get your rank
	MPI_Get_processor_name(proc_name, &namelen);	// finding out own computer name

	if ( (c=strchr(proc_name,'.'))  !=  NULL) *c = '\0';  // separate the first part of hostname

	if (my_rank == 0) {
		 printf("%03d: process runing on %s!\n", my_rank, proc_name );
		 master( );                                
	}
	else {
		 slave( proc_name, my_rank );               
	}


	MPI_Finalize();		                       // finalizing MPI interface 
	return 0;		                       // end of progam with exit code 0 
} 

////////////////////////////////END OF FILE///////////////////////////////////
