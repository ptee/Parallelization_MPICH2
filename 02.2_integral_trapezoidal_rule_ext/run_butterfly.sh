#!/bin/bash

# ################################################################
# Assignment: 2
# Task:       2
# Author:     Pattreeya Tanisaro <pattreeya@gmail.com>
#
# ############################################################### 

# Number of processors
procs=(1 8 16 32 128 256)

# Number of steps
N=(100000000)

# select communication methods
comm_methods=(Send Ssend Bsend Isend Issend Ibsend)

echo "======================= script start =========================="
for (( i = 0; i < ${#procs[@]}; i++))
do
   for (( j = 0; j < ${#N[@]}; j++))
   do
     # first loop the communication methods
     for (( k = 0; k < ${#comm_methods[@]}; k++))
     do
	      echo ""
	      mpiexec -np ${procs[$i]} -f hosts ./butterfly a  ${N[$j]} 0.1 10000 ${comm_methods[$k]}
         mpiexec -np ${procs[$i]} -f hosts ./butterfly b  ${N[$j]}  10  2000 ${comm_methods[$k]}
	      echo ""
      done

   done
done

echo ""
echo "======================= script done =========================="

##########################END OF FILE###############################
