#!/bin/bash

# ################################################################
# Assignment: 2
# Task:       3
# Author:     Pattreeya Tanisaro
#
# ############################################################### 

# Number of processors
procs=(1 4 8 16 64 128)

# Number of steps
N=(1000000 100000000)


echo "======================= script start =========================="
for (( i = 0; i < ${#procs[@]}; i++))
do
   for (( j = 0; j < ${#N[@]}; j++))
   do
	     echo ""
		 mpiexec -np ${procs[$i]} -f hosts ./collectives a ${N[$j]} 0.1 10000 Alltoall
         mpiexec -np ${procs[$i]} -f hosts ./collectives a ${N[$j]} 0.1 10000 Reduce
         mpiexec -np ${procs[$i]} -f hosts ./collectives b ${N[$j]} 10   2000 Alltoall
         mpiexec -np ${procs[$i]} -f hosts ./collectives b ${N[$j]} 10   2000 Reduce
	     echo ""
   done
done

echo ""
echo "======================= script done =========================="

##########################END OF FILE###############################
