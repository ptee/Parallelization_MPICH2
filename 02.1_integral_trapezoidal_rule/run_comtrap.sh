#!/bin/bash

# ################################################################
# Task:       02.1
# Author:     Pattreeya Tanisaro <pattreeya@gmail.com>
# Inputs:     host file called "hosts" and the executable called "comptrap"
# ############################################################### 

# Number of processors
procs=(1 16 32 128 256)

# Number of steps
N=(1000000 100000000)

echo "======================= script start =========================="
for (( i = 0; i < ${#procs[@]}; i++))
do
   for (( j = 0; j < ${#N[@]}; j++))
   do
	      echo ""
	      mpiexec -np ${procs[$i]} -f hosts ./comtrap a  ${N[$j]} 0.1 10000
			mpiexec -np ${procs[$i]} -f hosts ./comtrap b  ${N[$j]}  10  2000
	      echo ""
   done
done

echo ""
echo "======================= script done =========================="

##########################END OF FILE###############################
