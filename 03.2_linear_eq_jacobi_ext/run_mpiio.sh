#!/bin/bash

# ################################################################
# Assignment: 3
# Task:       2
# Author:     Pattreeya Tanisaro <pattreeya@gmail.com>
#
# ############################################################### 

# Number of processors
procs=(1 2 4 8)

# select communication methods
comm_methods=(a  b)

echo "======================= script start =========================="
for (( k = 0; k < ${#comm_methods[@]}; k++))
  do
  for (( j = 0; j < ${#procs[@]}; j++))
    do
     # first loop the communication methods
     echo ""
     mpiexec -np ${procs[$j]} -f hosts ./mpiio Matrix_A_8x8 Vector_b_8x 0.001 out_${comm_methods[$k]}_np${procs[$j]}.dat  ${comm_methods[$k]}
     echo ""
  done
done

echo ""
echo "======================= script done =========================="

##########################END OF FILE###############################
