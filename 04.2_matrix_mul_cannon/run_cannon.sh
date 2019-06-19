#!/bin/bash

# ################################################################
# Assignment: 4
# Task:       2+3
# Author:     Pattreeya Tanisaro
#
# ############################################################### 

# Number of processors
procs=(1 4) # for performance analysis, we need huge matrices to run the test
#procs=(1 4 16 64 256) 

# files for performance analysis
# for performance analysis, we need huge matrices to run the test
#matrixA=(A512x512.dat A1024x1024.dat)
#matrixB=(B512x512.dat B1024x1024.dat)
#matrixC=(C512x512.dat C1024x1024.dat)
matrixA=(A_8x8 A_16x16) #A512x512.dat A1024x1024.dat)
matrixB=(B_8x8 B_16x16) #B512x512.dat B1024x1024.dat)
matrixC=(C_8x8 C_16x16) #C512x512.dat C1024x1024.dat)
path="/home/lab/2011/pattreey/share/data"

# Test existing path
if [ ! -d  $path ]
then
    echo "Error! Directory $path which used to store data not exist!"
    exit 0;
fi

# Test existing data files of matrix A
for (( i=0; i < ${#matrixA}; i++))
do
  if [ ! -e $path/${matrixA[$i]} ]; then 
      echo "Error! $path/${i} not exist!"
      exit 0;
  fi
done


# Test existing data files of matrix B
for (( i=0; i < ${#matrixB}; i++))
do
  if [ ! -e $path/${matrixB[$i]} ]; then 
      echo "Error! $path/${i} not exist!"
      exit 0;
  fi
done



echo "======================= script start =========================="

for (( j = 0; j < ${#procs[@]}; j++))
  do
      echo ""
      for (( i = 0; i < ${#matrixA[@]}; i++))
      do
        echo "script: please wait..."
        echo ""
        mpiexec -np ${procs[$j]} -f hosts ./cannonmmul $path/${matrixA[$i]} $path/${matrixB[$i]} $path/${matrixC[$i]}_np${procs[$j]}_a a
        mpiexec -np ${procs[$j]} -f hosts ./cannonmmul $path/${matrixA[$i]} $path/${matrixB[$i]} $path/${matrixC[$i]}_np${procs[$j]}_b b
      done
done

echo ""
echo "======================= script done =========================="

##########################END OF FILE###############################
