#!/bin/bash

# ################################################################
# Assignment: 4
# Task:       1 a+b
# Author:     Pattreeya Tanisaro <pattreeya@gmail.com>
#
# ############################################################### 

m=5

# Number of processors
procs=(4 8 12 32)


echo "======================= script start =========================="

for (( j = 0; j < ${#procs[@]}; j++))
  do
  echo ""
  mpiexec -np ${procs[$j]} ./imagefilter ffm_1280x960.gray 960 $m row_blur_np${procs[$j]}_m$m.gray
  mpiexec -np ${procs[$j]} ./checkerboard ffm_1280x960.gray 960 $m ckb_blur_np${procs[$j]}_m$m.gray

  echo ""
done

echo ""
echo "======================= script done =========================="

##########################END OF FILE###############################
