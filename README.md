# Parallelization_MPICH2_C
A collection of codes using MPI (High Performance Message Passing Interface) to handle algorithms which requires large resources on distributed system. [[Old Stuff from old days]] 

Firstly, test your environment for MPICH using "hellompi.c"
The executing program should run on all compute nodes (2^n).
We use the root process to print out the output from all processes.

![Image1](parallel_p3.png =400x300) 

![Image2](parallel_p4.png =400x300)


More information about MPICH and MPI can be found under: https://www.mpich.org/



Resources for MPI and GPUs can be found at:

* NVIDIA MPI Solution for GPUs: https://developer.nvidia.com/mpi-solutions-gpus

* Multi-GPU Programming with MPI (Slides) from J. Kraus (Nvidia): http://on-demand.gputechconf.com/gtc/2015/presentation/S5117-Jiri-Kraus.pdf
* Multi-GPU Programming with MPI (Slides) J.Kraus & P. Messmer (Nvidia) http://on-demand.gputechconf.com/gtc/2014/presentations/S4236-multi-gpu-programming-mpi.pdf
