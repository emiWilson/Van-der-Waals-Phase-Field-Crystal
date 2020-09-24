all:
	mpicxx MPFC_2D_edgenoise.cpp -I/software/fftw-openmpi/include -L/software/fftw-openmpi/lib -lm -O3 -Wall -lfftw3_mpi -lfftw3 -o MPFC_2D
	mpiexec -n 38 MPFC_2D

