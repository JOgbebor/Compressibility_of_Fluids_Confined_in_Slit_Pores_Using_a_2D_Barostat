#!/bin/bash -l

#SBATCH -J IP_BARO
#SBATCH -o %x.%j.out
#SBATCH --mail-user=jeo27@njit.edu
#Default in slurm
#SBATCH --mail-type=ALL
#SBATCH -p gor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

module load gnu8/8.3.0  openmpi3/3.1.4 lammps/20200602

srun lmp_mpi  -in  input_2.in
