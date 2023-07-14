#!/bin/bash -l

#SBATCH -J Slit_2_3
#SBATCH -o %x.%j.out
#SBATCH --mail-user=jeo27@njit.edu
#Default in slurm
#SBATCH --mail-type=ALL
#SBATCH -p gor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6

module load foss/2021b LAMMPS

srun -n $SLURM_NTASKS lmp -in input_3.in
