#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=job_batch_test
#SBATCH --mail-type=END
#SBATCH --mail-user=ci411@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge
module load anaconda3/4.3.1
module load intel/17.0.1
source activate py2.7

#s0, Q are parameters sent here
#outdir="/scratch/ci411/Data/Simulating/${rundate}/simulated_burst_s0${s0}_Q${Q}"
python /home/ci411/QPP/SolarFlareGPs/scripts/simulation_script.py $rundate $s0 $Q

