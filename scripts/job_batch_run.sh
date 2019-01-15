#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=QPP_send
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ci411@nyu.edu
  
module purge
module load anaconda3/4.3.1
module load intel/17.0.1
source activate py2.7

#s0, Q are parameters sent here
python /home/ci411/QPP/SolarFlareGPs/scripts/simulation_script.py $rundate $s0 $Q

