!#/bin/bash
rundate="test"
s0s=(10)
Qs=(1.5) 

for s0 in ${s0s[@]}
	do
		for Q in ${Qs[@]}
			do
				sbatch --export=rundate=$rundate,Q=$Q,s0=$s0 job_batch_run.sh
			done
	done
