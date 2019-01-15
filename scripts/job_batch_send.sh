!#/bin/bash
rundate="011419"
s0s=(1.00000000) #2.67978841 4.35957681 6.03936522 7.71915362 9.39894203 11.07873043 12.75851884 14.43830725 16.11809565)
Qs=(1.00000000) # 1.22174803 1.44349606 1.66524409 1.88699212 2.10874015 2.33048818 2.55223621 2.77398424 2.99573227) 

for s0 in ${s0s[@]}
	do
		for Q in ${Qs[@]}
			do
				sbatch --export=rundate=$rundate,Q=$Q,s0=$s0 job_batch_run.sh
				#echo $rundate $Q $s0
			done
		done
