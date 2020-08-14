#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job.
#SBATCH --export=ALL
# Define job name
#SBATCH -J ASE_test
# Define a standard output file. When the job is running, %u will be replaced by user name,
# %N will be replaced by the name of the node that runs the batch script, and %j will be replaced by job id number.
#SBATCH -o vasp.%u.%N.%j.out
# Define a standard error file
#SBATCH -e vasp.%u.%N.%j.err
#qos
#SBATCH --qos=dedicated
# Request the partition
#SBATCH -p rosseinsky
# Request the number of rosseinsky
#SBATCH -N 1
# Request the total number of cores
#SBATCH -n 40
# This asks for 0 days, 1 hour, 0 minutes and 0 seconds of time.
#SBATCH -t 2-00:00:00
# Specify memory per core
#SBATCH --mem-per-cpu=8000M
#
module purge
module load vasp544

export VASP_SCRIPT=$HOME/volatile/ASE_testing/Scripts/run_vasp.py
export VASP_PP_PATH=/opt/apps/chemistry/VASP/POTCAR_Files/potpaw_PBE.54/


echo =========================================================   
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

echo Executable file:                              
echo MPI parallel job.                                  
echo -------------  
echo Job output begins                                           
echo -----------------                                           
echo

hostname

echo "Print the following environmetal variables:"
echo "Job name                     : $SLURM_JOB_NAME"
echo "Job ID                       : $SLURM_JOB_ID"
echo "Job user                     : $SLURM_JOB_USER"
echo "Job array index              : $SLURM_ARRAY_TASK_ID"
echo "Submit directory             : $SLURM_SUBMIT_DIR"
echo "Temporary directory          : $TMPDIR"
echo "Submit host                  : $SLURM_SUBMIT_HOST"
echo "Queue/Partition name         : $SLURM_JOB_PARTITION"
echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Hostname of 1st node         : $HOSTNAME"
echo "Number of rosseinsky allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of processes          : $SLURM_NTASKS"
echo "Number of processes per node : $SLURM_TASKS_PER_NODE"
echo "Requested tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"




echo "Running parallel job:"

# If you use all of the cores specified in the -n line above, you do not need
# to specify how many MPI processes to use - that is the default
# the ret flag is the return code, so you can spot easily if your code failed.


python3.6 MnFe2O4_testing.py

ret=$?

# If you only wanted to some of those slots, specify the precise number:
#mpirun  -np 12 $EXEC 
#ret=$?


echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   
exit $ret
