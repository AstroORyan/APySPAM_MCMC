#$ -S /bin/bash 	# Just need this, don't know why.

#$ -q parallel		# The queue you want to use: either serial or parallel.
#$ -N Run_Systems_0_5	# The name of your job - what appears with qstat command.
#$ -l h_vmem=0.5G	# The memory of every node (default: 500MB). When parallel, memory for every core.
#$ -l np=16		# Number of nodes to use (only important with parallel queue).
#$ -m e			# Command to send email when job finishes (either crashed or finished.)
#$ -M d.oryan@lancaster.ac.uk	# Specify email to send to.

source /etc/profile	# Always needed, don't know why.

module add anaconda3	# Modules to add to the script.

python /mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/Run_MCMC.py	# Path to script to run.