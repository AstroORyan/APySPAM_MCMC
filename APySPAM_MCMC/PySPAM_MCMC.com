#$ -S /bin/bash

#$ -q parallel
#$ -N Run_Systems_0_5
#$ -l h_vmem=0.5G
#$ -l np=16
#$ -m e
#$ -M d.oryan@lancaster.ac.uk

source /etc/profile

module add anaconda3

python /mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/Run_MCMC.py