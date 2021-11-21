#$ -S /bin/bash

#$ -q parallel
#$ -N Run_Test
#$ -l h_vmem=2G
#$ -l np=16
#$ -m e
#$ -M d.oryan@lancaster.ac.uk

source /etc/profile

module add anaconda3

python /mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_HEC/Run_MCMC.py