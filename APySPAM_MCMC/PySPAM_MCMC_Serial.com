#$ -S /bin/bash

#$ -q serial
#$ -N Run_Test
#$ -l h_vmem=2G
#$ -m e

source /etc/profile

module add anaconda3

python /mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/Run_MCMC.py