#$ -S /bin/bash

#$ -q parallel
#$ -N Run_Test_More
#$ -l h_vmem=8G
#$ -l np=16
#$ -l node_type=10Geth128G
#$ -m e
#$ -M d.oryan@lancaster.ac.uk

source /etc/profile

module add anaconda3

python /mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Run_MCMC.py