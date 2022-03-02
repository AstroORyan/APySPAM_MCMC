# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:21:55 2021

@author: oryan

This is going to be the integration script with APySPAM in its new form. The new form is much more efficinet and fast.

"""
# Required Packages
import glob
import numpy as np
import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from scipy import ndimage
from pandas import read_csv
import uuid
import os
import time
from multiprocessing import Pool
import sys
import corner

# Required Scripts Import
from Run import Run
from Import_Procedure import Imports
from Setup_Parameters import Setup_Parameters
from Secondary_Placer import *
from Secondary_Centered import *
from colour import colour

class Main_Script:
    def MCMC(ndim, nwalkers, nsteps, start, Input_Image, Input_Binary, Sigma_Image, Gal_Name):
        p0 = [(start + np.concatenate([0.1*(-1+2*np.random.random(2)),2.5*(-1+2*np.random.random(1)),1.5*(-1+2*np.random.random(3)),2.4*(-1+2*np.random.random(2)),
                                      1.5*(-1+2*np.random.random(2)),170*(-1+2*np.random.random(2)),170*(-1+2*np.random.random(2)), (-1+2*np.random.random(1))])) for i in range(nwalkers)]
        
        print('Beginning Burnin of Input ',Gal_Name,'.')
        # Initiate the burn in phase utilising a Differential Evolution Move in the sampler.
        move = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
        with Pool(processes=64) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2,args=([Input_Image,Input_Binary,Sigma_Image]),moves=move,pool=pool)
            sampler.run_mcmc(p0,nsteps,progress=True)

        run_save = sampler.chain()
        run_save_flat = sampler.chain[:,:,:].reshape((-1,ndim))
        try:
            filename = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Run_Samples_'+Gal_Name+'.npy'
            np.save(filename,run_save)
        except:
            output_name = str(uuid.uuid4())
            filename = f'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Run_Samples_{output_name}.npy'
            print(f'WARNING! Savename didn\'t work. Have saved current run as {output_name}')
            np.save(filename,run_save)
            
        try:
            filename_flat = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Run_Flat_Samples_'+Gal_Name+'.npy'
            np.save(filename_flat,run_save_flat)
        except:
            output_name = str(uuid.uuid4())
            filename = f'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Run_Flat_Samples_{output_name}.npy'
            print(f'WARNING! Savename didn\'t work. Have saved current flat run as {output_name}')
            np.save(filename_flat,run_save_flat)
        
        print('Autocorrelation time of Main: {0:.2f} steps'.format(sampler.get_autocorr_time()[0]))
        
        time.sleep(10)

        run_save_no_burnin = run_save[int(0.1*nsteps):,:,:]
        run_save_no_burnin_flat = run_save_no_burnin.reshape((-1,ndim))
        
        return run_save_no_burnin_flat
        
    def Prior(theta,dims):
        x,y = theta[0:2]
        z = theta[2]
        v = theta[3:6]
        m1, m2, r1, r2, p1, p2, t1, t2, t = theta[6:]
        phis = np.asarray([p1,p2])
        thetas = np.asarray([t1,t2])
        if any(np.asarray([m1,m2,r1,r2]) < 0.0):    # Negative values for these parameters obviously break the code and are unphysical.
            Chi_Squared = -np.inf
        elif (x < Limits[0]) or (x > Limits[1]) or (y < Limits[2]) or (y > Limits[3]):
            Chi_Squared = -np.inf
        elif z < -5 or z > 5:
            Chi_Squared = -np.inf
        elif v[0] < -5 or v[0] > 5:
            Chi_Squared = -np.inf
        elif v[1] < -3 or v[1] > 3:
            Chi_Squared = -np.inf
        elif v[2] > 3 or v[2] < -3:
            Chi_Squared = -np.inf
        elif np.linalg.norm(v) == 0.0:
            Chi_Squared = -7e6
        elif np.linalg.norm([x,y,z]) == 0.0:
            Chi_Squared = -7e6
        elif m1 > 5 or m2 > 5 or m1 <= 0.1 or m2 <= 0.1:      # Note, this is an upper limit on mass of 5x10^12 Solar Masses.
            Chi_Squared = -np.inf
        elif r1 > 4 or r2 > 4:    
            Chi_Squared = -np.inf
        elif any(phis > 360) or any(phis < 0.0) or any(thetas < 0) or any(thetas > 360):
            Chi_Squared = -np.inf
        elif int(2000*(m1/(m1 + m2))) == 0 or int(2000*(m2/(m1 + m2))) == 0:
            Chi_Squared = -np.inf         
        elif m1/r1 > 5 or m2/r2 > 5: 	# This says, if I have a maximum mass of 5e11 SM, then my minimum radius is 15kpc.
            Chi_Squared = -np.inf
        elif m1/r1 < 0.5 or m2/r2 < 0.5:  # This says, if I have a minimum mass of 1e10, the maximum radius it can have is 5kpc.
            Chi_Squared = -np.inf
        elif t > -4 or t < -6:
            Chi_Squared = -np.inf
        else: 
            Chi_Squared = 0
                
        return Chi_Squared

    
def Sigma_Calc(Input_Image):
    # Artificial Cleaning of Input 
    input_image = Input_Image.copy()
    input_image[input_image <= 0] = np.min(abs(input_image))
    input_image = input_image/1e-28
    sigma_image = np.sqrt(input_image)
    sigma_image[sigma_image == 0] = np.min(sigma_image[sigma_image > 0])
    sigma_image_exp = (sigma_image * 1e-28).astype('float64')
    
    return sigma_image_exp

    
def lnprob(theta,Input_Image,Input_Binary,Sigma_Image):
    Chi_Squared = 0
    Chi_Squared = Main_Script.Prior(theta,[Input_Image.shape[0],Input_Image.shape[1]])
    if Chi_Squared < 0:
        return Chi_Squared
    
    # As long as priors are satisfied, run the APySPAM simulation.
    #print('Simulation started at: ', datetime.datetime.now())
    candidate_sim_image = Run.main(theta,Conversion,Resolution,filters,[Input_Image.shape[0],Input_Image.shape[1]],Spectral_Density_1,Spectral_Density_2,z,Input_Image.shape[1])
    
    # Now have candidate simulation image. Want to compare to observation using a Chi Squared.
    N = Input_Image.shape[0]*Input_Image.shape[1]
    Sigma_Array = (Input_Image - candidate_sim_image)**2/(2*(Sigma_Image)**2)
    Chi_Squared = (1/N)*np.sum(Sigma_Array)
    
    mask = candidate_sim_image > 0

    if np.sum(mask) == 0.0:
        return -np.inf
    
    active_chi_sq = (1/np.sum(mask))*np.sum((Input_Image[mask] - candidate_sim_image[mask])**2/(2*(Sigma_Image[mask])**2))

    # Now, use our likelihood function to see the probability that these parameters (and simualted image) represent the observed data.
    ln_like = -0.5*(Chi_Squared/2 + active_chi_sq/2)

    if ln_like >= -2.5:
        output_name = str(uuid.uuid4())
        plt.figure()
        plt.imshow(-2.5*np.log10(candidate_sim_image) - 48.6)
        plt.title(['Sim. Ln_like = ', str(ln_like)])
        plt.savefig(f'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Test_Images/{output_name}.png')
    
    return ln_like

def Observation_Import(path,redshifts):
    Input_Image = np.load(path)
    
    Galaxy_file = os.path.basename(path)
    Galaxy_Name = os.path.splitext(Galaxy_file)[0]
    
    temp_z = redshifts.query('Names == @Galaxy_Name')['Redshift'].iloc[0]
    
    Block_Reduce = redshifts.query('Names == @Galaxy_Name')['block_reduce'].iloc[0]
    
    return Input_Image, temp_z, Block_Reduce,Galaxy_Name

def Binary_Creator(Image):
    Input_Image_Mags = -2.5*np.log10(Image) - 48.6
    
    Input_Image_Mags[np.isnan(Input_Image_Mags)] = 30
    cutoff = np.min(np.min(Input_Image_Mags)) + 5
    
    if cutoff >= 28.5:
        cutoff = 28.5
    elif cutoff < 17.5:
        cutoff = 17.5
    
    Input_Image_Binary = Input_Image_Mags.copy()
    Input_Image_Binary[Input_Image_Binary < cutoff] = 1
    Input_Image_Binary[Input_Image_Binary >= cutoff] = 0

    return Input_Image_Binary

def Run_MCMC():
    global Resolution, filters, Position, Conversion, Limits, vlim,block_reduce, Spectral_Density_1, Spectral_Density_2,z
    # Setup inputs and imports that can be done before iterating through MCMC
    input_folder = r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Inputs/'
    #input_folder = 'C:\\Users\\oryan\\Documents\\PySPAM_Original_Python_MCMC\\APySPAM_MCMC\\Inputs\\'
    input_paths = glob.glob(input_folder+'*.*')
    n_inputs = len(input_paths)
    filters = colour.get_filters(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/')
    #filters = colour.get_filters('C:\\Users\\oryan\\Documents\\PySPAM_Original_Python_MCMC\\APySPAM_MCMC\\')
    redshifts = read_csv('/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Redshifts/Redshifts.csv')
    #redshifts = read_csv('C:\\Users\\oryan\\Documents\\PySPAM_Original_Python_MCMC\\APySPAM_MCMC\\Redshifts\\Redshifts.csv')

    
    # Setup MCMC run
    ndim = 15
    nwalkers = 100
    nsteps = 8500
    
    Labels = ['x','y','z','vx','vy','vz','M1','M2','R1','R2','phi1','phi2','theta1','theta2','t']

    centre_flag = True

    Spectral_Density_1 = np.loadtxt(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Spectra/Raw_Spectral_Data_Z_0.0001.txt')
    Spectral_Density_2 = np.loadtxt(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Spectra/Raw_Spectral_Data_Z_0.004.txt')
    #Spectral_Density_1 = np.loadtxt(r'C:\Users\oryan\Documents\PySPAM_Original_Python_MCMC\APySPAM_MCMC\Spectra\Raw_Spectral_Data_Z_0.0001.txt')
    #Spectral_Density_2 = np.loadtxt(r'C:\Users\oryan\Documents\PySPAM_Original_Python_MCMC\APySPAM_MCMC\Spectra\Raw_Spectral_Data_Z_0.004.txt')
    
    # The MCMC Run
    for p in range(0,n_inputs):
        print('Beginning run for ', input_paths[p])
        Input_Image,z,block_reduce,Name = Observation_Import(input_paths[p], redshifts)
        Input_Binary = Binary_Creator(Input_Image)
        if centre_flag:
            Conversion,Position,Limits,Resolution = Secondary_Center(Input_Image, Input_Binary, z, block_reduce, Name)
        else:
            Conversion, Position, Limits, Resolution = Secondary_Placer(Input_Image, Input_Binary,z,block_reduce,Name)
        Sigma_Image = Sigma_Calc(Input_Image,)
        start = Setup_Parameters.Starting_Locations(Input_Image,Position,Resolution)
        samples = Main_Script.MCMC(ndim, nwalkers, nsteps, start, Input_Image, Input_Binary, Sigma_Image,Name)
        filename = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Main_Corner_Plot_'+Name+'.png'
        fig = corner.corner(samples,labels=Labels,show_titles=True)
        fig.savefig(filename)

        del Input_Image, start, Input_Binary, Sigma_Image, samples

    

if __name__ == "__main__" : 
    Run_MCMC()