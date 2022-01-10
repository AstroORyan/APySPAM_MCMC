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
import corner

# Required Scripts Import
from Run import Run
from Import_Procedure import Imports
from Setup_Parameters import Setup_Parameters
from Secondary_Placer import *
from colour import colour

class Main_Script:
    def MCMC(ndim, nwalkers, nsteps, burnin, start, Input_Image, Input_Binary, Sigma_Image, Gal_Name):
        pool = Pool(processes=32)
        p0 = [(start + np.concatenate([0.25*(-1+2*np.random.random(2)),20*(-1+2*np.random.random(1)),1.5*(-1+2*np.random.random(3)),2.4*(-1+2*np.random.random(2)),
                                      ((Input_Image.shape[0]*Resolution/15)/2)*(-1+2*np.random.random(2)),45*(-1+2*np.random.random(2)),90*(-1+2*np.random.random(2))])) for i in range(nwalkers)]
        
        # p0 = [(start + np.concatenate([0.025*(-1+2*np.random.random(2)), 0.20*(-1+2*np.random.random(1)),0.25*(-1+2*np.random.random(3)),0.25*(-1+2*np.random.random(2)),
        #                                0.5*(-1+2*np.random.random(2)),10*(-1+2*np.random.random(4)),0.5*(-1+2*np.random.random(2)),0.01*(-1+2*np.random.random(2)),0.005*(-1+2*np.random.random(2))])) for i in range(nwalkers)]
        
        print('Beginning Burnin of Input ',Gal_Name,'.')
        # Initiate the burn in phase utilising a Differential Evolution Move in the sampler.
        move = [(emcee.moves.WalkMove(), 0.5), (emcee.moves.StretchMove(), 0.5),]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=2,args=([Input_Image,Input_Binary,Sigma_Image]),moves=move,pool=pool)
        sampler.run_mcmc(p0,burnin,progress=True)
        burnin_final = sampler.get_chain()
        main_sequence_start = burnin_final[-1]
        burnin_save = sampler.chain[:,:,:].reshape((-1,ndim))
        filename = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Burnin_Samples_'+Gal_Name+'.npy'
        np.save(filename,burnin_save)
        del burnin_save, burnin_final
        
        # Now, we set off the main run:
        move = [(emcee.moves.WalkMove(), 0.5), (emcee.moves.StretchMove(), 0.5),]
        sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=([Input_Image,Input_Binary,Sigma_Image]),moves=move,pool=pool)
        print('Burn in completed... Beginning main emcee run for input ',Gal_Name,'.')
        sampler.run_mcmc(main_sequence_start, nsteps,progress=True)
        samples_final = sampler.get_chain(flat=True)
        samples = sampler.chain[:,:,:].reshape((-1,ndim))
        filename = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Main_Samples_'+Gal_Name+'.npy'
        np.save(filename,samples)
        print('Main run completed for input ',Gal_Name,'.')
        
        pool.close()
        pool.terminate()
        pool.join()
        
        time.sleep(10)
        
        return samples
        
    def Prior(theta,dims):
        x,y = theta[0:2]
        z = theta[2]
        v = theta[3:6]
        m1, m2, r1, r2, p1, p2, t1, t2 = theta[6:]
        phis = np.asarray([p1,p2])
        thetas = np.asarray([t1,t2])
        if any(np.asarray([m1,m2,r1,r2]) < 0.0):    # Negative values for these parameters obviously break the code and are unphysical.
            Chi_Squared = -np.inf
        elif (x < Limits[0]) or (x > Limits[1]) or (y < Limits[2]) or (y > Limits[3]):
            Chi_Squared = -np.inf
        elif z < -3 or z > 3:
            Chi_Squared = -np.inf
        elif v[0] < -3 or v[0] > 3:
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
        elif r1 > ((dims[0]*Resolution/15)) or r2 > ((dims[0]*Resolution/15)):  # Note, this is a cutoff of a Galactic radius of 600kpc
            Chi_Squared = -np.inf
        elif any(phis > 100) or any(phis < 0.0) or any(thetas < 90) or any(thetas > 270):
            Chi_Squared = -np.inf
        elif int(1500*(m1/(m1 + m2))) == 0 or int(1500*(m2/(m1 + m2))) == 0:
            Chi_Squared = -np.inf         
        elif m1/r1 > 10 or m2/r2 > 10:
            Chi_Squared = -np.inf
        elif r1/m1 > 10 or r2/m2 > 10:
            Chi_Squared = -np.inf
        else: 
            Chi_Squared = 0
                
        return Chi_Squared

    
def Sigma_Calc(Input_Image):
    # Artificial Cleaning of Input Image
    Noise_Array = 1e-30*np.random.poisson(10,[Input_Image.shape[0],Input_Image.shape[1]])
    Input_Image_Sigma = Input_Image.copy()
    # Input_Image_Sigma += Noise_Array
    
    h = 6.626e-34
    c = 2.998e8
    wavelength = 467.178
        
    Gain = 4.745
    NCOMBINE = 5
    DARK_VARIANCE = 0.81
        
    # Equation pulled from https://pixinsight.com/doc/tools/FluxCalibration/FluxCalibration.html . Need to generalise.
    ADU = 53.907*(np.pi*(2.5e3**2 - 0.625e3**2)/4)*176.672*0.8*Gain*0.6*1*(wavelength/c/h)
    
    Input_Image_ADU = Input_Image_Sigma*ADU
    
    Input_Image_elec = Input_Image_ADU
    
    sky = Noise_Array*ADU
    sky_rms = np.sqrt(np.mean(sky**2))
            
                
    # This equation has come from an SDSS tutorial, found at: http://classic.sdss.org/dr6/algorithms/fluxcal.html
    Sigma_Counts = np.sqrt(Input_Image_elec**2 + (np.sqrt(NCOMBINE)*sky_rms)**2)
                
    Sigma_ADU = Sigma_Counts/(Gain)
        
    Sigma = Sigma_ADU/(ADU)
    
    Sigma = Sigma.astype('float64')
                        
    return Sigma
    
def lnprob(theta,Input_Image,Input_Binary,Sigma_Image):
    Chi_Squared = 0
    Chi_Squared = Main_Script.Prior(theta,[Input_Image.shape[0],Input_Image.shape[1]])
    if Chi_Squared < 0:
        return Chi_Squared
    
    # As long as priors are satisfied, run the APySPAM simulation.
    #print('Simulation started at: ', datetime.datetime.now())
    candidate_sim_image = Run.main(theta,Conversion,Resolution,filters,[Input_Image.shape[0],Input_Image.shape[1]],Spectral_Density_1,Spectral_Density_2)
    candidate_sim_image = np.flip(np.rot90(candidate_sim_image,1),axis=1)
    #print('Simulation finished at: ', datetime.datetime.now())
    # candidate_sim_image = ndimage.gaussian_filter(candidate_sim_image,sigma=0.5,mode='nearest')
    
    # Now have candidate simulation image. Want to compare to observation using a Chi Squared.
    N = Input_Image.shape[0]*Input_Image.shape[1]
    Sigma_Array = (Input_Image - candidate_sim_image)**2/(2*(Sigma_Image)**2)
    Chi_Squared = (1/N)*np.sum(Sigma_Array)
    
    temp_binary = candidate_sim_image.copy()
    temp_binary[temp_binary > 0] = 1

    if np.sum(temp_binary) == 0.0:
        return -np.inf
    
    N = 50#len(Input_Binary[Input_Binary > 0])
    Sigma_Binary = np.sum(Input_Binary) - np.sum(temp_binary)
    Chi_Squared_Binary = abs((1/N)*Sigma_Binary)
    
    # Now, use our likelihood function to see the probability that these parameters (and simualted image) represent the observed data.
    ln_like = -0.5*(Chi_Squared + Chi_Squared_Binary)
    
    #Testing Stuff
    #print('Chi_Squared = '+ str(Chi_Squared))
    # print('Chi Squared Morph = '+ str(Chi_Squared_Morph))
    #print('Chi Squared N = '+ str(Chi_Squared_Binary))
    #print('Percentile likelihood: ', ln_like)
    
    # Chi_Squared_2 = (1/N_dof)*np.sum(Sigma_Array_2)
    # ln_like_2 = -0.5*(Chi_Squared_2 + Chi_Squared_Morph + Chi_Squared_N)
    
    # print('Full Likelihood: ', ln_like_2)
        
#    plt.figure()
#    plt.imshow(-2.5*np.log10(Input_Image) - 48.6)
#    plt.title('Input')
#    plt.savefig(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Input.png')
#    if ln_like >= -10:
    output_name = str(uuid.uuid4())
    plt.figure()
    plt.imshow(-2.5*np.log10(candidate_sim_image) - 48.6)
    plt.title(['Sim. Ln_like = ', str(ln_like)])
    plt.savefig(f'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/{output_name}.png')
    
#    plt.figure()
#    plt.imshow(Input_Binary)
#    plt.title('Input Binary')
#    plt.savefig(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/Results/In_Bin.png')
    
#    plt.figure()
#    plt.imshow(temp_binary)
#    plt.title('Sim Binary')
#    plt.savefig(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/Results/Sim_Bin.png')
    
    return ln_like

def Observation_Import(path,redshifts):
    Input_Image = np.load(path)
    # Input_Image = np.load(r'C:\Users\oryan\Documents\Raw_Observational_Data\All_Gals_Reduced\Arp_256_Reduced_Observed_Data.npy')
    # Input_Image /= 53.907456
    # Input_Image = np.rot90(Input_Image,1)
    
    Galaxy_file = os.path.basename(path)
    Galaxy_Name = os.path.splitext(Galaxy_file)[0]
    
    z_Column = redshifts[redshifts['Name'] == Galaxy_Name]['Redshift']
    temp_z = np.asarray(z_Column)[0]
    
    block_reduce_column = redshifts[redshifts['Name'] == Galaxy_Name]['Block_Reduce']
    Block_Reduce = np.asarray(block_reduce_column)[0]
    
    return Input_Image, temp_z, Block_Reduce,Galaxy_Name

def Binary_Creator(Image):
    Input_Image_Mags = -2.5*np.log10(Image) - 48.6
    
    Input_Image_Mags[np.isnan(Input_Image_Mags)] = 30
    cutoff = np.min(np.min(Input_Image_Mags)) + 3
    
    if cutoff >= 28.5:
        cutoff = 28.5
    elif cutoff < 17.5:
        cutoff = 17.5
    
    Input_Image_Binary = Input_Image_Mags.copy()
    Input_Image_Binary[Input_Image_Binary < cutoff] = 1
    Input_Image_Binary[Input_Image_Binary >= cutoff] = 0

    return Input_Image_Binary

def Run_MCMC():
    global Resolution, filters, Position, Conversion, Limits, vlim,block_reduce, Spectral_Density_1, Spectral_Density_2
    # Setup inputs and imports that can be done before iterating through MCMC
    input_folder = r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Inputs/'
    input_paths = glob.glob(input_folder+'*.*')
    n_inputs = len(input_paths)
    filters = colour.get_filters(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/')
    redshifts = read_csv('/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Redshifts/Redshifts.csv')

    
    # Setup MCMC run
    ndim = 14
    nwalkers = 80
    nsteps = 2500
    burnin = 1000
    
    Labels = ['x','y','z','vx','vy','vz','M1','M2','R1','R2','phi1','phi2','theta1','theta2']

    Spectral_Density_1 = np.loadtxt(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Spectra/Raw_Spectral_Data_Z_0.0001.txt')
    Spectral_Density_2 = np.loadtxt(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Spectra/Raw_Spectral_Data_Z_0.004.txt')

    
    # The MCMC Run
    for p in range(0,3):
        print('Beginning run for ', input_paths[p])
        Input_Image,z,block_reduce,Name = Observation_Import(input_paths[p], redshifts)
        Input_Binary = Binary_Creator(Input_Image)
        Conversion, Position, Limits, Resolution = Secondary_Placer(Input_Image, Input_Binary,z,block_reduce,Name)
        Sigma_Image = Sigma_Calc(Input_Image,)
        start = Setup_Parameters.Starting_Locations(Input_Image,Position,Resolution)
        samples = Main_Script.MCMC(ndim, nwalkers, nsteps, burnin, start, Input_Image, Input_Binary, Sigma_Image,Name)
        filename = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Results/Main_Corner_Plot_'+Name+'.png'
        fig = corner.corner(samples,labels=Labels,show_titles=True)
        fig.savefig(filename)

        del Input_Image, start, Input_Binary, Sigma_Image, samples

    

if __name__ == "__main__" : 
    Run_MCMC()