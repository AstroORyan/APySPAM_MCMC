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
from astropy.stats import sigma_clip

# Required Scripts Import
from Run import Run
from Import_Procedure import Imports
from Setup_Parameters import Setup_Parameters
from Secondary_Placer import Secondary_Placer
from colour import colour
from ShapelyUtils import ShapelyUtils

class Main_Script:
    def MCMC(ndim, nwalkers, nsteps, start, Input_Image, Sigma_Image, Gal_Name):
        
        cwd = os.getcwd() + '/PySPAM_Original_Python_MCMC'
        
        p0 = [(start + np.concatenate([2.5*(-1+2*np.random.random(1)),1.5*(-1+2*np.random.random(3)),2.4*(-1+2*np.random.random(2)),
                                      1.5*(-1+2*np.random.random(2)),45*(-1+2*np.random.random(2)),45*(-1+2*np.random.random(2)), (-1+2*np.random.random(1))])) for i in range(nwalkers)]
        
        print('Beginning Burnin of Input ',Gal_Name,'.')

        # filename = f'{cwd}/Results/' + Gal_Name + '_Full.h5'
        # backend = emcee.backends.HDFBackend(filename)
        # backend.reset(nwalkers,ndim)

        # Initiate the burn in phase utilising a Differential Evolution Move in the sampler.
        move = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
        print('Made it to setting up the Pool!')
        # with Pool(processes=16) as pool:
        #     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([Input_Image,Sigma_Image]),moves=move,pool=pool,backend=backend)
        #     sampler.run_mcmc(p0,nsteps,progress=True)
        #     pool.close()
        #     pool.join()

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=([Input_Image,Sigma_Image]),moves=move)
        sampler.run_mcmc(p0,nsteps,progress=True)

        time.sleep(10)

        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(2 * np.nanmax(tau))
        thin = int(0.5 * np.nanmin(tau))

        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_prob_samples = sampler.get_log_prob(discard=burnin,flat=True,thin=thin)
        #log_prior_samples = sampler.get_blobs(discard=burnin,flat=True,thin=thin)

        print("Burnin-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("Flat chain shape: {0}".format(samples.shape))
        print("flat log prob shape: {0}".format(log_prob_samples.shape))
        #print("flat log prior shape: {0}".format(log_prior_samples.shape))

        all_samples = np.concatenate((samples, log_prob_samples[:,None]), axis = 1)

        try:
            filename = f'{cwd}/Results/Run_Samples_{Gal_Name}.npy'
            np.save(filename,all_samples)
        except:
            output_name = str(uuid.uuid4())
            filename = f'{cwd}/Results/Run_Samples_{output_name}.npy'
            print(f'WARNING! Savename didn\'t work. Have saved current run as {output_name}')
            np.save(filename,all_samples)
        
        return all_samples[:,:ndim]
        
    def Prior(theta,dims):
        #x,y = theta[0:2]
        z = theta[0]
        v = theta[1:4]
        m1, m2, r1, r2, p1, p2, t1, t2, t = theta[4:]
        phis = np.asarray([p1,p2])
        thetas = np.asarray([t1,t2])
        if any(np.asarray([m1,m2,r1,r2]) < 0.0):    # Negative values for these parameters obviously break the code and are unphysical.
            Chi_Squared = -np.inf
        # elif (x < Limits[0]) or (x > Limits[1]) or (y < Limits[2]) or (y > Limits[3]):
        #     Chi_Squared = -np.inf
        elif z < -100 or z > 100:
            Chi_Squared = -np.inf
        elif v[0] < -100 or v[0] > 100:
            Chi_Squared = -np.inf
        elif v[1] < -100 or v[1] > 100:
            Chi_Squared = -np.inf
        elif v[2] > 100 or v[2] < -100:
            Chi_Squared = -np.inf
        elif np.linalg.norm(v) == 0.0:
            Chi_Squared = -7e6
        elif m1 > 10 or m2 > 10 or m1 <= 0.01 or m2 <= 0.01:      # Note, this is an upper limit on mass of 5x10^12 Solar Masses.
            Chi_Squared = -np.inf
        elif r1 > 5 or r2 > 5 or r1 <= 0.01 or r2 <= 0.01:    
            Chi_Squared = -np.inf            
        elif any(phis > 360) or any(phis < 0.0) or any(thetas < 0.0) or any(thetas > 360):
            Chi_Squared = -np.inf
        elif int(1000*(m1/(m1 + m2))) == 0 or int(1000*(m2/(m1 + m2))) == 0:
            Chi_Squared = -np.inf         
        elif m1/r1 > 5 or m2/r2 > 5: 	# This says, if I have a maximum mass of 5e11 SM, then my minimum radius is 15kpc.
            Chi_Squared = -np.inf
        elif m1/r1 < 0.5 or m2/r2 < 0.5:  # This says, if I have a minimum mass of 1e10, the maximum radius it can have is 5kpc.
            Chi_Squared = -np.inf
        elif t > -3 or t < -7:
            Chi_Squared = -np.inf
        else: 
            Chi_Squared = 0

        m1 = abs(m1)
        m2 = abs(m2)
        r1 = abs(r1)
        r2 = abs(r2)
        p1 = abs(p1)
        p2 = abs(p2)
        t1 = abs(t1)
        t2 = abs(t2)
        if t > 0:
             t = -t
        if t < -3:
             t = -5
        if t > -7:
             t = -5
        theta = np.array([z,v[0],v[1],v[2],m1,m2,r1,r2,p1,p2,t1,t2,t])
                
        return Chi_Squared, theta

    
def Sigma_Calc(Input_Image):
    # Artificial Cleaning of Input 
    input_image = Input_Image.copy()
    input_image[input_image <= 0] = np.min(abs(input_image))
    factor = 1e-28
    input_image = input_image/factor
    sigma_image = np.sqrt(input_image)
    sigma_image[sigma_image == 0] = np.min(sigma_image[sigma_image > 0])
    sigma_image_exp = (sigma_image * factor).astype('float64')
    
    return sigma_image_exp

    
def lnprob(theta,Input_Image,Sigma_Image):
    ln_prior, theta = Main_Script.Prior(theta,[Input_Image.shape[0],Input_Image.shape[1]])
    
    if ln_prior == -np.inf:
        return -np.inf

    # As long as priors are satisfied, run the APySPAM simulation.
    #print('Simulation started at: ', datetime.datetime.now())
    sim_image, no_int_flag = Run.main(theta,xy_pos,Resolution,filters,[Input_Image.shape[0],Input_Image.shape[1]],Spectral_Density_1,Spectral_Density_2,z)
    
    if no_int_flag:
        return -np.inf
    
    # Now have candidate simulation image. Want to compare to observation using a Chi Squared.
    N = Input_Image.shape[0]*Input_Image.shape[1]
    Sigma_Array = (Input_Image - sim_image)**2/(2*(Sigma_Image**2))
    Chi_Squared = (1/N)*np.sum(Sigma_Array) - 1

    sim_cutouts, sim_polygons, succeed_flag = ShapelyUtils.get_galaxy_polygon(sim_image, theta[:2])

    if len(sim_cutouts) == 0 and len(sim_polygons) == 0 and not succeed_flag:
        print('Failed. Trying new sim...')
        return -np.inf

    if succeed_flag:
        prim_flux = ShapelyUtils.calculate_polygon_flux(cutouts[0], polygons[0])
        sec_flux = ShapelyUtils.calculate_polygon_flux(cutouts[1], polygons[1])
        sim_prim_flux = ShapelyUtils.calculate_polygon_flux(sim_cutouts[0], sim_polygons[0])
        sim_sec_flux = ShapelyUtils.calculate_polygon_flux(sim_cutouts[1], sim_polygons[1])
        jaccard_dist = ShapelyUtils.get_jaccard_dist(polygons, sim_polygons)

        Chi_Squared_flux = (( prim_flux / sim_prim_flux ) * jaccard_dist[0]) + (( sec_flux / sim_sec_flux ) * jaccard_dist[1]) - 1
        
        ln_like = -((Chi_Squared/2) - (Chi_Squared_flux/2))/2 + ln_prior
        print('Parameters : ', theta)
        print('Log Likelihood: ', ln_like)
        print('Chi Squared Flux: ', Chi_Squared_flux)
        print('Chi Squared: ', Chi_Squared)
    else:
        ln_like = -(Chi_Squared/2) + ln_prior
        print('Parameters : ', theta)
        print('Log Likelihood: ', ln_like)
        print('Chi Squared: ', Chi_Squared)


    sys.exit()

    # Now, use our likelihood function to see the probability that these parameters (and simualted image) represent the observed data.

## Testing Lines
#    if ln_like >= -2.0:
#        output_name = str(uuid.uuid4())
#        plt.figure()
#        plt.imshow(-2.5*np.log10(candidate_sim_image) - 48.6)
#        plt.title(['Sim. Ln_like = ', str(ln_like)])
#        plt.savefig(f'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Test_Images/{output_name}.png')
#        plt.close()

    if np.isnan(ln_like):
        ln_like = -np.inf
    
    return ln_like

def star_remove(im):
    n = 99.9
    cut = np.percentile(im, n)

    mask = im.copy()
    mask[mask > cut] = True
    mask[mask <= cut] = False
    mask = np.ma.make_mask(mask)

    im[mask] = 0

    return im


def Observation_Import(path):
    input_im = np.load(path)
    
    Galaxy_Name = os.path.splitext(os.path.basename(path))[0]
        
    Input_Image = star_remove(input_im)

    return Input_Image, Galaxy_Name

def Run_MCMC():
    global Resolution, filters, cutouts, polygons, Spectral_Density_1, Spectral_Density_2,z,xy_pos
    # Setup inputs and imports that can be done before iterating through MCMC
    
    cwd = os.getcwd() #+ '/PySPAM_Original_Python_MCMC' # To run locally, remove this addition.
    
    input_folder = f'{cwd}/All_Inputs/'
    input_paths = glob.glob(input_folder+'*.*')
    filters = colour.get_filters(cwd)
    
    if len(filters) < 0.5:
        print('WARNING: No Filters input.')
        sys.exit()
    
    redshifts = read_csv(f'{cwd}/Redshifts/Redshifts.csv')
    
    # Setup MCMC run
    ndim = 13
    nwalkers = 96
    nsteps = 20000

    Spectral_Density_1 = np.loadtxt(f'{cwd}/Spectra/Raw_Spectral_Data_Z_0.0001.txt')
    Spectral_Density_2 = np.loadtxt(f'{cwd}/Spectra/Raw_Spectral_Data_Z_0.004.txt')
    
    # The MCMC Run
    for p in range(0,5):
        print('Beginning run for ', input_paths[p])
        Input_Image, Name = Observation_Import(input_paths[p])
        Sigma_Image = Sigma_Calc(Input_Image)
        xy_pos, Resolution, z, skip_flag = Secondary_Placer.get_secondary_coords(Name, redshifts)
        if skip_flag:
            continue
        cutouts, polygons, _ = ShapelyUtils.get_galaxy_polygon(Input_Image, xy_pos)

        start = Setup_Parameters.Starting_Locations()
        samples = Main_Script.MCMC(ndim, nwalkers, nsteps, start, Input_Image, Sigma_Image,Name)
        filename = '/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC/Results/Main_Corner_Plot_'+Name+'.png'
        Labels = ['z','vx','vy','vz','M1','M2','R1','R2','phi1','phi2','theta1','theta2','t']
        fig = corner.corner(samples,labels=Labels,show_titles=True)
        fig.savefig(filename)
        plt.close()

        del Input_Image, polygons, start, Sigma_Image, samples

    

if __name__ == "__main__" : 
    Run_MCMC()