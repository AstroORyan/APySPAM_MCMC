# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:21:55 2021

@author: oryan

The Simulation-Based Inference version of APySPAM MCMC. This algorithm uses a combination of MCMC and neural networks in order to explore the posterior distribution of the parameters.

"""
# Imports
import numpy as np

import os
import torch

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Required Scripts
from Run import Run
from Import_Procedure import Imports
from Setup_Parameters import Setup_Parameters
from Secondary_Placer import Secondary_Placer
from colour import colour

## Main Functions
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

def Observation_Import(path):
    Input_Image = np.load(path)
    
    Galaxy_Name = os.path.splitext(os.path.basename(path))[0]
        
    return Input_Image, Galaxy_Name

def get_priors():
    ### [z, vx, vy, vz, M1, M2, R1, R2, phi1, phi2, theta1, theta2, time]
    min_priors = [-20, -20, -20, -20, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, -7]
    max_priors = [20, 20, 20, 20, 10, 10, 4, 4, 360, 360, 360, 360, -3]

    prior = utils.torchutils.BoxUniform(
        low = torch.as_tensor(min_priors),
        high = torch.as_tensor(max_priors)
    )

    return prior

def simulator(params):

    params_arr = np.array(params)

    sim_im, no_int_flag = Run(params_arr, xy_pos, res, filters, input_im.shape, spec_data_1, spec_data_2, z)

    if no_int_flag:
        return np.zeros(sim_im.shape) * 1e70
    
    return sim_im
    
    # sim_im_flip_0 = np.flip(sim_im, 0)
    # sim_im_flip_1 = np.flip(sim_im, 1)

    # N = sim_im.shape[0] * sim_im.shape[1]

    # c_sqr_0 = (1/N) * (((input_im - sim_im)**2) / ((2 * sigma_im)**2))
    # c_sqr_1 = (1/N) * (((input_im - sim_im_flip_0)**2) / ((2 * sigma_im)**2))
    # c_sqr_2 = (1/N) * (((input_im - sim_im_flip_1)**2) / ((2 * sigma_im)**2))

    # c_sqr = np.min([c_sqr_0, c_sqr_1, c_sqr_2])

    # ln_like = -(c_sqr/2)

    return ln_like

## Main Function
def Run_MCMC():
    global xy_pos, input_im, sigma_im, res, filters, spec_data_1, spec_data_2, z

    ## Old Stuff, get Simulation Parameters above like in the original PySPAM. Just need to call the right functions

    ## New Stuff, Setting up Simulations
    priors = get_priors()

    simulator, prior = prepare_for_sbi(simulator, prior)

    inference = SNPE(prior = prior)

    theta, x = simulate_for_sbi(simulator, proposal = prior, num_simulations = 1000)

    inference = inference.append_simulations(theta, x)

    density_estimator = inference.train()

    posterior = inference.build_posterior(density_estimator)

    posterior_samples = posterior.sample((10000,), x = input_im)

    _ = analysis.pairplot(
        posterior_samples, 
        limits = [[-20,20], [-20,20], [-20,20], [-20,20], [0.01,10], [0.01,10],[0.01,4],[0.01,4],[0,360],[0,360],[0,360],[0,360],[-7,-3]],
        figsize = (10,10)
    )

    plt.gca()
    plt.savefig('/mmfs1/home/users/oryan/APySPAM_SBI/test.jpeg')

## Initialization
if __name__ == "__main__" : 
    Run_MCMC()