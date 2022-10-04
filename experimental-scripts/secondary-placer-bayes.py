'''
Author: David O'Ryan
Date: 03/05/2022

This script will do a very fast likelihood exploration of where the secondary galaxy might be in an image. This is done by creating a small cutout of the center of the primary galaxy (assumed to be in the center) and then searched around somewhere which
looks approximately similar.
'''
## Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

## Functions
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

def pixel_like_calc(im, sig, centre):
    pix_likelihood = np.zeros([im.shape[0],im.shape[1]])
    for i in range(im.shape[0]):       
        for j in range(im.shape[1]):
            x_up_lim = i + 5
            x_low_lim = i - 5
            y_up_lim = j + 5
            y_low_lim = j - 5
            
            if x_low_lim < 0:
                x_low_lim = 0
            if x_up_lim >= im.shape[0]:
                x_up_lim = im.shape[0]
            if y_low_lim < 0:
                y_low_lim = 0
            if y_up_lim > im.shape[1]:
                y_up_lim = im.shape[1]
                
            im_area = im[y_low_lim:y_up_lim,x_low_lim:x_up_lim]
            sig_area = sig[y_low_lim:y_up_lim,x_low_lim:x_up_lim]
            centre_area = centre[:im_area.shape[0], :im_area.shape[1]]
            N = im_area.shape[0] * im_area.shape[1]
                
            pix_likelihood[i,j] = (1/N) * np.sum((im_area - centre_area) ** 2 / (2*(sig_area) ** 2))
    return pix_likelihood

def find_res(z,b):
    cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.275 * u.K, Om0 = 0.308)
    d = np.float64(cosmo.luminosity_distance(z)) * 1e3
    Resolution = np.float64(cosmo.kpc_proper_per_arcmin(z)) * (0.396 / 60) * b
    return Resolution


## Main Function
def Secondary_Center(Input_Image,z,block_reduce):
    sigma = Sigma_Calc(Input_Image)
    
    centre = int(Input_Image.shape[0]/2)

    prim_centre = Input_Image[centre - 5: centre + 5, centre - 5: centre + 5]

    likelihood_dist = pixel_like_calc(Input_Image, sigma, prim_centre)

    likelihood_dist_min = likelihood_dist.copy()
    likelihood_dist_min[15:35, 15:35] = 1e30

    y, x = np.where(likelihood_dist_min == np.nanmin(likelihood_dist_min))
    if len(y) > 1:
        y = y[0]
    if len(x) > 1:
        x = x[0]

    Resolution = find_res(z, block_reduce)

    x_sim_pos = ((x - 25) * Resolution) / 15
    y_sim_pos = ((y - 25) * Resolution) / 15

    return [x_sim_pos, y_sim_pos], sigma, Resolution

# ## Temp Initialization Function
# if __name__ == '__main__':
#     df = pd.read_csv('C:/Users/oryan/Documents/PySPAM_Original_Python_MCMC/APySPAM_MCMC/Redshifts/Redshifts.csv')
    
#     for k in range(len(df)):
#         row = df.iloc[k]
#         z = row.Redshift
#         block_reduce = row.block_reduce
#         name = row.Names
#         Input_Image = np.load(f'C:/Users/oryan/Documents/PySPAM_Original_Python_MCMC/APySPAM_MCMC/All_Inputs/{name}.npy')

#         Secondary_Center(Input_Image, z, block_reduce, name)

#     print('Algorithm Complete.')

    
