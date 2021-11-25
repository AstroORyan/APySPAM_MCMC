# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:34:05 2021

@author: oryan

Algorithm to find the colour fluxes from each pixel in our image.

"""
import glob
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class colour:
    def get_filters(folder):
        filter_folder = folder+'\\Filters\\'
        filters = glob.glob(filter_folder+'*.*')
        
        filter_data = []
        
        for i in range(len(filters)):
            temp = np.loadtxt(filters[i])
            filter_data.append(temp)
            
        return filter_data
    
    def get_dist(z):
      Omega_Matter = 0.308   #These values utilise a Reference Frame defined by the NASA ED
      Omega_Vacuum = 0.692
      Omega_tot = (Omega_Matter + Omega_Vacuum)
      H_0 = 67.8
      q_0 = -0.59
      
      Z = z - (z**2)*(1+q_0)/2
      Dnow = 2.9998e8*Z/H_0
      x = (1-Omega_tot)*Z**2
      if x < 0:
          J_x = np.sin(np.sqrt(-x))/np.sqrt(-x)
      elif x > 0:
          J_x = np.sinh(np.sqrt(x))/np.sqrt(x)
      elif x == 0:
          J_x = 1 + (x/6) + (x**2)/120
      D_A = Dnow*J_x/(1+z)
      D_L = ((1+z)**2)*D_A   #This gives the distance in kiloparsecs.
      
      parsec_cm = 3.086e21                  #This is kiloparsecs in centimeters.
      d_cm = D_L*parsec_cm
      
      return d_cm
  
    def get_ext(wav,flux):
    #Quickly Apply an extinction algorithm
      Coefficient = np.zeros(len(wav))
      test_wav = wav*1e-10/1e-6
      E_B_V = 0.44
      for i in range(len(wav)):              # Equations are from arxiv astro-ph/0109035
          if test_wav[i] < 0.12:
              Coefficient[i] = 0
          elif (test_wav[i] >= 0.12 and test_wav[i] <= 0.63):
              Coefficient[i] = 1.17*(-2.156 + 1.509/test_wav[i] - 0.198/(test_wav[i]**2) + 0.011/(test_wav[i]**3)) + 1.78
          elif (test_wav[i] > 0.63 and test_wav[i] <= 2.2):
              Coefficient[i] = 1.17*(-1.857 + 1.040/test_wav[i]) + 1.78
          else:
              Coefficient[i] = 0
         
      flux = flux*10**(-0.4*Coefficient*E_B_V)
      
      return flux
    
    def get_colour(SED,Wavelength,filters,z):
        
        c = 2.988e8
        conv_units = 3.826e33   # To convert units from default GALEX units to ergs/s. Found from GALEX documentation. 
        
        d_cm = colour.get_dist(z)
        
        part_colours = []
        
        for i in range(len(filters)):
            tmp = filters[i]
            filter_wav = tmp[:,0]
            filter_tran = tmp[:,1]
            del tmp
            
            # Apply redshift to observed wavelengths.            
            wav_obs = Wavelength*(1+z)
            
            # Now, must pull out the flux distribution from wavelength_obs which matches filter:
            start_index = np.where(wav_obs >= filter_wav[0])[0][0]
            final_index = np.where(wav_obs >= filter_wav[-1])[0][0]
            
            # Get relevent stuff from data.
            wav_obs_fil = wav_obs[start_index:final_index]
            flux_dist = SED[:,start_index:final_index]
            
            flux_dist = colour.get_ext(wav_obs_fil,flux_dist)
            
            # Quickly get emitted frequency:
            freq_obs = 2.988e8/(1e-10*wav_obs_fil)
            freq_obs = np.flip(freq_obs)
            
            # Create an interpolation object so that it will match the data.
            f = interpolate.interp1d(filter_wav,filter_tran,kind='cubic')
            
            # Interpolate and get the transmission for the observed data.
            tran_new = f(wav_obs_fil)
            
            # Now, need to use this to get the population colours.
            colour_dist_per_A = flux_dist*tran_new[np.newaxis,:]
            
            # Now, need to conduct a lot of conversions to get this into useful units. 
            colour_dist_L = colour_dist_per_A*wav_obs_fil*((wav_obs_fil*1e-10)/c)
            
            # Now, need to convert this into frequency units as well as account for the distance to the object.
            colour_dist_flux = (conv_units*colour_dist_L)/(4*np.pi*d_cm**2)
    
            # Now, we can integrate:
            int_colour = np.trapz((1/freq_obs)*colour_dist_flux, freq_obs)
            int_colour = int_colour/np.trapz(tran_new/freq_obs, freq_obs)   # Second division is as described in the GALEX docs.
            
            # Add result to list:
            part_colours.append(int_colour)
            
        return part_colours