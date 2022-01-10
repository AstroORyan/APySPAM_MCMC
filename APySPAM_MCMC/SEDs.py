# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:54:48 2021

@author: oryan
"""
import numpy as np
# import datetime
# import sys

class SED:
    def getSED(Spectral_Density_1,Spectral_Density_2,Ages,n1,n2,time,Weights,Part_Mass,SFR_Mass,h):
        Initial_Flux_Dist = np.zeros([n1+n2,6900])
        Additional_Flux = np.zeros([n1+n2,6900])
        
        #Load in original SED.
        Initial_Flux_Dist[:n1,:] = SED.initSED(Spectral_Density_1,Ages[0],n1)
        Initial_Flux_Dist[n1:,:] = SED.initSED(Spectral_Density_2,Ages[1],n2)
        
        Weights = Weights[:,np.newaxis]
        
        # Multiply by the mass each particle has.
        Initial_Flux_Dist[:n1,:] = Initial_Flux_Dist[:n1,:]*(Weights[:n1,:])*(Part_Mass[0]*1e11)
        Initial_Flux_Dist[n1:,:] = Initial_Flux_Dist[n1:,:]*(Weights[n1:,:])*(Part_Mass[1]*1e11)
        
        # Now, the really rough part... Finding the contribution from all the mass formed. 
        Additional_Flux[:n1,:] = SED.SFR_Flux(Spectral_Density_1[0,:],Spectral_Density_1[1:,1:],Ages[0],n1,time,SFR_Mass[:n1,:],h)
        Additional_Flux[n1:,:] = SED.SFR_Flux(Spectral_Density_2[0,:],Spectral_Density_2[1:,1:],Ages[1],n2,time,SFR_Mass[n1:,:],h)
                
        Flux_Dist = Initial_Flux_Dist + Additional_Flux
        wavelength = Spectral_Density_1[1:,0]
        
        return [Flux_Dist,wavelength]
                
        
    def initSED(SEDs,Age,n):
        # First, load in required files
                
        # Wavelengths = SEDs[1:,0]
        Ages = SEDs[0,1:]/1e9
        Fluxes = SEDs[1:,1:]
        
        Age_Index = np.where(Age >= Ages)[0][-1]
        
        Spectral_Density = Fluxes[:,Age_Index - 1].reshape(Fluxes.shape[0],1)*np.ones(n)
        Spectral_Density = Spectral_Density.transpose()
        
        return Spectral_Density
    
    def SFR_Flux(Spec_Ages,Spectra,Age,n,time,Mass,h):
        t_index = int(time/h)
        
        # Then, define the 3D array which will have what I need. 
        temp_SFR = np.zeros([6900,t_index])
        age_index = np.zeros(t_index)
        
        # Now, fine ages which match index.
        Pop_Ages = np.linspace(Age+time*h,Age+0.0001,t_index)
        # age_index = np.zeros(t_index)
        # print(datetime.datetime.now())
        for i in range(t_index):
            age_index[i] = np.where(Pop_Ages[i] <= Spec_Ages)[0][0]

        for i in range(t_index):
            temp_SFR[:,i] = Spectra[:,int(age_index[i])]
            
        SFR_Flux = np.zeros([n,6900])
        for i in range(n):
            for j in range(t_index):
                SFR_Flux[i,:] += Mass[i,j]*temp_SFR[:,j]
                
        return SFR_Flux