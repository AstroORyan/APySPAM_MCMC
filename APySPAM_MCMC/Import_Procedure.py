# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:12:54 2021

@author: oryan

This algorithm handles all of the imports and exports that APySPAM will require. The only input required is for the SSP 
import which requires the metallicities defined in Setup_Parameters.

Doing it this way allows us to only load once, and to check that the correct folders exist.

"""
# Packages
import os
import numpy as np
import glob

# Local scripts
from SEDs import SED
from IOUtil import IOUtil


class Imports():
    def Filters():
        dir_path = os.getcwd()
        filter_folder = dir_path+'\\Filters\\'
        filters = glob.glob(filter_folder+'*.*')
        
        filter_data = []
        
        for i in range(len(filters)):
            temp = np.loadtxt(filters[i])
            filter_data.append(temp)
            
        return filter_data
    
    def SSPs(metallicity):
        dir_path = os.getcwd()
        metals_folder = dir_path+'\\Spectral_Data\\'
        
        Spectral_Density_Array_1 = SED.Metal_Interpolation(metallicity[0],metals_folder)
        Spectral_Density_Array_2 = SED.Metal_Interpolation(metallicity[1],metals_folder)
        
        Wavelength = Spectral_Density_Array_1[1:,0]
        
        return Spectral_Density_Array_1, Spectral_Density_Array_2, Wavelength
    
    def Results():
        directory = os.getcwd()
        directory_results = directory+'\\Results\\'
        return directory_results
    
    def Export(Vectors,Fluxes,Formed_Stellar_Masses,SFRs,directory_result,Name):
        Formed_Stellar_Masses = np.sum(Formed_Stellar_Masses,1)
        
        x = Fluxes.shape[0]
        y = Vectors.shape[1] + Fluxes.shape[1] + 1 + 1      # Note, the ones here are the y lengths of SFRs and Masses. Like this as using .shape is out of range of 1D arrays.
        Data = np.zeros([x,y])
        
        Vectors = Vectors[:x,:]
        
        Data[:,:Vectors.shape[1]] = Vectors
        Data[:,Vectors.shape[1]:Vectors.shape[1] + Fluxes.shape[1]] =  Fluxes
        Data[:,-1-1] = Formed_Stellar_Masses
        Data[:,-1] = SFRs
        
        # For further Python use, results are saved quickly as a numpy array.
        np.save(directory_result+Name+'.npy', Data)
        
        # For more general use, they are also saved as a .txt array.
        filename = directory_result+Name+'.txt'
        IOUtil.outputParticles(filename,Data)
        
        
    
        