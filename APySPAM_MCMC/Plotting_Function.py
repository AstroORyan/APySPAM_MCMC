# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:22:15 2021

@author: oryan

Quickly creates an image of the system using imshow and a meshgrid.

"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import uuid

class Plotting_Function:
    def plotting(Coordinates,part_flux,SFRs,n_filters,Resolution,dims,Conversion):
        # Define Constants        
        # First, extract the wanted dimensions. 
        x = Coordinates[:,0] + (Conversion[0]*(Resolution))
        y = Coordinates[:,1] + (Conversion[1]*(Resolution))
        
        total_flux = np.zeros(len(x) - 1)
        
        for i in range(n_filters):
            total_flux += part_flux[i]
        
        # Define size of observed space
        Image = np.zeros([dims[0],dims[1]])
        x_min = (-Image.shape[0]/2)*Resolution
        x_max = (Image.shape[0]/2)*Resolution
        
        y_min = (-Image.shape[1]/2)*Resolution
        y_max = (Image.shape[1]/2)*Resolution
        
        x_pixel_value = np.linspace(x_min,x_max,Image.shape[0],endpoint=False)
        y_pixel_value = np.linspace(y_min,y_max,Image.shape[1],endpoint=False)
        
        for i in range(len(x) - 1):
            if x[i] > x_max or x[i] < x_min:
                continue
            elif y[i] > y_max or y[i] < y_min:
                continue
            else:
                p = np.where(x[i] > x_pixel_value)[0][-1]
                q = np.where(y[i] > y_pixel_value)[0][-1]

                Image[q,p] += total_flux[i]
        
        # output_name = str(uuid.uuid4())        
        # plt.figure()
        # plt.imshow(-2.5*np.log10(Image) - 48.6)
        # plt.title('White Image')
        # plt.savefig(f'C:/Users/oryan/Documents/PySPAM_Original_Python_MCMC/APySPAM_MCMC/test-image/{output_name}.jpeg', dpi=100, bbox_layout='tight')
        #sys.exit()
        
        # plt.figure()
        # plt.imshow(-2.5*np.log10(Image_flip) - 48.6)
        # plt.title('Flipped Image')
        # plt.savefig(f'C:/Users/oryan/Documents/PySPAM_Original_Python_MCMC/APySPAM_MCMC/test-image/{output_name}-flip.jpeg', dpi=100, bbox_layout='tight')

        return Image
