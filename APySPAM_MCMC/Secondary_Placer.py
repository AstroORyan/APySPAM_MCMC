# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:41:15 2021

@author: oryan

This is a little experimental script. It's aim is to find the position of the secondary galaxy. What it will do, is move a 10x10 grid about a binary
image of the input. It will look for the 2 places where this is minimum and use that as the x and y position. We can then use this to define a very
narrow prior for the secondary galaxy x and y positions.

"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import sys

def Distance_Calc(z):
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
      
      return D_L
  
def Prim_Remover(arr,shape,center):
    limit_x_low = int(center[0] - 2)
    limit_x_up = int(center[0] + 2)
    limit_y_low = int(center[1] - 2)
    limit_y_up = int(center[1] + 2)
    
    if limit_x_low < 0:
        limit_x_low = 0
    if limit_x_up > shape:
        limit_x_up = shape
    if limit_y_low < 0:
        limit_y_low = 0
    if limit_y_up > shape:
        limit_y_up = shape
          
    arr[limit_x_low:limit_x_up,limit_y_low:limit_y_up] = 1e6
    
    return arr

def Flux_Selector(input_image,position_x,position_y,step_size):
    total_flux = np.zeros(len(position_x))
    for i in range(len(position_x)):
        limit_x_low = position_x[i] - step_size
        limit_x_up = position_x[i] + step_size
        limit_y_low = position_y[i] - step_size
        limit_y_up = position_y[i] + step_size
        
        if limit_x_low < 0:
            limit_x_low = 0
        if limit_x_up > input_image.shape[0]:
            limit_x_up = input_image.shape[0]
        if limit_y_low < 0:
            limit_y_low = 0
        if limit_y_up > input_image.shape[1]:
            limit_y_up = input_image.shape[1]
            
        total_flux[i] = np.sum(input_image[limit_y_low:limit_y_up,limit_x_low:limit_x_up])
    
    index = np.where(total_flux == np.max(total_flux))    
    return position_x[index],position_y[index],index
        


def Secondary_Placer(Input_Image,Input_Image_Binary,z,block_reduce,Name):
    # Testing Temp Lines
    DU = 15
    # block_reduce = 6
    cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.275 * u.K, Om0 = 0.308)
    d = np.float64(cosmo.luminosity_distance(z)) * 1e3
    Resolution = np.float64(cosmo.kpc_proper_per_arcmin(z)) * (0.396 / 60) * block_reduce
    
    # Then, define the 10x10 grid of ones
    if Input_Image.shape[0] == 50:
        Filter_Size = 6
    elif Input_Image.shape[0] == 100:
        Filter_Size = 10
        
    Step_Size = int(Filter_Size/2)
    Filter = np.ones([Filter_Size,Filter_Size])
    
    # Create an empty list of X^2 which will correspond to bin positions.
    Chi_Squared = np.zeros([int(Input_Image_Binary.shape[0]/Filter_Size),int(Input_Image_Binary.shape[1]/Filter_Size)])
    
    # Set up Loop through Main Image
    Filter_Position = [Step_Size,Step_Size]
    Final_x_Position = Input_Image_Binary.shape[0] - Step_Size
    Final_y_Position = Input_Image_Binary.shape[1] - Step_Size
    i,j = [0,0]
    
    # Main for loop through the image
    while Filter_Position[0] <= Final_x_Position:
        while Filter_Position[1] <= Final_y_Position:
            Test_Area = Input_Image_Binary[Filter_Position[0]-Step_Size:Filter_Position[0]+Step_Size,Filter_Position[1]-Step_Size:Filter_Position[1]+Step_Size]
            Chi_Squared[i,j] = np.sum((Test_Area - Filter)**2)
            j += 1
            Filter_Position[1] += Filter_Size
        i += 1
        j = 0
        Filter_Position[0] += Filter_Size
        Filter_Position[1] = Step_Size
    
    # Extract the two min values
    Prim_Min = np.min(np.min(Chi_Squared))
    
    Position_x,Position_y = np.where(Chi_Squared == Prim_Min)
    Position_x_im,Position_y_im,index = Flux_Selector(Input_Image,Step_Size + Position_x*Filter_Size,Step_Size + Position_y*Filter_Size,Step_Size)
    Chi_Squared = Prim_Remover(Chi_Squared,Input_Image_Binary.shape[0],[Position_x[index],Position_y[index]])
    Position_x_im = Position_x_im[0]
    Position_y_im = Position_y_im[0]
    Position = [Position_x_im,Position_y_im]   #[Step_Size + Position_x*Filter_Size,Step_Size + Position_y*Filter_Size]
    
    # Create Primary Disk cutout
    Limit_x = [Position[1] - Step_Size, Position[1] + Step_Size]
    Limit_y = [Position[0] - Step_Size, Position[0] + Step_Size]
    
    if Limit_x[0] < 0:
        Limit_x[0] = 0
    if Limit_x[1] > Input_Image_Binary.shape[0]:
        Limit_x[1] = Input_Image_Binary.shape[0]        
    if Limit_y[0] < 0:
        Limit_y[0] = 0
    if Limit_y[1] > Input_Image_Binary.shape[0]:
        Limit_y[1] = Input_Image_Binary.shape[0]
    
    # Cutout = Input_Image[Limit_x[0]:Limit_x[1],Limit_y[0]:Limit_y[1]]
    # maxi = np.max(np.max(Cutout))
    # Adjust_x,Adjust_y = np.where(Cutout == maxi)
    # Adjust_x = Adjust_x[0] - Step_Size
    # Adjust_y = Adjust_y[0] - Step_Size
    Position_Prim = [Position[0], Position[1]]
    
    # Now, must get the Secondary Disk Position
    # Chi_Squared[np.where(Chi_Squared == Prim_Min)] = 1e6
    Sec_Min = np.min(np.min(Chi_Squared))
    
    Position_x,Position_y = np.where(Chi_Squared == Sec_Min)
    Position_x_im,Position_y_im,index = Flux_Selector(Input_Image,Step_Size + Position_x*Filter_Size,Step_Size + Position_y*Filter_Size,Step_Size)    
    Position_x_im = Position_x_im[0]
    Position_y_im = Position_y_im[0]
    Position = [Position_x_im,Position_y_im]  #[Step_Size + Position_x*Filter_Size,Step_Size + Position_y*Filter_Size]
    
    # Create Secondaru Disk cutout
    Limit_x = [Position[1] - Step_Size, Position[1] + Step_Size]
    Limit_y = [Position[0] - Step_Size, Position[0] + Step_Size]
    
    if Limit_x[0] < 0:
        Limit_x[0] = 0
    if Limit_x[1] > Input_Image_Binary.shape[0]:
        Limit_x[1] = Input_Image_Binary.shape[0]        
    if Limit_y[0] < 0:
        Limit_y[0] = 0
    if Limit_y[1] > Input_Image_Binary.shape[0]:
        Limit_y[1] = Input_Image_Binary.shape[0]
    
    # Cutout = Input_Image[Limit_x[0]:Limit_x[1],Limit_y[0]:Limit_y[1]]
    # maxi = np.max(np.max(Cutout))
    # Adjust_x,Adjust_y = np.where(Cutout == maxi)
    # Adjust_x = Adjust_x[0] - Step_Size
    # Adjust_y = Adjust_y[0] - Step_Size
    Position_Sec = [Position[0], Position[1]]

    # Convert bin positions into Primaries Frame
    Conversion = [Position_Prim[0] - Input_Image.shape[0]/2, Position_Prim[1] - Input_Image.shape[1]/2]
    
    # Check Positions are Correct
    plt.figure()
    plt.imshow(Input_Image)
    plt.scatter(Position_Prim[1],Position_Prim[0])
    plt.scatter(Position_Sec[1],Position_Sec[0])
    plt.legend(['Primary','Secondary'])
    plt.savefig(r'/mmfs1/home/users/oryan/PySPAM_Original_Python_MCMC_Full/Test_Images/'+Name+'.png')
    plt.close()
        
    # Convert into Galaxy Unit Distances and into the frame of the Primary 
    Position[0] = ((Position_Sec[0] - Position_Prim[0]))*(Resolution/DU)
    Position[1] = ((Position_Sec[1] - Position_Prim[1]))*(Resolution/DU)

    print(f'Position_Prim = [{Position_Prim[0]*(Resolution/DU)} , {Position_Prim[1]*(Resolution/DU)}]')
    print(f'Position_Sec = [{Position_Sec[0]*(Resolution/DU)} , {Position_Sec[1]*(Resolution/DU)}]')
    print(f'Conversion = [{Conversion[0]*(Resolution/DU)}, {Conversion[1]*(Resolution/DU)}')
    print(f'Converted_Prim = [{(Position_Prim[0] - Conversion[0])*(Resolution/DU)}, {(Position_Prim[1] - Conversion[1])*(Resolution/DU)}]')
    print(f'Converted_Sec = [{Position[0]},{Position[1]}]')
    
    # Check Orientation:
#    if Position_Sec[0] > Position_Prim[0]:
#        Position[0] = -Position[0]
#    if Position_Sec[1] > Position_Prim[1]:
#        Position[1] = -Position[1]
    
    # Prepare Stuff to export:
    x = Position[0]
    y = Position[1]

    # Return
    return Conversion,[x,y], [x-0.1,x+0.1,y-0.1,y+0.1],Resolution