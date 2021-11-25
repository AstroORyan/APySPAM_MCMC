# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:06:44 2021

@author: oryan
"""
import numpy as np

class Gas_Dist:
    def MN_Dist(r1,r2,n1,n,Gas_Mass,x0,Sec_Initial_Coords):
        # First, need to define conversion to physical units.
        DU = 15
        R = np.zeros(n)
        Particle_Weights = np.zeros(n)
        Tracer_Mass = np.zeros(n)
        
        # Then, need to find the radius to each particle in their own reference frames.
        R[:n1] = np.sqrt(x0[:n1,0]*x0[:n1,0] + x0[:n1,1]*x0[:n1,1] + x0[:n1,2]*x0[:n1,2])*DU
        x00 = x0[n1:n,:3] - Sec_Initial_Coords
        R[n1:] = np.sqrt(x00[:,0]*x00[:,0] + x00[:,1]*x00[:,1] + x00[:,2]*x00[:,2])*DU
        
        # Now, need to define scale lengths for each disk.
        a1 = 0.25*r1*DU/1.69
        a2 = 0.25*r2*DU/1.69
        
        b1 = 0.1#0.2*a1
        b2 = 0.1
        
        # Now, can calculate the density distribution of the disk using a Miyamoto-Nagai distribution. These will be used as Weights on the particles:
        Particle_Weights[:n1] = ((b1**2)*(a1*R[:n1]**2 + (a1 + 3*(b1**2)**0.5)*(a1 + (b1**2)**0.5)**2))/(4*np.pi*((R[:n1]**2 + (a1 + (b1**2)**0.5)**2)**(5/2))*(b1**2)**(3/2))
        Particle_Weights[n1:] = ((b2**2)*(a2*R[n1:]**2 + (a2 + 3*(b2**2)**0.5)*(a2 + (b2**2)**0.5)**2))/(4*np.pi*((R[n1:]**2 + (a2 + (b2**2)**0.5)**2)**(5/2))*(b2**2)**(3/2))
        
        # Normalise as they are weights
        Particle_Weights[:n1] /= np.sum(Particle_Weights[:n1])
        Particle_Weights[n1:] /= np.sum(Particle_Weights[n1:])        
        
        # Use to get Gas Mass Distribution
        Tracer_Mass[:n1] = Particle_Weights[:n1]*Gas_Mass[0]
        Tracer_Mass[n1:] = Particle_Weights[n1:]*Gas_Mass[1]
        
        return Particle_Weights, Tracer_Mass