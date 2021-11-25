# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:30:07 2021

@author: oryan

Calculates the SFR at different points of the interaction, and calculates the flux to add onto the tagging particles.

"""
import numpy as np

class SFR_Calculations:
    def SFR(Gas,mass1,mass2,r1,r2,Sep,h,time,Weights,n1,n,init_ages):
        # First, initialise the array of the different population masses and SFRs.
        Population_Mass = np.zeros([n,int(time/h)])
        Age = np.zeros(int(time/h)+1)
        SFRs = np.zeros(n)
        
        # Initialise Constants
        DU = 15
        TU = 87e6   # One time unit is 87Myrs according to ArXiv: 1511.05041
        e_1 = 1.5
        e_2 = 1.5
        counter = 0
        Baryonic_Fraction = (1 + 0.3333)/7.1333
                
        Mass_Ratio_1 = mass2/mass1
        Mass_Ratio_2 = mass1/mass2
        
        # Now, we can calculate the SFR at each particle through each timestep.
        for i in range(Population_Mass.shape[1]):
            Distance_Ratio_1 = r1/Sep[i]
            Distance_Ratio_2 = r2/Sep[i]
            
            Age_1 = init_ages[0] + counter*(h*TU/1e9)
            Age_2 = init_ages[1] + counter*(h*TU/1e9)
            
            Starburst_Enhancement_prim = 1 + 0.25*(Mass_Ratio_1)*(Distance_Ratio_1**2)
            Starburst_Enhancement_sec =  1 + 0.25*(Mass_Ratio_2)*(Distance_Ratio_2**2)
           
            SFR_1_Total = Starburst_Enhancement_prim*((1/(e_1**2))*Age_1*np.exp(-(Age_1)/e_1))*((Baryonic_Fraction*mass1*1e11)/1e9)  # 1e9 here transforms SFR from M_0/Gyr to M_0/yr
            SFR_2_Total = Starburst_Enhancement_sec*((1/(e_2**2))*Age_2*np.exp(-(Age_2)/e_2))*((Baryonic_Fraction*mass2*1e11)/1e9)
            
            SFR_1_Base = ((1/(e_1**2))*Age_1*np.exp(-(Age_1)/e_1))*((Baryonic_Fraction*mass1*1e11)/1e9)
            SFR_2_Base = ((1/(e_2**2))*Age_2*np.exp(-(Age_2)/e_2))*((Baryonic_Fraction*mass2*1e11)/1e9)

            SFRs_1 = SFR_1_Total - SFR_1_Base
            SFRs_2 = SFR_2_Total - SFR_2_Base
            
            if SFRs_1 > 0:
                Population_Mass[:n1,i] += Weights[:n1]*SFRs_1*h*TU
            else:
                pass
            
            if SFRs_2 > 0:
                Population_Mass[n1:,i] += Weights[n1:]*SFRs_2*h*TU
            else:
                pass
            
            counter += 1
                                
                
        SFRs[:n1] = Weights[:n1]*SFRs_1
        SFRs[n1:] = Weights[n1:]*SFRs_2
        
        return SFRs, Population_Mass
        
        