# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:07:42 2021

@author: oryan
"""

from numpy import zeros,save

class IOUtil:

  @staticmethod
  def outputParticles(filename,folder,x0,fluxes,SFRs):
    fo = open(filename,'w')
    IOUtil.outputParticlesToFile(fo,folder,x0,fluxes,SFRs)


  @staticmethod
  def outputParticlesToFile(fo,folder,x0,fluxes,SFRs):
    size = len(x0) - 1
    
    
    output = zeros([size,x0.shape[1] + len(fluxes) + 1])
    output[:,:x0.shape[1]] = x0[:size,:].copy()
    
    for i in range(len(fluxes)):
        output[:,x0.shape[1]+i] = fluxes[i].copy()
        
    output[:,-1] = SFRs.copy()
    
    for i in range(size):
      dtmp = output[i]
      for j in range(x0.shape[1]+len(fluxes)):
        fo.write(IOUtil.formatDouble(dtmp[j]))
      # new line 
      fo.write("\n")
  
    fo.close()
    
    save(folder+'results.npy',output)

  @staticmethod
  def formatDouble(num):
    return "%16.8e"%(num)