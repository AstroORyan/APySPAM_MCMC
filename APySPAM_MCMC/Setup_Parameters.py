# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:06:45 2020

@author: oryan
This function is the input script for the APySPAM algorithm. Here, all of the parameters are set into the Parameter object in self. This Parameter
object is defined in the Run.py script as a section of self. self.params will only contain parameters intrinsic to the interaction.

Note, there will be a brief description of each parameter next to it but for a full breakdown of what they do see the MkDocs page for APySPAM
(*Insert link here*)

Inputs: self - The main object which will contain the different branches of variables. Here, the underlying parameters are defined as self.params.variable.

Outputs: None, but self is considered a global object of the class Run.py. Therefore, to call these parameters in main(), use params.variable.

"""
class Setup_Parameters:
  def Starting_Locations(Input_Image,Position,Resolution):      
      x = Position[0]
      y = Position[1]
      z = 0
      vx = 1.5
      vy = 1.5
      vz = 0
      mass1 = 2.5#3.0823#1.6199594298899755#15#10#
      mass2 = 2.5#0.07388#3.2343629373210243#15#10#
      rout1 = 2#(Input_Image.shape[0]*Resolution/15)/2 #10#
      rout2 = 2#(Input_Image.shape[0]*Resolution/15)/2 #0.5569210#10
      phi1 = 45 #90 #
      phi2 = 45 #90 #
      theta1 = 180# 90 #
      theta2 = 180#90 #
      time = -5
      
      return [x,y,z,vx,vy,vz,mass1, mass2, rout1, rout2, phi1, phi2, theta1, theta2, time]