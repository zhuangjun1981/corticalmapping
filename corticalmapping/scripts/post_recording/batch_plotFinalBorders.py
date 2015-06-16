# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:03:36 2015

@author: junz
"""

import corticalmapping.core.RetinotopicMapping as rm
import os
import matplotlib.pyplot as plt
import corticalmapping.core.tifffile as tf

plt.ioff()

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trialList = [
             '20150508_M173310_Trial1_2_3_4.pkl',
             ]

             
for trialName in trialList: 

    trialPath = os.path.join(currFolder,trialName)
    trial, _ = rm.loadTrial(trialPath)
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    trial.plotFinalPatchBorders(plotAxis = ax,borderWidth=4)
    plt.show()
    f.savefig(trialName[0:-4]+'_borders.pdf',dpi=600)
    f.savefig(trialName[0:-4]+'_borders.png',dpi=300)
    
#    tf.imsave(trialName[0:-4]+'_vasculatureMap.tif',trial.vasculatureMap)
    
#    f2 = plt.figure(figsize=(10,10))
#    ax2 = f2.add_subplot(111)
#    ax2.imshow(trial.vasculatureMap,cmap='gray',interpolation='nearest')
#    ax2.set_axis_off()
#    ax2.set_title(trial.getName())
#    f2.savefig(trialName[0:-4]+'_vasculatureMap.pdf',dpi=600)
#    
#plt.show()
    
    