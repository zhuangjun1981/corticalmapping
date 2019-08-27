# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 15:51:19 2014

@author: junz
"""

import corticalmapping.RetinotopicMapping as rm
import corticalmapping.core.FileTools as ft
import os
import matplotlib.pyplot as plt

plt.ioff()

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

patchesToShow = 'finalPatches'
#patchesToShow = 'finalPatchesMarked'

columnNum = 4

trialList = [
             '160208_M193206_Trial1.pkl',
             ]


for k, trialName in enumerate(trialList):

    trialPath = os.path.join(currFolder,trialName)
    trial, _ = rm.loadTrial(trialPath)
    finalPatches = getattr(trial,patchesToShow)
    numOfPatches = len(list(finalPatches.keys()))
    rowNum = numOfPatches // columnNum + 1
    f = plt.figure(figsize=(10,10))
    f.suptitle(trialName)
    ax = f.add_subplot(111)
    rm.plotPatches(finalPatches,plotaxis=ax,markersize=0)
    
    
    for key,patch in finalPatches.items():
        
        center = patch.getCenter()
        ax.text(center[1],center[0],key,verticalalignment='center', horizontalalignment='center')

plt.show()


