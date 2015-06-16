# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:46:38 2014

@author: junz
"""
import os
import corticalmapping.core.FileTools as ft

trialName = '20150116_M156569_Trial1_2_3_4.pkl'

names = [
         ['patch01', 'V1'],
         ['patch02', 'RL'],
         ['patch03', 'PM'],
         ['patch04', 'LI'],
         ['patch05', 'LM'],
         ['patch06', 'AL'],
         ['patch07', 'MMA'],
#         ['patch08', 'AM'],
         ['patch09', 'P'],
         ['patch10', 'AM'],
         ['patch11', 'LLA'],
         ['patch12', 'MMP'],
#         ['patch13', 'MMP']
         # ['patch14', 'MMP']
         ]

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trialPath = os.path.join(currFolder,trialName)

trialDict = ft.loadFile(trialPath)

finalPatches = dict(trialDict['finalPatches'])

for i, namePair in enumerate(names):
    currPatch = finalPatches.pop(namePair[0])
    newPatchDict = {namePair[1]:currPatch}
    finalPatches.update(newPatchDict)
    
trialDict.update({'finalPatchesMarked':finalPatches})

ft.saveFile(trialPath,trialDict)