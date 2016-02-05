# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:23:59 2014

@author: junz
"""

from corticalmapping.core.FileTools import loadFile, saveFile
from corticalmapping.RetinotopicMapping import phasePosition,generatePhaseMap2
from corticalmapping.core.ImageAnalysis import normalize_movie
import corticalmapping.core.tifffile as tf
import os
import numpy as np
import matplotlib.pyplot as plt


trialList = ['1','2','3','4'] # list of trials to average

isSave = True

altForeNumList = ['110','115','119','123'] #list of bottom up movie numbers
altBackNumList = ['111','116','120','124'] #list of top down movie numbers 
aziForeNumList = ['113','117','121','125'] #list of anterior posterior movie numbers
aziBackNumList = ['114','118','122','127'] #list of posterior anterior movie numbers

FFTmode = 'peak' # 'peak' or 'valley'

preGapDur = 2. # second, pre-gap duration
powerThr = 0.05 #for plotting, the threshod for power mask
cycleInMovie = 1 #how many cycles in the averaged movie

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

plt.ioff()

trialDict = {}
f = open(__file__,'r')
trialDict.update({'scriptText_avearage': f.read()})
f.close()

if FFTmode == 'peak':
    isReverse = False
elif FFTmode == 'valley':
    isReverse = True

scriptText_wrap = []
imageExposureTime = []
trialNum = ''

movB2U=[]
movU2B=[]
movA2P=[]
movP2A=[]
    
for i, trialName in enumerate(trialList):
    
    currTrialDict = loadFile('Trial_'+trialName+'.pkl')
    
    if i == 0:
        trialDict.update({
                          'dateRecorded':currTrialDict['dateRecorded'],
                          'isAnesthetized':currTrialDict['isAnesthetized'],
                          'mouseID':currTrialDict['mouseID'],
                          'mouseType':currTrialDict['mouseType'],
                          'vasculatureMap':currTrialDict['vasculatureMap'],
                          'visualStimBackground':currTrialDict['visualStimBackground'],
                          'visualStimType':currTrialDict['visualStimType'],
                          })
        trialNum = trialNum + trialName
        
        fileDict = loadFile('File' + altForeNumList[i] + '.pkl')
        altForeDisplayLog = fileDict['displayLog']  
        
        fileDict = loadFile('File' + altBackNumList[i] + '.pkl')
        altBackDisplayLog = fileDict['displayLog'] 
        
        fileDict = loadFile('File' + aziForeNumList[i] + '.pkl')
        aziForeDisplayLog = fileDict['displayLog'] 
        
        fileDict = loadFile('File' + aziBackNumList[i] + '.pkl')
        aziBackDisplayLog = fileDict['displayLog'] 
        
    else:
        trialNum = trialNum + '_' + trialName
    
    scriptText_wrap.append(currTrialDict['scriptText_wrap'])
    imageExposureTime.append(currTrialDict['imageExposureTime'])
    
    currAltForeMov = tf.imread('aveMovie_File' + altForeNumList[i] + '.tif')
    movB2U.append(currAltForeMov)
    
    currAltBackMov = tf.imread('aveMovie_File' + altBackNumList[i] + '.tif')
    movU2B.append(currAltBackMov)
    
    currAziForeMov = tf.imread('aveMovie_File' + aziForeNumList[i] + '.tif')
    movA2P.append(currAziForeMov)
    
    currAziBackMov = tf.imread('aveMovie_File' + aziBackNumList[i] + '.tif')
    movP2A.append(currAziBackMov)
    
trialDict.update({
                  'trialNum':trialNum,
                  'scriptText_wrap':scriptText_wrap,
                  'imageExposureTime':np.mean(imageExposureTime,axis=0)
                  })
                  
altForeMov = np.mean(movB2U,axis=0)
altBackMov = np.mean(movU2B,axis=0)
aziForeMov = np.mean(movA2P,axis=0)
aziBackMov = np.mean(movP2A,axis=0)

preGapFrameNum = int(np.round(preGapDur/np.mean(imageExposureTime,axis=0)))

_,altForeMovNor,_ = normalize_movie(altForeMov, baselinePic=np.mean(altForeMov[0:preGapFrameNum, :, :], axis=0))
_,altBackMovNor,_ = normalize_movie(altBackMov, baselinePic=np.mean(altBackMov[0:preGapFrameNum, :, :], axis=0))
_,aziForeMovNor,_ = normalize_movie(aziForeMov, baselinePic=np.mean(aziForeMov[0:preGapFrameNum, :, :], axis=0))
_,aziBackMovNor,_ = normalize_movie(aziBackMov, baselinePic=np.mean(aziBackMov[0:preGapFrameNum, :, :], axis=0))

altForePhaseMap, altForePowerMap = generatePhaseMap2(altForeMovNor,cycleInMovie,isReverse = isReverse)
altForePowerMap = altForePowerMap/np.amax(altForePowerMap)
altForePositionMap = phasePosition(altForePhaseMap,altForeDisplayLog)
    
altBackPhaseMap, altBackPowerMap = generatePhaseMap2(altBackMovNor,cycleInMovie,isReverse = isReverse)
altBackPowerMap = altBackPowerMap/np.amax(altBackPowerMap)
altBackPositionMap = phasePosition(altBackPhaseMap,altBackDisplayLog)
    
aziForePhaseMap, aziForePowerMap = generatePhaseMap2(aziForeMovNor,cycleInMovie,isReverse = isReverse)
aziForePowerMap = aziForePowerMap/np.amax(aziForePowerMap)
aziForePositionMap = phasePosition(aziForePhaseMap,aziForeDisplayLog)
    
aziBackPhaseMap, aziBackPowerMap = generatePhaseMap2(aziBackMovNor,cycleInMovie,isReverse = isReverse)
aziBackPowerMap = aziBackPowerMap/np.amax(aziBackPowerMap)
aziBackPositionMap = phasePosition(aziBackPhaseMap,aziBackDisplayLog)
    
#get altPosMap and altPowerMap
altPosMap = np.mean([altForePositionMap,altBackPositionMap],axis=0)
altPowerMap = np.mean([altForePowerMap,altBackPowerMap],axis=0)
altPowerMap = altPowerMap/np.amax(altPowerMap)    
    
#get altPosMap and altPowerMap
aziPosMap = np.mean([aziForePositionMap,aziBackPositionMap],axis=0)
aziPowerMap = np.mean([aziForePowerMap,aziBackPowerMap],axis=0)
aziPowerMap = aziPowerMap/np.amax(aziPowerMap)

trialDict.update({
                  'altPosMap':altPosMap,
                  'aziPosMap':aziPosMap,
                  'altPowerMap':altPowerMap,
                  'aziPowerMap':aziPowerMap
                  })
                  
f = plt.figure()
f.suptitle('Trial_'+trialNum)

plt.subplot(221)
plt.imshow(altPosMap, vmin=-30,vmax=50,cmap='hsv',interpolation='nearest')
plt.colorbar()
plt.title('alt position map')

plt.subplot(222)
plt.imshow(altPowerMap, vmin=0,vmax=1,cmap='hot',interpolation='nearest')
plt.colorbar()
plt.title('alt power map')

plt.subplot(223)
plt.imshow(aziPosMap, vmin=0,vmax=120,cmap='hsv',interpolation='nearest')
plt.colorbar()
plt.title('azi position map')

plt.subplot(224)
plt.imshow(aziPowerMap, vmin=0,vmax=1,cmap='hot',interpolation='nearest')
plt.colorbar()
plt.title('alt power map')

plt.show()
                  
if isSave:
    f.savefig('Trial_' + trialNum + '.pdf', orientation='landscape',papertype = 'letter')
    saveFile('Trial_'+trialNum+'.pkl',trialDict)
    tf.imsave('altForeMov' + trialNum + '.tif',altForeMov)
    tf.imsave('altForeMovNor' + trialNum + '.tif',altForeMovNor)
    tf.imsave('altBackMov' + trialNum + '.tif',altBackMov)
    tf.imsave('altBackMovNor' + trialNum + '.tif',altBackMovNor)
    tf.imsave('aziForeMov' + trialNum + '.tif',aziForeMov)
    tf.imsave('aziForeMovNor' + trialNum + '.tif',aziForeMovNor)
    tf.imsave('aziBackMov' + trialNum + '.tif',aziBackMov)
    tf.imsave('aziBackMovNor' + trialNum + '.tif',aziBackMovNor)
    