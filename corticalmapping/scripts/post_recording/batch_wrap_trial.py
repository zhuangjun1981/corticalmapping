__author__ = 'junz'



import os
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.core.tifffile as tf
from corticalmapping.core.FileTools import loadFile, saveFile
from corticalmapping.core.RetinotopicMapping import generatePhaseMap2,phasePosition




dateRecorded = 20140911 # int, date recorded, yearmonthday
trialNum = '1' # str, number of the trail on that day
# 'Emx1-IRES-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6f)'
# 'Emx1-IRES-Cre;Ai96(GCaMP6s)'
# 'Emx1-IRES-Cre;Ai95(GCaMP6f)'
# 'Scnn1a-Tg3-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6f)'
# 'Rorb-IRES2-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6f)'
mouseType = 'Emx1-IRES-Cre;Camk2a-tTA;Ai93(TITL-GCaMP6f)' # str, mouse Genotype
isAnesthetized = False

FFTmode = 'peak' # 'peak' or 'valley'
altForeNum = '110'
altBackNum = '111'
aziForeNum = '112'
aziBackNum = '113'

powerThr = 0.05 #for plotting, the threshod for power mask
cycleInMovie = 1 #how many cycles in the averaged movie

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trialDict = {}
f = open(__file__,'r')
trialDict.update({'scriptText_wrap': f.read()})
f.close()

if FFTmode == 'peak':
    isReverse = False
elif FFTmode == 'valley':
    isReverse = True


#analyze altFore file
fileDict = loadFile('File' + altForeNum + '.pkl')
altForeDisplayLog = fileDict['displayLog']
vasculatureMap = fileDict['vasculatureMap']
try:
    vasculatureMapEnhanced = fileDict['vasculatureMapEnhanced']
except KeyError:
    pass
mouseID = altForeDisplayLog['presentation']['mouseid']
if altForeDisplayLog['stimulation']['background'] == 0:
    visualStimBackground = 'gray'
elif altForeDisplayLog['stimulation']['background'] == -1:
    visualStimBackground = 'black'
visualStimType = altForeDisplayLog['stimulation']['stimName']
altForeImageExposureTime = fileDict['imageExposureTime']

moviePath = os.path.join(currFolder, 'aveMovieNor_File' + altForeNum + '.tif')
altForePhaseMap, altForePowerMap = generatePhaseMap2(tf.imread(moviePath),cycleInMovie,isReverse = isReverse)
altForePowerMap = altForePowerMap/np.amax(altForePowerMap)
altForePositionMap = phasePosition(altForePhaseMap,altForeDisplayLog)


#analyze altBack File
fileDict = loadFile('File' + altBackNum + '.pkl')
altBackDisplayLog = fileDict['displayLog']
altBackImageExposureTime = fileDict['imageExposureTime']
moviePath = os.path.join(currFolder, 'aveMovieNor_File' + altBackNum + '.tif')
altBackPhaseMap, altBackPowerMap = generatePhaseMap2(tf.imread(moviePath),cycleInMovie,isReverse = isReverse)
altBackPowerMap = altBackPowerMap/np.amax(altBackPowerMap)
altBackPositionMap = phasePosition(altBackPhaseMap,altBackDisplayLog)

#get altPosMap and altPowerMap
altPosMap = np.mean([altForePositionMap,altBackPositionMap],axis=0)
altPowerMap = np.mean([altForePowerMap,altBackPowerMap],axis=0)
altPowerMap = altPowerMap/np.amax(altPowerMap)


#analyze aziFore File
fileDict = loadFile('File' + aziForeNum + '.pkl')
aziForeDisplayLog = fileDict['displayLog']
aziForeImageExposureTime = fileDict['imageExposureTime']
moviePath = os.path.join(currFolder, 'aveMovieNor_File' + aziForeNum + '.tif')
aziForePhaseMap, aziForePowerMap = generatePhaseMap2(tf.imread(moviePath),cycleInMovie,isReverse = isReverse)
aziForePowerMap = aziForePowerMap/np.amax(aziForePowerMap)
aziForePositionMap = phasePosition(aziForePhaseMap,aziForeDisplayLog)

#analyze aziBack File
fileDict = loadFile('File' + aziBackNum + '.pkl')
aziBackDisplayLog = fileDict['displayLog']
aziBackImageExposureTime = fileDict['imageExposureTime']
moviePath = os.path.join(currFolder, 'aveMovieNor_File' + aziBackNum + '.tif')
aziBackPhaseMap, aziBackPowerMap = generatePhaseMap2(tf.imread(moviePath),cycleInMovie,isReverse = isReverse)
aziBackPowerMap = aziBackPowerMap/np.amax(aziBackPowerMap)
aziBackPositionMap = phasePosition(aziBackPhaseMap,aziBackDisplayLog)

#get altPosMap and altPowerMap
aziPosMap = np.mean([aziForePositionMap,aziBackPositionMap],axis=0)
aziPowerMap = np.mean([aziForePowerMap,aziBackPowerMap],axis=0)
aziPowerMap = aziPowerMap/np.amax(aziPowerMap)


trialDict.update({'mouseID':mouseID,
                  'dateRecorded':dateRecorded,
                  'trialNum':trialNum,
                  'mouseType':mouseType,
                  'isAnesthetized':isAnesthetized,
                  'vasculatureMap': vasculatureMap,
                  'visualStimBackground':visualStimBackground,
                  'visualStimType':visualStimType,
                  'imageExposureTime':np.mean([altForeImageExposureTime,altBackImageExposureTime,aziForeImageExposureTime,aziBackImageExposureTime]),
                  'altPosMap':altPosMap,
                  'aziPosMap':aziPosMap,
                  'altPowerMap':altPowerMap,
                  'aziPowerMap':aziPowerMap
                  })

try:trialDict.update({'vasculatureMapEnhanced':vasculatureMapEnhanced})
except NameError:pass

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

f.savefig('Trial_' + trialNum + '.pdf', orientation='landscape',papertype = 'letter')
saveFile('Trial_'+trialNum+'.pkl',trialDict)
