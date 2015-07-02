__author__ = 'junz'


import os
import matplotlib.pyplot as plt

from corticalmapping.core.FileTools import loadFile, saveFile
from corticalmapping.RetinotopicMapping import RetinotopicMappingTrial


trialNum = '1_2_3_4'

isSave = True

A2PMovName = 'aziForeMov1_2_3_4.tif'

params = {'phaseMapFilterSigma': 1.,
          'signMapFilterSigma': 9.,
          'signMapThr': 0.3,
          'eccMapFilterSigma': 15.0,
          'splitLocalMinCutStep': 10.,
          'closeIter': 3,
          'openIter': 3,
          'dilationIter': 15,
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1,
          'mergeOverlapThr': 0.1
          }

currFolder = os.path.dirname(os.path.realpath(__file__))
os.chdir(currFolder)

trialPath = 'Trial_' + trialNum + '.pkl'
preTrialDict = loadFile(trialPath)

plt.ioff()

trialObj = RetinotopicMappingTrial(mouseID = preTrialDict['mouseID'], # str, mouseID
                                   dateRecorded = preTrialDict['dateRecorded'], # int, date recorded, yearmonthday
                                   trialNum = preTrialDict['trialNum'], # int, number of the trail on that day
                                   mouseType = preTrialDict['mouseType'], # str, mouse Genotype
                                   visualStimType = preTrialDict['visualStimType'], # str, stimulation type
                                   visualStimBackground = preTrialDict['visualStimBackground'], # str, background of visual stimulation
                                   imageExposureTime = preTrialDict['imageExposureTime'], # exposure time of the recorded image
                                   altPosMap = preTrialDict['altPosMap'], # altitute position map
                                   aziPosMap = preTrialDict['aziPosMap'], # azimuth position map
                                   altPowerMap = preTrialDict['altPowerMap'], # altitude power map
                                   aziPowerMap = preTrialDict['aziPowerMap'], # azimuth power map
                                   vasculatureMap = preTrialDict['vasculatureMap'], # vasculature map
                                   isAnesthetized = preTrialDict['isAnesthetized'],
                                   params = params # parameters for imaging analysis
                                   )

_ = trialObj._getSignMap(isPlot=True)
plt.show()
_=trialObj._getRawPatchMap(isPlot=True)
plt.show()
_=trialObj._getRawPatches(isPlot=True)
plt.show()
_=trialObj._getDeterminantMap(isPlot=True)
plt.show()
_=trialObj._getEccentricityMap(isPlot=True)
plt.show()
_=trialObj._splitPatches(isPlot=True)
plt.show()
_=trialObj._mergePatches(isPlot=True)
plt.show()

try:
    traces = trialObj.getTraces(moviePath=A2PMovName,
                                resampleFrequency = 10, # at which frequency the traces are resampled
                                centerPatch = 'patch01', # the patch to get ROI
                                ROIcenters = [[30.,0.],[60.,0.],[90.,0.]], # visual space centers of ROIs
                                ROIsearchRange = 0.5, #range to search pixels in ROI
                                ROIsize = 10, # ROI size (pixel)
                                ROIcolor = ['#ff0000','#00ff00','#0000ff'],#color for each ROI
                                isPlot = True)
except Exception:
    print 'Can not generate traces!!'
    traces = None
                                

plt.show()

#trialDict = trialObj.generateTrialDict(keysToRetain = ('altPosMap',
#                                                       'aziPosMap',
#                                                       'finalPatches',
#                                                       'mouseID',
#                                                       'dateRecorded',
#                                                       'trialNum',
#                                                       'mouseType',
#                                                       'visualStimType',
#                                                       'visualStimBackground',
#                                                       'imageExposureTime',
#                                                       'altPowerMap',
#                                                       'aziPowerMap',
#                                                       'vasculatureMap',
#                                                       'params',
#                                                       'isAnesthetized'))

trialDict = trialObj.generateTrialDict()

trialDict.update({'traces':traces})

trialObj.plotTrial(isSave=isSave,saveFolder=currFolder)

plt.show()

#trialObj.plotTrial()

if isSave:
    saveFile(str(trialDict['dateRecorded'])+'_M'+trialDict['mouseID']+'_Trial'+trialNum+'.pkl',trialDict)
