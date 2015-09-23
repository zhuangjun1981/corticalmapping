import sys
import numpy as np
from psychopy import visual
import corticalmapping.VisualStim as vs


mouseID = 'TEST' #'147861' #'TEST'
userID = 'Jun'
numOfTrials = 2
isTriggered = False

psychopyMonitor = 'testMonitor' #'smartTVgamma'

logFolder = r'C:\data'
backupFolder = r'\\aibsdata2\nc-ophys\CorticalMapping'

mon=vs.MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30.,downSampleRate=20)

indicator=vs.IndicatorJun(mon,width_cm=3.,height_cm=3.,position = 'northeast',isSync=True,freq=1.)           
                          
KSstimB2U=vs.KSstimJun(mon,indicator,iteration=1,direction='B2U',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)
KSstimU2B=vs.KSstimJun(mon,indicator,iteration=1,direction='U2B',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)                 
KSstimL2R=vs.KSstimJun(mon,indicator,iteration=1,direction='L2R',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)
KSstimR2L=vs.KSstimJun(mon,indicator,iteration=1,direction='R2L',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)
                       
movB2U, dictB2U = KSstimB2U.generateMovie()
movU2B, dictU2B = KSstimU2B.generateMovie()      
movL2R, dictL2R = KSstimL2R.generateMovie()
movR2L, dictR2L = KSstimR2L.generateMovie()

mov = np.vstack((movB2U,movU2B,movL2R,movR2L))

stimulation = dict(dictB2U['stimulation'])
stimulation['stimName'] = 'KSstimAllDir'
_ = stimulation.pop('direction')
_ = stimulation.pop('frames')
_ = stimulation.pop('sweepTable')
stimulation['direction'] = ['B2U','U2B','L2R','R2L']
stimulation['frames'] = dictB2U['stimulation']['frames']+dictU2B['stimulation']['frames']+dictL2R['stimulation']['frames']+dictR2L['stimulation']['frames']
stimulation['sweepTable'] = [dictB2U['stimulation']['sweepTable'],dictU2B['stimulation']['sweepTable'],dictL2R['stimulation']['sweepTable'],dictR2L['stimulation']['sweepTable']]
log = {'monitor':movB2U['monitor'],
       'indicator':movB2U['indicator'],
       'stimulation':stimulation}


ds = vs.DisplaySequence(logdir=logFolder,backupdir=backupFolder,displayIteration=numOfTrials,psychopyMonitor=psychopyMonitor,
                        displayOrder=1,mouseid=mouseID,userid=userID,isVideoRecord=True,videoRecordIP='w7dtmj007lhu',videoRecordPort=10000,isTriggered=isTriggered,
                        triggerNIDev='Dev1',triggerNIPort=1,triggerNILine=3,triggerType="NegativeEdge",isSyncPulse=False,syncPulseNIDev='Dev3',syncPulseNIPort=1,
                        syncPulseNILine=2,displayScreen=0,initialBackgroundColor=0.)

ds.setAnyArray(mov,log)

# this is bad, set up the background in between
window = visual.Window(monitor=psychopyMonitor,fullscr=True,screen=0,color=0.)
window.flip()

ds.triggerDisplay()
