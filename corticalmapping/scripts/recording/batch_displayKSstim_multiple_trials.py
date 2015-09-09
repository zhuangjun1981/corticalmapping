import os
import sys
from PIL import Image
from psychopy import visual
import corticalmapping.VisualStim as vs


mouseID = 'TEST' #'147861' #'TEST'
userID = 'Jun'
numOfTrials = 2
isTriggered = False

psychopyMonitor = 'smartTVgamma'

logFolder = r'C:\data'
backupFolder = r'\\aibsdata2\nc-ophys\CorticalMapping'

mon=vs.MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30.,downSampleRate=5)

indicator=vs.IndicatorJun(mon,width_cm=3.,height_cm=3.,position = 'northeast',isSync=True,freq=1.)           
                          
KSstimB2U=vs.KSstimJun(mon,indicator,iteration=1,direction='B2U',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)
KSstimU2B=vs.KSstimJun(mon,indicator,iteration=1,direction='U2B',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)                 
KSstimL2R=vs.KSstimJun(mon,indicator,iteration=1,direction='L2R',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)
KSstimR2L=vs.KSstimJun(mon,indicator,iteration=1,direction='R2L',background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)
                       
movB2U, dictB2U = KSstimB2U.generateMovie()
movU2B, dictU2B = KSstimU2B.generateMovie()      
movL2R, dictL2R = KSstimL2R.generateMovie()
movR2L, dictR2L = KSstimR2L.generateMovie()

dsTriggered= vs.DisplaySequence(logdir=logFolder,backupdir=backupFolder,displayIteration=10,psychopyMonitor=psychopyMonitor,
                                displayOrder=1,mouseid=mouseID,userid=userID,isVideoRecord=True,videoRecordIP='w7dtmj007lhu',videoRecordPort=10000,isTriggered=True,
                                triggerNIDev='Dev1',triggerNIPort=1,triggerNILine=3,triggerType="NegativeEdge",isSyncPulse=True,syncPulseNIDev='Dev3',syncPulseNIPort=1,
                                syncPulseNILine=2,displayScreen=0,initialBackgroundColor=0.)
                                
dsNotTriggered= vs.DisplaySequence(logdir=logFolder,backupdir=backupFolder,displayIteration=10,psychopyMonitor=psychopyMonitor,
                                   displayOrder=1,mouseid=mouseID,userid=userID,isVideoRecord=True,videoRecordIP='w7dtmj007lhu',videoRecordPort=10000,isTriggered=False,
                                   triggerNIDev='Dev1',triggerNIPort=1,triggerNILine=3,triggerType="NegativeEdge",isSyncPulse=True,syncPulseNIDev='Dev3',syncPulseNIPort=1,
                                   syncPulseNILine=2,displayScreen=0,initialBackgroundColor=0.)


# this is bad, set up the background in between
window = visual.Window(monitor=psychopyMonitor,fullscr=True,screen=0,color=0.)
window.flip()

# display stimulus
if not isTriggered:
    for i in range(numOfTrials):
        dsNotTriggered.setAnyArray(movB2U, dictB2U)
        completed = dsNotTriggered.triggerDisplay()
        if not completed: sys.exit()
            
        dsNotTriggered.setAnyArray(movU2B, dictU2B)
        completed = dsNotTriggered.triggerDisplay()
        if not completed: sys.exit()
        
        dsNotTriggered.setAnyArray(movL2R, dictL2R)
        completed = dsNotTriggered.triggerDisplay()
        if not completed: sys.exit()
        
        dsNotTriggered.setAnyArray(movR2L, dictR2L)
        completed = dsNotTriggered.triggerDisplay()
        if not completed: sys.exit()
        
else:
    dsTriggered.setAnyArray(movB2U, dictB2U)
    completed = dsTriggered.triggerDisplay()
    if not completed: sys.exit()
    
    dsNotTriggered.setAnyArray(movU2B, dictU2B)
    completed = dsNotTriggered.triggerDisplay()
    if not completed: sys.exit()
    
    dsNotTriggered.setAnyArray(movL2R, dictL2R)
    completed = dsNotTriggered.triggerDisplay()
    if not completed: sys.exit()

    dsNotTriggered.setAnyArray(movR2L, dictR2L)
    completed = dsNotTriggered.triggerDisplay()
    if not completed: sys.exit()
    
    if numOfTrials > 1:
        for i in range(numOfTrials-1):
            dsNotTriggered.setAnyArray(movB2U, dictB2U)
            completed = dsNotTriggered.triggerDisplay()
            if not completed: sys.exit()
            
            dsNotTriggered.setAnyArray(movU2B, dictU2B)
            completed = dsNotTriggered.triggerDisplay()
            if not completed: sys.exit()
                
            dsNotTriggered.setAnyArray(movL2R, dictL2R)
            completed = dsNotTriggered.triggerDisplay()
            if not completed: sys.exit()
                
            dsNotTriggered.setAnyArray(movR2L, dictR2L)
            completed = dsNotTriggered.triggerDisplay()
            if not completed: sys.exit()
