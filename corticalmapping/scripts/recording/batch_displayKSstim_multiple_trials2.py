import matplotlib.pyplot as plt
import corticalmapping.VisualStim as vs


mouseID = 'TEST' #'147861' #'TEST'
userID = 'Jun'
numOfTrials = 2
isTriggered = False

psychopyMonitor = 'testMonitor' #'smartTVgamma'

logFolder = r'C:\data'
backupFolder = r'\\aibsdata2\nc-ophys\CorticalMapping'

mon=vs.MonitorJun(resolution=(1080, 1920),dis=13.5,monWcm=88.8,monHcm=50.1,C2Tcm=33.1,C2Acm=46.4,monTilt=30.,downSampleRate=5)

indicator=vs.IndicatorJun(mon,width_cm=3.,height_cm=3.,position = 'northeast',isSync=True,freq=1.)

stim = vs.KSstimAllDir(mon,indicator,iteration=1,background=0.,squareSize=25.,sweepWidth=20.,stepWidth=0.15,sweepFrame=1,flickerFrame=10,preGapFrame=120,postGapFrame=180)

ds = vs.DisplaySequence(logdir=logFolder,backupdir=backupFolder,displayIteration=numOfTrials,psychopyMonitor=psychopyMonitor,
                        displayOrder=1,mouseid=mouseID,userid=userID,isVideoRecord=True,videoRecordIP='w7dtmj007lhu',videoRecordPort=10000,isTriggered=isTriggered,
                        triggerNIDev='Dev1',triggerNIPort=1,triggerNILine=3,triggerType="NegativeEdge",isSyncPulse=False,syncPulseNIDev='Dev3',syncPulseNIPort=1,
                        syncPulseNILine=2,displayScreen=0,initialBackgroundColor=0.)

ds.setStim(stim)

ds.triggerDisplay()

plt.show()
