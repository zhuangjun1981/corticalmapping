import matplotlib.pyplot as plt
import CorticalMapping.corticalmapping.VisualStim as vs


mouseID = '326981' #'147861' #'TEST'
userID = 'Natalia'
numOfTrials = 40 # 20

logFolder = r'C:\data'

isTriggered = False
isRemoteSync = False
psychopyMonitor = 'smartTVgamma' #'smartTVgamma'
logFolder = r'C:\data'
backupFolder = r'\\W7DTMJ03jgl2\data'
remoteSyncIP = 'w7dtmj19vtx'
remoteSyncPort = 11001
syncOutputFolder = None

mon=vs.Monitor(resolution=(1080, 1920),
               dis=9.5,
               monWcm=52.0,
               monHcm=30.25,
               C2Tcm=15.125,
               C2Acm=25.5,
               monTilt=0.0,
               downSampleRate=5)
                  
#mon.plot_map()
#plt.show()                  
                  
indicator=vs.Indicator(mon,
                       width_cm=6.,
                       height_cm=6.,
                       position = 'southeast',
                       isSync=True,
                       freq=1.)

stim = vs.KSstimAllDir(mon,
                       indicator,
                       background=0.,
                       iteration=1,
                       squareSize=25.,
                       sweepWidth=20.,
                       stepWidth=0.15,
                       sweepFrame=1,
                       flickerFrame=10,
                       preGapDur=2.,
                       postGapDur=3.)

ds = vs.DisplaySequence(logdir=logFolder,
                        backupdir=None,
                        displayIteration=numOfTrials,
                        psychopyMonitor=psychopyMonitor,
                        displayOrder=1,
                        mouseid=mouseID,
                        userid=userID,
                        isInterpolate=False,
                        isRemoteSync=False,
                        remoteSyncIP=remoteSyncIP,
                        remoteSyncPort=remoteSyncPort,
                        remoteSyncTriggerEvent="positiveEdge",
                        remoteSyncSaveWaitTime=5.,
                        isTriggered=False,
                        triggerNIDev='Dev1',
                        triggerNIPort=1,
                        triggerNILine=0,
                        displayTriggerEvent="NegativeEdge",
                        isSyncPulse=False,
                        syncPulseNIDev='Dev1',
                        syncPulseNIPort=1,
                        syncPulseNILine=1,
                        displayScreen=1,
                        initialBackgroundColor=0.,
                        isVideoRecord=False,
                        videoRecordIP='w7dtmj007lhu',
                        videoRecordPort=10000,
                        displayControlIP = 'localhost',
                        displayControlPort = 10002,
                        fileNumNIDev = 'Dev1',
                        fileNumNIPort = 0,
                        fileNumNILines = '0:7')

ds.set_stim(stim)

ds.trigger_display()

plt.show()
