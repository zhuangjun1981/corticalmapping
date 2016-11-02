import corticalmapping.VisualStim as vs
import matplotlib.pyplot as plt


mouseID = 'TEST' # '256896' #'TEST'
userID = 'Jun'
numOfTrials = 1 # 20

background = 0.
duration = 600.
color = 0.

logFolder = r'C:\data'



mon=vs.Monitor(resolution=(1080, 1920),
               dis=15.3,
               monWcm=88.8,
               monHcm=50.1,
               C2Tcm=31.1,
               C2Acm=41.91,
               monTilt=26.56,
               downSampleRate=5)
                  
#mon.plot_map()
                          
indicator=vs.Indicator(mon,
                       width_cm=3.,
                       height_cm=3.,
                       position = 'southeast',
                       isSync=True,
                       freq=1.)

UniformContrast = vs.UniformContrast(mon,
                                     indicator,
                                     coordinate='degree',
                                     background=background,
                                     duration=duration,
                                     color=color,
                                     preGapDur=0.,
                                     postGapDur=0.)
                                     
ds = vs.DisplaySequence(logdir=logFolder,
                        backupdir=None,
                        displayIteration=numOfTrials,
                        psychopyMonitor='testMonitor',
                        displayOrder=1,
                        mouseid=mouseID,
                        userid=userID,
                        isInterpolate=False,
                        isRemoteSync=False,
                        remoteSyncIP='localhost',
                        remoteSyncPort=10003,
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
                        displayScreen=0,
                        initialBackgroundColor=0.,
                        isVideoRecord=False,
                        videoRecordIP='w7dtmj007lhu',
                        videoRecordPort=10000,
                        displayControlIP = 'localhost',
                        displayControlPort = 10002,
                        fileNumNIDev = 'Dev1',
                        fileNumNIPort = 0,
                        fileNumNILines = '0:7')

ds.set_stim(UniformContrast)

ds.trigger_display()

plt.show()
