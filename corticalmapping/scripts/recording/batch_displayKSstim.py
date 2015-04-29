


import corticalmapping.core.VisualStim as vs
import matplotlib.pyplot as plt



mon=vs.MonitorJun(resolution=(1080, 1920),
                  dis=13.5,
                  monWcm=88.8,
                  monHcm=50.1,
                  C2Tcm=33.1,
                  C2Acm=46.4,
                  monTilt=16.22,
                  downSampleRate=5)
#mon.plotMap()

indicator=vs.IndicatorJun(mon,
                          width_cm = 3., 
                          height_cm = 3., 
                          position = 'northeast',
                          isSync = True,
                          freq = 1.)
                          
                          
KSstim=vs.KSstimJun(mon,
                    indicator,
                    iteration = 1,
                    direction = "R2L",
                    background = 0.,
                    squareSize=25.,
                    sweepWidth=20.,
                    stepWidth=0.15,
                    sweepFrame=1,
                    flickerFrame=10,
                    preGapFrame=120,
                    postGapFrame=180)

ds= vs.DisplaySequence(
                       logdir=r'C:\data',
                       backupdir=r'\\aibsdata2\nc-ophys\CorticalMapping',
                       displayIteration = 10,
                       displayOrder = 1, # 1: the right order; -1: the reverse order
                       mouseid = 'test',
                       userid = 'Jun',
                       isVideoRecord = True,
                       isTriggered = True,
                       triggerNIDev = 'Dev1',
                       triggerNIPort = 1,
                       triggerNILine = 3,
                       triggerType = "NegativeEdge", # should be one of "NegativeEdge", "PositiveEdge", "HighLevel", or "LowLevel"
                       isSyncPulse = True,
                       syncPulseNIDev = 'Dev3',
                       syncPulseNIPort = 1,
                       syncPulseNILine = 2,
                       displayScreen = 0,
                       initialBackgroundColor = 0)

ds.setStim(KSstim)

ds.triggerDisplay()

plt.show()



                      
