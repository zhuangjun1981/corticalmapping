import corticalmapping.VisualStim as vs
import matplotlib.pyplot as plt



mon=vs.MonitorJun(resolution=(1080, 1920),
                  dis=13.5,
                  monWcm=88.8,
                  monHcm=50.1,
                  C2Tcm=33.1,
                  C2Acm=46.4,
                  monTilt=30.,
                  downSampleRate=5)
#mon.plotMap()

indicator=vs.IndicatorJun(mon,
                          width_cm=3.,
                          height_cm=3.,
                          position='northeast',
                          isSync=True,
                          freq=1.)


SparseNoise=vs.SparseNoise(mon,
                           indicator,
                           coordinate='degree', #'degree' or 'linear'
                           background=0., #back ground color [-1,1]
                           gridSpace=(10.,10.), #(alt,azi)
                           probeSize=(10.,10.), #size of flicker probes (width,height)
                           probeOrientation=0., #orientation of flicker probes
                           probeFrameNum=6,
                           subregion=[-15, 50, -20, 70],
                           sign='ON-OFF',
                           iteration=1,
                           preGapFrame=0,
                           postGapFrame=0)

ds=vs.DisplaySequence(
                      logdir=r'C:\data',
                      backupdir=r'\\aibsdata2\nc-ophys\CorticalMapping',
                      displayIteration=10,
                      displayOrder=1,
                      mouseid='test',
                      userid='Jun',
                      isVideoRecord=False,
                      videoRecordIP='localhost',
                      videoRecordPort='10000',
                      isTriggered=False,
                      triggerNIDev='Dev1',
                      triggerNIPort=1,
                      triggerNILine=3,
                      triggerType="NegativeEdge",
                      isSyncPulse=False,
                      syncPulseNIDev='Dev3',
                      syncPulseNIPort=1,
                      syncPulseNILine=2,
                      displayScreen=0,
                      initialBackgroundColor=0)

ds.setStim(SparseNoise)

ds.triggerDisplay()

plt.show()
