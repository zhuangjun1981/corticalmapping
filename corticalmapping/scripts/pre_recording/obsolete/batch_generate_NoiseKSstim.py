__author__ = 'junz'


import corticalmapping.VisualStim as vs
from corticalmapping.core.FileTools import saveFile
import numpy as np
import os


saveFolder = r'C:\data\NoiseKSstim'

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

KSstim=vs.NoiseKSstimJun(mon,
                         indicator,
                         tempFreqCeil = 8, # cutoff temporal frequency (Hz)
                         spatialFreqCeil = 0.01, # cutoff spatial frequency (cycle/degree)
                         filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                         sweepWidth = 10., # width of sweeps (unit same as Map, cm or deg)
                         sweepSigma=5., # sigma of sweep edges (unit same as Map, cm or deg)
                         sweepEdgeWidth=3., # number of sigmas to smooth the edge of sweeps on each side
                         stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                         isWarp = False, # warp noise or not
                         direction='B2U', # the direction of sweep movement, should be one of "B2U","U2B","L2R","R2L"
                         sweepFrame=1, # display frame numbers for each step
                         coordinate='degree', #'degree' or 'linear'
                         iteration=1,
                         preGapFrame=120, # gap frame number before flash
                         postGapFrame=180,
                         enhanceExp=0.3) # gap frame number after flash


mov,log=KSstim.generateMovie()
np.save(os.path.join(saveFolder,'NoiseKSstim-B2U.npy'),mov)
saveFile(os.path.join(saveFolder,'NoiseKSstim-B2U.pkl'),log)

KSstim.setDirection('U2B')
mov,log=KSstim.generateMovie()
np.save(os.path.join(saveFolder,'NoiseKSstim-U2B.npy'),mov)
saveFile(os.path.join(saveFolder,'NoiseKSstim-U2B.pkl'),log)

KSstim.setDirection('L2R')
mov,log=KSstim.generateMovie()
np.save(os.path.join(saveFolder,'NoiseKSstim-L2R.npy'),mov)
saveFile(os.path.join(saveFolder,'NoiseKSstim-L2R.pkl'),log)

KSstim.setDirection('R2L')
mov,log=KSstim.generateMovie()
np.save(os.path.join(saveFolder,'NoiseKSstim-R2L.npy'),mov)
saveFile(os.path.join(saveFolder,'NoiseKSstim-R2L.pkl'),log)