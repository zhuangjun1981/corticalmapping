# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 17:18:26 2013

@author: junz
"""

#version 15

import corticalmapping.core.VisualStim as vs
import numpy as np

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
						  
GNstim=vs.GaussianNoise(mon,
                        indicator,
                        tempFreqCeil = 8, # cutoff temporal frequency (Hz)
                        spatialFreqCeil = 0.05, # cutoff spatial frequency (cycle/degree)
                        filterMode = 'box', # type of filter of movie, '1/f' or 'box'
                        sweepSigma=5., # sigma of sweeps (unit same as Map, cm or deg)
                        sweepWidth=10., # width of sweeps (unit same as Map, cm or deg)
                        sweepEdgeWidth = 3., # number of sigmas to smooth the edge of sweeps on each side
                        stepWidth=0.15, # width of steps (unit same as Map, cm or deg)
                        sweepFrame=1, # display frame numbers for each step
                        iteration=5, # time to flash
                        preGapFrame=60, # gap frame number before flash
                        postGapFrame=180, # gap frame number after flash
                        isWarp = False,
                        background = 0.,
                        contrast = 1.,
#                        sequencePath= r'C:\JunZhuang\labwork\data\python_temp_folder',
                        sequencePath = r'\\aibsdata2\nc-ophys\CorticalMapping\sequence_display_log',
                        color = np.array([255,255,255], dtype = np.uint8))

for i in range(10):
    GNstim.generateGaussianNoiseMovie()

GNstim.set_sweepWidth(20.)

for i in range(10):
    GNstim.generateGaussianNoiseMovie()
