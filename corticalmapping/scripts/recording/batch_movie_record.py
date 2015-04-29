# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:06:02 2014

@author: svc_ncbehavior
"""

from aibs.Eyetracking import movierecorder as mr
import atexit

eyeMovieFolder = r'C:\data\eyetracking'
eyeMovieBackupFolder = r'\\aibsdata2\nc-ophys\CorticalMapping\eyetracking'

wcr = mr.WebcamRecorder(saveFolder = eyeMovieFolder,
                        backupFolder = eyeMovieBackupFolder)

params = {
        'cameras':2,
        'size':(640,480),
        'exposure':-6,
        'brightness':100,
        'contrast':100,
        'saturation':100,
        }
print params
atexit.register(wcr.shutdown)
wcr.run(**params)