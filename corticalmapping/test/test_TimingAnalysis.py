__author__ = 'junz'


import numpy as np
import corticalmapping.core.TimingAnalysis as ta

def test_getOnsetTimeStamps():
    trace = np.array(([0.] * 5 + [5.] * 5) * 5)
    ts = ta.getOnsetTimeStamps(trace, Fs=10000., onsetType='raising')
    assert(ts[2] == 0.0025)
    ts2 = ta.getOnsetTimeStamps(trace, Fs=10000., onsetType='falling')
    assert(ts2[2] == 0.0030)
