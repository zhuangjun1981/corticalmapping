__author__ = 'junz'

import numpy as np
import matplotlib.pyplot as plt

def up_crossings(data, threshold=0):
    pos = data > threshold
    return (~pos[:-1] & pos[1:]).nonzero()[0]

def down_crossings(data, threshold=0):
    pos = data > threshold
    return (pos[:-1] & ~pos[1:]).nonzero()[0]

def all_crossings(data, threshold=0):
    pos = data > threshold
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

def thresholdOnset(data, threshold=0, direction='up', fs=10000.):
    '''

    :param data: time trace
    :param threshold: threshold value
    :param direction: 'up', 'down', 'both'
    :param fs: sampling rate
    :return: timing of each crossing
    '''

    if direction == 'up': onsetInd = up_crossings(data, threshold)
    elif direction == 'down': onsetInd = down_crossings(data, threshold)
    elif direction == 'both': onsetInd = all_crossings(data, threshold)
    return onsetInd/float(fs)

def discreteCrossCorrelation(ts1,ts2,range=(-1.,1.),bins=100, isPlot=False):

    binWidth = (float(range[1])-float(range[0]))/bins
    t = np.arange((range[0]+binWidth/2),(range[1]+binWidth/2),binWidth)
    intervals = list(np.array([(t-binWidth/2),(t+binWidth/2)]).transpose())
    values = np.zeros(bins)

    for ts in list(ts1):
        currIntervals = [x + ts for x in intervals]
        for i, interval in enumerate(currIntervals):
            values[i] += len(np.where(np.logical_and(ts2>interval[0],ts2<=interval[1]))[0])

    if isPlot:
        f = plt.figure(figsize=(15,4)); ax = f.add_subplot(111)
        ax.bar([a[0] for a in intervals],values,binWidth*0.9);ax.set_xticks(t)

    return t,values

def findNearest(trace,value):
    '''
    return the index in "trace" having the closest value to "value"
    '''

    return np.argmin(np.abs(trace-value))

def getOnsetTimeStamps(trace, Fs=10000., threshold = 3., onsetType='raising'):
    '''
    param trace: time trace of digital signal recorded as analog
    param Fs: sampling rate
    return onset time stamps
    '''

    pos = trace > threshold
    if onsetType == 'raising':
        return ((~pos[:-1] & pos[1:]).nonzero()[0]+1)/float(Fs)
    if onsetType == 'falling':
        return ((pos[:-1] & ~pos[1:]).nonzero()[0]+1)/float(Fs)




if __name__=='__main__':

    #============================================================================================================
    # a=np.arange(100,dtype=np.float)
    # b=a+0.5+(np.random.rand(100)-0.5)*0.1
    # c=discreteCrossCorrelation(a,b,range=(0,1),bins=50,isPlot=True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    trace = np.array(([0.] * 5 + [5.] * 5) * 5)
    ts = getOnsetTimeStamps(trace, Fs=10000., onsetType='raising')
    assert(ts[2] == 0.0025)
    ts2 = getOnsetTimeStamps(trace, Fs=10000., onsetType='falling')
    assert(ts2[2] == 0.0030)
    #============================================================================================================

    print 'for debugging...'