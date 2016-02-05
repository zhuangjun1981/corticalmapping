__author__ = 'junz'


import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.SingleCellAnalysis as sca

import corticalmapping.core.FileTools as ft

plt.ioff()
currFolder = os.path.dirname(os.path.realpath(__file__))
testDataFolder = os.path.join(currFolder,'data')

sparseNoiseDisplayLogPath = os.path.join(testDataFolder,'SparseNoiseDisplayLog.pkl')
testH5Path = os.path.join(testDataFolder,'test.hdf5')
STRFDataPath = os.path.join(testDataFolder,'cellsSTRF.hdf5')


print sparseNoiseDisplayLogPath
def test_mergeROIs():
    roi1 = sca.WeightedROI(np.arange(9).reshape((3,3)))
    roi2 = sca.WeightedROI(np.arange(1,10).reshape((3,3)))

    merged_ROI = sca.mergeWeightedROIs(roi1,roi2)
    merged_ROI2 = sca.mergeBinaryROIs(roi1,roi2)

    assert(np.array_equal(merged_ROI.getWeightedMask(),np.arange(1,18,2).reshape((3,3))))
    assert(np.array_equal(merged_ROI2.getBinaryMask(),np.ones((3,3))))

def test_getSparseNoiseOnsetIndex():
    allOnsetInd, onsetIndWithLocationSign = sca.getSparseNoiseOnsetIndex(ft.loadFile(sparseNoiseDisplayLogPath))
    # print list(allOnsetInd[0:10])
    # print onsetIndWithLocationSign[2][0]
    assert(list(allOnsetInd[0:10])==[0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    assert(np.array_equal(onsetIndWithLocationSign[2][0],np.array([-41.53125,  96.25])))


def test_load_STRF_From_H5():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    assert(STRF.data['traces'][20][4][8]-0.934942 < 1e-10)
    # STRF.plotTraces(figSize=(15,10),yRange=[-5,50],columnSpacing=0.002,rowSpacing=0.002)

def test_ROI():
    a = np.zeros((10,10))
    a[5:7,3:6]=1
    a[8:9,7:10]=np.nan
    roi = sca.ROI(a)
    # plt.imshow(roi.getBinaryMask(),interpolation='nearest')
    assert(list(roi.getCenter())==[5.5,4.])

def test_ROI_getBinaryTrace():
    mov = np.random.rand(5,4,4); mask = np.zeros((4,4)); mask[2,3]=1; trace1 = mov[:,2,3]
    roi = sca.ROI(mask);trace2 = roi.getBinaryTrace(mov)
    assert(np.array_equal(trace1,trace2))

def test_WeigthedROI_getWeightedCenter():
    aa = np.random.rand(5,5); mask = np.zeros((5,5))
    mask[2,3]=aa[2,3]; mask[1,4]=aa[1,4]; mask[3,4]=aa[3,4]
    roi = sca.WeightedROI(mask); center = roi.getWeightedCenter()
    assert(center[0] == (2*aa[2,3]+1*aa[1,4]+3*aa[3,4])/(aa[2,3]+aa[1,4]+aa[3,4]))

def test_plot_ROIs():
    aa = np.zeros((50,50));aa[15:20,30:35] = np.random.rand(5,5)
    roi1 = sca.ROI(aa)
    _ = roi1.plotBinaryMaskBorder(); _ = roi1.plotBinaryMask()
    roi2 = sca.WeightedROI(aa)
    _ = roi2.plotBinaryMaskBorder(); _ = roi2.plotBinaryMask(); _ = roi2.plotWeightedMask()

def test_WeightedROI_getWeightedCenterInCoordinate():
    aa = np.zeros((5,5));aa[1:3,2:4] = 0.5
    roi = sca.WeightedROI(aa)
    assert(list(roi.getWeightedCenterInCoordinate(range(2,7),range(1,6)))==[3.5,3.5])

def test_SpatialTemporalReceptiveField():
    locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
    signs = [1,1,1,1,-1,-1,-1,-1]
    traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],[np.arange(7,11)],[np.arange(8,12)]]
    time = np.arange(4,8)
    STRF = sca.SpatialTemporalReceptiveField(locations,signs,traces,time)
    assert(STRF.data['traces'][0][0][1]==8)
    assert(STRF.data['sign'][4]==1)
    assert(np.array_equal(STRF.getLocations()[2],np.array([3.,4.,-1.])))
    newLocations = [[location[0]+1,location[1]+1] for location in locations[0:4]]
    newSigns = [1,1,1,1]
    STRF.addTraces(newLocations,newSigns,traces[0:4])
    assert(STRF.data['traces'][7][1][2]==4)
    # _ = STRF.plotTraces()

def test_SpatialTemporalReceptiveField_IO():
    locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
    signs = [1,1,1,1,-1,-1,-1,-1]
    traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],[np.arange(7,11)],[np.arange(8,12)]]
    time = np.arange(4,8)

    STRF = sca.SpatialTemporalReceptiveField(locations,signs,traces,time)
    if os.path.isfile(testH5Path):os.remove(testH5Path)
    testFile = h5py.File(testH5Path)
    STRFGroup = testFile.create_group('spatial_temporal_receptive_field')
    STRF.toH5Group(STRFGroup)
    testFile.close()

    h5File = h5py.File(testH5Path)
    STRF = sca.load_STRF_From_H5(h5File['spatial_temporal_receptive_field'])
    h5File.close()
    assert(STRF.data['traces'][3][0][1]==7)

def test_SpatialTemporalReceptiveField_getAmpLitudeMap():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    ampON, ampOFF, altPos, aziPos = STRF.getAmplitudeMap()
    assert(ampON[7,10]-(-0.0258248019964) < 1e-10)
    assert(ampOFF[8,9]-(-0.501572728157) < 1e-10)
    assert(altPos[5]==30.)
    assert(aziPos[3]==-5.)
    # sca.plot2DReceptiveField(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

def test_SpatialTemporalReceptiveField_getZscoreMap():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    zscoreON, zscoreOFF, altPos, aziPos = STRF.getZscoreMap()
    assert(zscoreON[7,10]-(-0.070735671412) < 1e-10)
    assert(zscoreOFF[8,9]-(-0.324245551387) < 1e-10)
    # sca.plot2DReceptiveField(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

def test_SpatialTemporalReceptiveField_getCenters():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    assert(STRF.getZscoreROICenters()[1][1]-(-2.1776047950146622)<1e-10)

def test_SpatialTemporalReceptiveField_getAmplitudeReceptiveField():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    ampRFON, ampRFOFF = STRF.getAmplitudeReceptiveField()
    assert(ampRFON.sign==1);assert(ampRFOFF.sign==-1)
    assert(ampRFOFF.getWeightedMask()[7,9]-3.2014527<1e-7)

def test_SpatialTemporalReceptiveField_getZscoreReceptiveField():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    zscoreRFON, zscoreRFOFF = STRF.getZscoreReceptiveField()
    assert(zscoreRFON.sign==1);assert(zscoreRFOFF.sign==-1)
    assert(zscoreRFOFF.getWeightedMask()[7,9]-1.3324414<1e-7)

def test_SpatialTemporalReceptiveField_shrink():
    f = h5py.File(STRFDataPath)
    STRF = sca.load_STRF_From_H5(f['cell0003']['spatial_temporal_receptive_field'])
    STRF.shrink([-10,10],None)
    assert(np.array_equal(np.unique(np.array(STRF.getLocations())[:,0]),np.array([-10.,-5.,0.,5.,10.])))
    STRF.shrink(None,[0,20])
    assert(np.array_equal(np.unique(np.array(STRF.getLocations())[:,1]),np.array([0.,5.,10.,15.,20.])))

def test_SpatialReceptiveField():
    SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3,3)),np.arange(3),np.arange(3))
    assert(np.array_equal(SRF.weights,np.arange(1,9)))

def test_SpatialReceptiveField_thresholdReceptiveField():
    SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3,3)),np.arange(3),np.arange(3))
    thresholdedSRF=SRF.thresholdReceptiveField(4)
    assert(np.array_equal(thresholdedSRF.weights,np.arange(4,9)))

def test_SpatialReceptiveField_interpolate():
    SRF = sca.SpatialReceptiveField(np.random.rand(5,5),np.arange(5)[::-1],np.arange(5))
    SRF.interpolate(5)
    assert(SRF.getWeightedMask().shape==(20,20))

plt.show()