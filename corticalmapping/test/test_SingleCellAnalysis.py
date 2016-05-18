import corticalmapping.core.ImageAnalysis

__author__ = 'junz'


import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import corticalmapping.SingleCellAnalysis as sca
import corticalmapping.core.ImageAnalysis

import corticalmapping.core.FileTools as ft

plt.ioff()
currFolder = os.path.dirname(os.path.realpath(__file__))
testDataFolder = os.path.join(currFolder,'data')

sparseNoiseDisplayLogPath = os.path.join(testDataFolder,'SparseNoiseDisplayLog.pkl')
testH5Path = os.path.join(testDataFolder,'test.hdf5')
STRFDataPath = os.path.join(testDataFolder,'cellsSTRF.hdf5')


print sparseNoiseDisplayLogPath
def test_mergeROIs():
    roi1 = corticalmapping.core.ImageAnalysis.WeightedROI(np.arange(9).reshape((3, 3)))
    roi2 = corticalmapping.core.ImageAnalysis.WeightedROI(np.arange(1, 10).reshape((3, 3)))

    merged_ROI = sca.merge_weighted_rois(roi1, roi2)
    merged_ROI2 = sca.merge_binary_rois(roi1, roi2)

    assert(np.array_equal(merged_ROI.get_weighted_mask(), np.arange(1, 18, 2).reshape((3, 3))))
    assert(np.array_equal(merged_ROI2.get_binary_mask(), np.ones((3, 3))))

def test_getSparseNoiseOnsetIndex():
    allOnsetInd, onsetIndWithLocationSign = sca.get_sparse_noise_onset_index(ft.loadFile(sparseNoiseDisplayLogPath))
    # print list(allOnsetInd[0:10])
    # print onsetIndWithLocationSign[2][0]
    assert(list(allOnsetInd[0:10])==[0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    assert(np.array_equal(onsetIndWithLocationSign[2][0],np.array([-41.53125,  96.25])))


def test_SpatialTemporalReceptiveField_from_h5_group():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    assert(STRF.data['traces'][20][4][8]-0.934942 < 1e-10)
    # STRF.plot_traces(figSize=(15,10),yRange=[-5,50],columnSpacing=0.002,rowSpacing=0.002)

def test_ROI():
    a = np.zeros((10,10))
    a[5:7,3:6]=1
    a[8:9,7:10]=np.nan
    roi = corticalmapping.core.ImageAnalysis.ROI(a)
    # plt.imshow(roi.get_binary_mask(),interpolation='nearest')
    assert(list(roi.get_center()) == [5.5, 4.])

def test_ROI_getBinaryTrace():
    mov = np.random.rand(5,4,4); mask = np.zeros((4,4)); mask[2,3]=1; trace1 = mov[:,2,3]
    roi = corticalmapping.core.ImageAnalysis.ROI(mask);trace2 = roi.get_binary_trace(mov)
    assert(np.array_equal(trace1,trace2))

def test_WeigthedROI_getWeightedCenter():
    aa = np.random.rand(5,5); mask = np.zeros((5,5))
    mask[2,3]=aa[2,3]; mask[1,4]=aa[1,4]; mask[3,4]=aa[3,4]
    roi = corticalmapping.core.ImageAnalysis.WeightedROI(mask); center = roi.get_weighted_center()
    assert(center[0] == (2*aa[2,3]+1*aa[1,4]+3*aa[3,4])/(aa[2,3]+aa[1,4]+aa[3,4]))

def test_plot_ROIs():
    aa = np.zeros((50,50));aa[15:20,30:35] = np.random.rand(5,5)
    roi1 = corticalmapping.core.ImageAnalysis.ROI(aa)
    _ = roi1.plot_binary_mask_border(); _ = roi1.plot_binary_mask()
    roi2 = corticalmapping.core.ImageAnalysis.WeightedROI(aa)
    _ = roi2.plot_binary_mask_border(); _ = roi2.plot_binary_mask(); _ = roi2.plot_weighted_mask()

def test_WeightedROI_getWeightedCenterInCoordinate():
    aa = np.zeros((5,5));aa[1:3,2:4] = 0.5
    roi = corticalmapping.core.ImageAnalysis.WeightedROI(aa)
    assert(list(roi.get_weighted_center_in_coordinate(range(2, 7), range(1, 6))) == [3.5, 3.5])

def test_SpatialTemporalReceptiveField():
    locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
    signs = [1,1,1,1,-1,-1,-1,-1]
    traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],[np.arange(7,11)],[np.arange(8,12)]]
    time = np.arange(4,8)
    STRF = sca.SpatialTemporalReceptiveField(locations,signs,traces,time)
    assert(STRF.data['traces'][0][0][1]==8)
    assert(STRF.data['sign'][4]==1)
    assert(np.array_equal(STRF.get_locations()[2], np.array([3., 4., -1.])))
    newLocations = [[location[0]+1,location[1]+1] for location in locations[0:4]]
    newSigns = [1,1,1,1]
    STRF.add_traces(newLocations, newSigns, traces[0:4])
    assert(STRF.data['traces'][7][1][2]==4)
    # _ = STRF.plot_traces()

def test_SpatialTemporalReceptiveField_IO():
    locations = [[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0],[3.0, 4.0], [3.0, 5.0], [2.0, 4.0], [2.0, 5.0]]
    signs = [1,1,1,1,-1,-1,-1,-1]
    traces=[[np.arange(4)],[np.arange(1,5)],[np.arange(2,6)],[np.arange(3,7)],[np.arange(5,9)],[np.arange(6,10)],[np.arange(7,11)],[np.arange(8,12)]]
    time = np.arange(4,8)

    STRF = sca.SpatialTemporalReceptiveField(locations,signs,traces,time)
    if os.path.isfile(testH5Path):os.remove(testH5Path)
    testFile = h5py.File(testH5Path)
    STRFGroup = testFile.create_group('spatial_temporal_receptive_field')
    STRF.to_h5_group(STRFGroup)
    testFile.close()

    h5File = h5py.File(testH5Path)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(h5File['spatial_temporal_receptive_field'])
    h5File.close()
    assert(STRF.data['traces'][3][0][1]==7)

def test_SpatialTemporalReceptiveField_getAmpLitudeMap():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    ampON, ampOFF, altPos, aziPos = STRF.get_amplitude_map()
    assert(ampON[7,10]-(-0.0258248019964) < 1e-10)
    assert(ampOFF[8,9]-(-0.501572728157) < 1e-10)
    assert(altPos[5]==30.)
    assert(aziPos[3]==-5.)
    # sca.plot_2d_receptive_field(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

def test_SpatialTemporalReceptiveField_getZscoreMap():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    zscoreON, zscoreOFF, altPos, aziPos = STRF.get_zscore_map()
    assert(zscoreON[7,10]-(-0.070735671412) < 1e-10)
    assert(zscoreOFF[8,9]-(-0.324245551387) < 1e-10)
    # sca.plot_2d_receptive_field(ampON,altPos,aziPos,cmap='gray_r',interpolation='nearest')

def test_SpatialTemporalReceptiveField_getCenters():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    assert(STRF.get_zscore_roi_centers()[1][1] - (-2.1776047950146622) < 1e-10)

def test_SpatialTemporalReceptiveField_getAmplitudeReceptiveField():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    ampRFON, ampRFOFF = STRF.get_amplitude_receptive_field()
    assert(ampRFON.sign==1);assert(ampRFOFF.sign==-1)
    assert(ampRFOFF.get_weighted_mask()[7, 9] - 3.2014527 < 1e-7)

def test_SpatialTemporalReceptiveField_getZscoreReceptiveField():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    zscoreRFON, zscoreRFOFF = STRF.get_zscore_receptive_field()
    assert(zscoreRFON.sign==1);assert(zscoreRFOFF.sign==-1)
    assert(zscoreRFOFF.get_weighted_mask()[7, 9] - 1.3324414 < 1e-7)

def test_SpatialTemporalReceptiveField_shrink():
    f = h5py.File(STRFDataPath)
    STRF = sca.SpatialTemporalReceptiveField.from_h5_group(f['cell0003']['spatial_temporal_receptive_field'])
    STRF.shrink([-10,10],None)
    assert(np.array_equal(np.unique(np.array(STRF.get_locations())[:, 0]), np.array([-10., -5., 0., 5., 10.])))
    STRF.shrink(None,[0,20])
    assert(np.array_equal(np.unique(np.array(STRF.get_locations())[:, 1]), np.array([0., 5., 10., 15., 20.])))

def test_SpatialReceptiveField():
    SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3,3)),np.arange(3),np.arange(3))
    assert(np.array_equal(SRF.weights,np.arange(1,9)))

def test_SpatialReceptiveField_thresholdReceptiveField():
    SRF = sca.SpatialReceptiveField(np.arange(9).reshape((3,3)),np.arange(3),np.arange(3))
    thresholdedSRF=SRF.threshold_receptive_field(4)
    assert(np.array_equal(thresholdedSRF.weights,np.arange(4,9)))

def test_SpatialReceptiveField_interpolate():
    SRF = sca.SpatialReceptiveField(np.random.rand(5,5),np.arange(5)[::-1],np.arange(5))
    SRF.interpolate(5)
    assert(SRF.get_weighted_mask().shape == (20, 20))

plt.show()