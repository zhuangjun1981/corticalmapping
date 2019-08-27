__author__ = 'junz'

import numpy as np
import corticalmapping.core.ImageAnalysis as ia
import unittest


class TestImageAnalysis(unittest.TestCase):

    def setup(self):
        pass

    def test_getTrace(self):
        mov = np.arange(64).reshape((4,4,4))
        # print mov

        mask1 = np.zeros((4,4)); mask1[2,2]=1; mask1[1,1]=1
        trace1 = ia.get_trace(mov, mask1, maskMode='binary')
        assert(trace1[2] == 39.5)

        mask2 = np.zeros((4,4),dtype=np.float); mask2[:]=np.nan; mask2[2,2]=1; mask2[1,1]=1
        trace2 = ia.get_trace(mov, mask2, maskMode='binaryNan')
        assert(trace2[2] == 39.5)

        mask3 = np.zeros((4,4),dtype=np.float); mask3[2,2]=1; mask3[1,1]=2
        trace3 = ia.get_trace(mov, mask3, maskMode='weighted')
        assert(trace3[2] == 58)

        mask4 = np.zeros((4,4),dtype=np.float); mask4[:]=np.nan; mask4[2,2]=1; mask4[1,1]=2
        trace4 = ia.get_trace(mov, mask4, maskMode='weightedNan')
        assert(trace4[2] == 58)


    def test_ROI_binary_overlap(self):
        roi1 = np.zeros((10, 10))
        roi1[4:8, 3:7] = 1
        roi1 = ia.ROI(roi1)
        roi2 = np.zeros((10, 10))
        roi2[5:9, 5:8] = 1
        roi2 = ia.ROI(roi2)
        assert(roi1.binary_overlap(roi2) == 6)


if __name__ == "__main__":
    TestImageAnalysis.test_getTrace()
    TestImageAnalysis.test_ROI_binary_overlap()