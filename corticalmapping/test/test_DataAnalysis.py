__author__ = 'junz'

import unittest
import numpy as np
import corticalmapping.core.DataAnalysis as da


class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        pass

    def test_interpolate_nans(self):
        y = np.array([0., 1., 2., 3., np.nan, np.nan, 6., 7., np.nan, 9.])
        y1 = da.interpolate_nans(y)
        # print(y1)
        # print(np.arange(10.))
        assert (np.array_equal(y1, np.arange(10.)))

    def test_downsample(self):
        y = np.arange(10)
        y1 = da.downsample(arr=y, rate=2, method=np.mean)
        print(y1)
        assert (np.array_equal(y1, np.arange(0.5, 9, 2.)))

