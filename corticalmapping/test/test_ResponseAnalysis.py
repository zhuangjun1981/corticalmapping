import os
import numpy as np
from pandas import DataFrame
import unittest
import corticalmapping.ResponseAnalysis as ra
from corticalmapping.ResponseAnalysis import SpatialTemporalReceptiveField as STRF


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestResponseAnalysis(unittest.TestCase):

    def setUp(self):
        pass

    def test_SpatialReceptiveField(self):
        srf = ra.SpatialReceptiveField(np.arange(9).reshape((3, 3)), np.arange(3), np.arange(3))
        assert (np.array_equal(srf.weights, np.arange(1, 9)))

    def test_SpatialReceptiveField_thresholdReceptiveField(self):
        srf = ra.SpatialReceptiveField(np.arange(9).reshape((3, 3)), np.arange(3), np.arange(3))
        thresholded_srf = srf.threshold(4)
        assert (np.array_equal(thresholded_srf.weights, np.arange(4, 9)))

    def test_SpatialReceptiveField_interpolate(self):
        srf = ra.SpatialReceptiveField(np.random.rand(5, 5), np.arange(5)[::-1], np.arange(5))
        interpolated_srf = srf.interpolate(5)
        # print(interpolated_srf.get_weighted_mask().shape)
        assert (interpolated_srf.get_weighted_mask().shape == (20, 20))


class TestResponseMatrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_initiation(self):

        param0 = np.arange(10)
        param1 = [str(c) for c in np.arange(10, 20)]
        traces = [np.ones((5, 8)) for _ in range(10)]
        time = np.arange(8) * 0.001 - 0.003

        data = DataFrame(columns=['param0', 'param1', 'traces'])
        data['param0'] = param0
        data['param1'] = param1
        data['traces'] = traces

        rm = ra.ResponseMatrix(data=data, time=time)
        assert (rm.data.columns.tolist() == ['param0', 'param1', 'trigger_ts', 'traces'])


class TestSpatialTemporalReceptiveField(unittest.TestCase):

    def setUp(self):
        pass

    def test_instantialtion(self):

        # locations
        locations1 = [(1.0, 2.), (3., 4.), (5, 6)]
        locations2 = ((1.0, 2.), (3., 4.), (5, 6))
        locations3 = np.array([[1.0, 2.], [3., 4.], [5, 6]])
        # print(locations3)

        # signs
        signs1 = [1., -1., 0.]
        signs2 = (-0.5, 0.6, 1)
        signs3 = np.array([1, 1, -1])

        # traces
        traces1 = [np.random.rand(3, 4),
                   np.ones((5, 4)),
                   np.zeros((6, 4))]
        traces2 = [np.random.rand(3, 4),
                   np.ones((5, 4), dtype=np.uint8),
                   np.zeros((6, 4), dtype=np.int16)]
        traces3 = np.random.randn(3, 5, 4)

        # time axis
        time = np.arange(-1, 3).astype(np.float32)

        # trigger_ts
        trigger_ts1 = [[0., 5., 6.], np.array([3.3, 2.5, 7., 7.5, 9.]), [8.1, 3.2, 4.0, 7.4, 20.9, 6.4]]
        trigger_ts2 = None
        trigger_ts3 = np.random.rand(3, 5)

        strf1 = STRF.from_components(locations=locations1, signs=signs1, traces=traces1,
                                     trigger_ts=trigger_ts1, time=time)
        strf2 = STRF.from_components(locations=locations2, signs=signs2, traces=traces2,
                                     trigger_ts=trigger_ts2, time=time)
        strf3 = STRF.from_components(locations=locations3, signs=signs3, traces=traces3,
                                     trigger_ts=trigger_ts3, time=time)

        print(strf1.get_conditions())
        print(strf2.get_conditions())
        print(strf3.get_conditions())

    def test_merge_duplicates(self):
        locations = [(1.0, 2.), (3., 4.), (5, 6), (3., 4.), (5., 6.)]
        signs = [1., -1., 0., -1., 0]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16),
                  np.random.randn(10, 4),
                  [np.array([5, 6, 7, 8])]]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5], np.arange(15, 25), [20.5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)
        strf.merge_duplicates()
        assert(len(strf.data) == 3)
        assert(len(strf.data.loc[1, 'trigger_ts']) == 15)
        assert(len(strf.data.loc[2, 'trigger_ts']) == 7)

        strf.check_integrity()

    def test_add_traces(self):
        locations = [(1.0, 2.), (3., 4.), (5, 6)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)

        locations2 = [(3., 4.), (5., 6.)]
        signs2 = [-1., 0]
        traces2 = [np.random.randn(3, 4),
                   [np.array([5, 6, 7, 8])]]
        trigger_ts2 = [np.arange(15, 18), [20.5]]

        strf.add_traces(locations=locations2, signs=signs2, traces=traces2, trigger_ts=trigger_ts2, verbose=False)

        assert (len(strf.data) == 3)
        assert (np.array_equal(strf.data.iloc[0]['trigger_ts'], np.array([0.0, 5.0, 6.0], dtype=np.float64)))
        assert (np.array_equal(strf.data.iloc[1]['trigger_ts'],
                               np.array([7.0, 7.5, 9.0, 9.8, 10.4, 15., 16., 17.], dtype=np.float64)))
        assert (np.array_equal(strf.data.iloc[2]['trigger_ts'],
                               np.array([8.1, 3.2, 4.0, 2.0, 4.0, 5.0, 20.5], dtype=np.float64)))

        strf.add_traces(locations=locations2, signs=signs2, traces=traces2, trigger_ts=None, verbose=False)

        assert (len(strf.data) == 3)
        assert (np.array_equal(strf.data.iloc[0]['trigger_ts'], np.array([0.0, 5.0, 6.0], dtype=np.float64)))
        assert (len(strf.data.iloc[1]['trigger_ts']) == 11)
        assert (len(strf.data.iloc[2]['trigger_ts']) == 8)

    def test_to_from_h5_group(self):
        locations = [(1.0, 2.), (3., 4.), (5, 6)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)

        import h5py
        if 'test.hdf5' in os.listdir(curr_folder) and os.path.isfile('test.hdf5'):
            os.remove('test.hdf5')
        test_f = h5py.File('test.hdf5')
        strf_grp = test_f.create_group('strf')
        strf.to_h5_group(strf_grp)
        test_f.close()

        load_f = h5py.File('test.hdf5', 'r')
        strf2 = STRF.from_h5_group(load_f['strf'])

        assert (np.array_equal(strf.data.loc[0, 'trigger_ts'], strf2.data.loc[0, 'trigger_ts']))
        assert (np.array_equal(strf.data.loc[1, 'trigger_ts'], strf2.data.loc[1, 'trigger_ts']))
        assert (np.array_equal(strf.data.loc[2, 'trigger_ts'], strf2.data.loc[2, 'trigger_ts']))
        assert (np.array_equal(strf.data.loc[0, 'traces'], strf2.data.loc[0, 'traces']))
        assert (np.array_equal(strf.data.loc[1, 'traces'], strf2.data.loc[1, 'traces']))
        assert (np.array_equal(strf.data.loc[2, 'traces'], strf2.data.loc[2, 'traces']))

        '''
        something can be added here to test the strf2 object
        '''
        load_f.close()

        os.remove('test.hdf5')

    def test_plot_traces(self):
        locations = [(1.0, 10.), (2., 11.), (3., 12)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)
        strf.plot_traces(f=None, figSize=(10, 10), yRange=[-0.1, 1.1], altRange=None, aziRange=None)

        import matplotlib.pyplot as plt
        # plt.show()
        plt.close('all')

    def test_get_amplitude_map(self):
        locations = [(1.0, 10.), (2., 11.), (3., 12)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)
        ampON, ampOFF, allAltPos, allAziPos = strf.get_amplitude_map(timeWindow=[0.5, 2.5])
        # print(ampON)
        # print(ampOFF)
        # print(allAltPos)
        # print(allAziPos)
        assert (np.array_equal(allAltPos.astype(np.float32), np.array([3., 2., 1.], dtype=np.float32)))
        assert (np.array_equal(allAziPos.astype(np.float32), np.array([10., 11., 12.], dtype=np.float32)))
        assert (ampON[2, 0] - 0.5964107 < 1E-7)
        assert (ampOFF[1, 1] - 1. < 1E-10)
        assert (np.isnan(ampON[0, 1]))
        assert (np.isnan(ampOFF[2, 2]))

    def test_get_delta_amplitude_map(self):
        np.random.seed(0)
        locations = [(1.0, 10.), (2., 11.), (3., 12)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)
        ampON, ampOFF, allAltPos, allAziPos = strf.get_delta_amplitude_map(timeWindow=[0.5, 2.5])
        # print(ampON)
        # print(ampOFF)
        # print(allAltPos)
        # print(allAziPos)
        assert (ampON[2, 0] + 0.01243919 < 1E-7)
        assert (ampOFF[1, 1] - 0. < 1E-10)
        assert (np.isnan(ampON[0, 0]))
        assert (np.isnan(ampOFF[0, 0]))

    def test_shink(self):
        np.random.seed(0)
        locations = [(1.0, 10.), (2., 11.), (3., 12)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)
        strf.shrink(altRange=[1.5, 3.5], aziRange=[10.5, 12.5])
        assert (len(strf.data) == 2)

    def test_get_local_dff_strf(self):
        np.random.seed(0)
        locations = [(1.0, 10.), (2., 11.), (3., 12)]
        signs = [1., -1., 0.]
        traces = [np.random.rand(3, 4),
                  np.ones((5, 4), dtype=np.uint8),
                  np.zeros((6, 4), dtype=np.int16)]
        trigger_ts = [[0., 5., 6.], (7., 7.5, 9., 9.8, 10.4), [8.1, 3.2, 4.0, 2, 4, 5]]
        time = np.arange(-1, 3).astype(np.float32)

        strf = STRF.from_components(locations=locations, signs=signs, traces=traces, trigger_ts=trigger_ts, time=time)
        strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=True, add_to_trace=3.)
        # print(strf_dff.data)
        # print(strf_dff.data.iloc[0]['traces'])
        # print(strf_dff.data.iloc[1]['traces'])
        # print(strf_dff.data.iloc[2]['traces'])
        assert (strf_dff.data.iloc[0]['traces'].shape ==
                strf_dff.data.iloc[1]['traces'].shape ==
                strf_dff.data.iloc[2]['traces'].shape ==
                (1, 4))
        assert (np.isnan(strf_dff.data.iloc[0]['trigger_ts']))
        assert (np.isnan(strf_dff.data.iloc[1]['trigger_ts']))
        assert (np.isnan(strf_dff.data.iloc[2]['trigger_ts']))
        assert (np.array_equal(strf_dff.data.iloc[1]['traces'], np.zeros((1, 4), dtype=np.float32)))
        assert (np.array_equal(strf_dff.data.iloc[2]['traces'], np.zeros((1, 4), dtype=np.float32)))

        strf_dff = strf.get_local_dff_strf(is_collaps_before_normalize=False, add_to_trace=3.)
        # print(strf_dff.data)
        # print(strf_dff.data.iloc[0]['traces'])
        # print(strf_dff.data.iloc[1]['traces'])
        # print(strf_dff.data.iloc[2]['traces'])
        assert (np.array_equal(strf_dff.data.iloc[1]['traces'], np.zeros((5, 4), dtype=np.float32)))
        assert (np.array_equal(strf_dff.data.iloc[2]['traces'], np.zeros((6, 4), dtype=np.float32)))