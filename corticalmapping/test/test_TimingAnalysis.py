__author__ = 'junz'


import numpy as np
import corticalmapping.core.TimingAnalysis as ta

def test_getOnsetTimeStamps():
    trace = np.array(([0.] * 5 + [5.] * 5) * 5)
    ts = ta.get_onset_timeStamps(trace, Fs=10000., onsetType='raising')
    assert(ts[2] == 0.0025)
    ts2 = ta.get_onset_timeStamps(trace, Fs=10000., onsetType='falling')
    assert(ts2[2] == 0.0030)

def test_get_burst():
    spikes = [0.3, 0.5, 0.501, 0.503, 0.505, 0.65, 0.7, 0.73, 0.733, 0.734, 0.735, 0.9, 1.5, 1.6,
              1.602, 1.603, 1.605, 1.94, 1.942]

    _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.1), inter_isi=0.004, spk_num_thr=2)
    assert(np.array_equal(burst_ind, [[1, 4], [13, 4], [17, 2]]))

    _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.01), inter_isi=0.004, spk_num_thr=2)
    assert (np.array_equal(burst_ind, [[1, 4], [7, 4], [13, 4], [17, 2]]))

    _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.01), inter_isi=0.002, spk_num_thr=2)
    assert (np.array_equal(burst_ind, [[1, 2]]))

    _, burst_ind = ta.get_burst(spikes, pre_isi=(-np.inf, -0.01), inter_isi=0.004, spk_num_thr=3)
    assert (np.array_equal(burst_ind, [[1, 4], [7, 4], [13, 4]]))


if __name__ == '__main__':
    test_getOnsetTimeStamps()
    test_get_burst()