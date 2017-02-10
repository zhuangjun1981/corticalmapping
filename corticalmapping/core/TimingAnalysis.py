__author__ = 'junz'

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

plt.ioff()

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


def threshold_onset(data, threshold=0, direction='up', fs=10000.):
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


def discrete_cross_correlation(ts1, ts2, t_range=(-1., 1.), bins=100, isPlot=False):
    """
    cross correlation of two time series of discrete events, return crosscorrelogram of total event 2 counts triggered
    by event 1.

    :param ts1: numpy.array, timestamps of the first event
    :param ts2: numpy.array, timestamps of the second event
    :param t_range: tuple of two elements, temporal window of crosscorrelogram, the first element should be smaller than
                    the second element.
    :param bins: int, number of bins
    :param isPlot:
    :return: t: numpy.array, time axis of crosscorrelorgam, mark the left edges of each time bin
             value: numpy array, total event 2 counts in each time bin
    """

    binWidth = (float(t_range[1]) - float(t_range[0])) / bins
    t = np.arange(bins) * binWidth + t_range[0]
    intervals = zip(t, t + binWidth)
    values = np.zeros(bins, dtype=np.int)

    for ts in list(ts1):
        currIntervals = [x + ts for x in intervals]
        for i, interval in enumerate(currIntervals):
            values[i] += len(np.where(np.logical_and(ts2>interval[0],ts2<=interval[1]))[0])

    if isPlot:
        f = plt.figure(figsize=(15,4)); ax = f.add_subplot(111)
        ax.bar([a[0] for a in intervals],values,binWidth*0.9);ax.set_xticks(t)

    return t, values


def find_nearest(trace, value, direction=0):
    '''
    return the index in "trace" having the closest value to "value"

    direction: int, can be 0, 1 or -1
               if 0, look for all elements in the trace
               if 1, only look for elements not smaller than the value, return None if all elements in the trace are
                     smaller than the value
               if -1, only look for elements not larger than the value, return None if all elements in the trace are
                     larger than the value
    '''

    diff = (trace - value).astype(np.float32)

    if direction == -1:
        diff[diff > 0] = np.nan
        diff = diff * -1
    elif direction == 1:
        diff[diff < 0] = np.nan
    elif direction == 0:
        diff = np.abs(diff)
    else:
        raise ValueError('"direction" should be 0, 1 or -1.')

    if np.isnan(diff).all():
        return None
    else:
        return np.nanargmin(diff)


def get_event_with_pre_iei(ts_event, iei=None):
    """
    get events which has a pre inter event interval (IEI) longer than a certain period of time

    :param ts_event: 1-d array, timestamps of events
    :param iei: float, criterion for pre IEI duration

    :return: 1-d array, refined timestamps
    """

    if not check_monotonicity(ts_event, direction='increasing'):
        raise ValueError('the input event timestamps should be monotonically increasing.')

    if iei is None:
        return ts_event
    else:
        ts_refined = np.empty(ts_event.shape)
        ts_refined[:] = np.nan
        ind = 0

        for i, ts in enumerate(ts_event):
            if (ts > iei) and (i > 0) and (ts - ts_event[i-1] > iei):
                ts_refined[ind] = ts
                ind += 1

        ts_refined = ts_refined[0: ind]

        return ts_refined


def get_onset_timeStamps(trace, Fs=10000., threshold = 3., onsetType='raising'):
    '''
    param trace: time trace of digital signal recorded as analog
    param Fs: sampling rate
    return onset time stamps
    '''

    pos = trace > threshold
    if onsetType == 'raising':
        return ((~pos[:-1] & pos[1:]).nonzero()[0]+1)/float(Fs)
    elif onsetType == 'falling':
        return ((pos[:-1] & ~pos[1:]).nonzero()[0]+1)/float(Fs)
    else:
        raise LookupError('onsetType should be either "raising" or "falling"!')


def power_spectrum(trace, fs, freq_range=(0., 300.), freq_bins=300, is_plot=False):
    '''
    return power spectrum of a signal trace (should be real numbers) at sampling rate of fs

    :param: trace, numpy.array, input trace
    :param: fs, float, sampling rate (Hz)
    :param: freq_range, tuple of two floats, range of analyzed frequencies
    :param: freq_bins, int, number of freq_bins of frequency axis
    '''
    spectrum_full = np.abs(np.fft.rfft(trace))**2 / float(len(trace))
    freqs = np.fft.rfftfreq(trace.size, 1. / fs)

    freq_bin_width = (freq_range[1] - freq_range[0]) / freq_bins
    freq_axis = np.arange(freq_bins, dtype=np.float32) * freq_bin_width + freq_range[0]

    spectrum = np.zeros(freq_axis.shape, dtype=np.float32)

    for i, freq in enumerate(freq_axis):
        curr_spect = spectrum_full[(freqs >= freq) & (freqs < (freq + freq_bin_width))]
        spectrum[i] = sum(curr_spect)

    if is_plot:
        f=plt.figure()
        ax=f.add_subplot(111)
        ax.plot(freq_axis, spectrum)
        ax.set_xlabel('frequency (Hz)')
        ax.set_ylabel('power')
        plt.show()

    return spectrum, freq_axis


def sliding_power_spectrum(trace, fs, sliding_window_length=5., sliding_step_length=None, freq_range=(0., 300.),
                           freq_bins=300, is_plot=False, **kwargs):
    '''
    calculate power_spectrum of a given trace over time

    :param: trace: input signal trace
    :param: fs: sampling rate (Hz)
    :param: sliding_window_length: length of sliding window (sec)
    :param: sliding_step_length: length of sliding step (sec), if None, equal to sliding_window_length
    :param: freq_range, tuple of two floats, range of analyzed frequencies
    :param: freq_bins, int, number of freq_bins of frequency axis
    :param: is_plot: bool, to plot or not

    :param: **kwargs, inputs to plt.imshow function

    :return
    spectrum: 2d array, power at each frequency at each time,
              time is from the first column to the last column
              frequence is from the last row to the first row
    times: time stamp for each column (starting point of each sliding window)
    freq_axis: frequency for each row (from low to high)
    '''

    if len(trace.shape) != 1: raise ValueError, 'Input trace should be 1d array!'

    total_length = len(trace) / float(fs)

    time_line = np.arange(len(trace)) * (1. / fs)

    freq_bin_width = (freq_range[1] - freq_range[0]) / freq_bins
    freq_axis = np.arange(freq_bins, dtype=np.float32) * freq_bin_width + freq_range[0]

    if sliding_step_length is None: sliding_step_length = sliding_window_length
    if sliding_step_length > sliding_window_length: print "Step length larger than window length, not using all data points!"
    times = np.arange(0., total_length, sliding_step_length)
    times = times[(times + sliding_window_length) < total_length]

    if len(times) == 0: raise ValueError, 'No time point found.'
    else:
        points_in_window = int(sliding_window_length * fs)
        if points_in_window <= 0: raise ValueError, 'Sliding window length too short!'
        else:
            spectrum = np.zeros((len(freq_axis), len(times)))
            for idx, start_time in enumerate(times):
                starting_point = find_nearest(time_line, start_time)
                ending_point = starting_point + points_in_window
                current_trace = trace[starting_point:ending_point]
                current_spectrum, freq_axis = power_spectrum(current_trace, fs, freq_range=freq_range, freq_bins=freq_bins,
                                                             is_plot=False)
                spectrum[:,idx] = current_spectrum

    if is_plot:
        f = plt.figure(figsize=(15, 6)); ax = f.add_subplot(111)
        fig = ax.imshow(spectrum, interpolation='nearest', **kwargs)
        ax.set_xlabel('times (sec)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_xticks(range(len(times))[::(len(times)//10)])
        ax.set_yticks(range(len(freq_axis))[::(len(freq_axis)//10)])
        ax.set_xticklabels(times[::(len(times)//10)])
        ax.set_yticklabels(freq_axis[::(len(freq_axis)//10)])
        ax.invert_yaxis()
        ax.set_aspect(float(len(times)) * 0.5 / float(len(freq_axis)))
        f.colorbar(fig)

        return spectrum, times, freq_axis, f
    else:
        return spectrum, times, freq_axis


def get_burst(spikes, pre_isi=(-np.inf, -0.1), inter_isi=0.004, spk_num_thr=2):
    """

    detect bursts with certain pre inter-spike-interval (ISI) and within ISI.

    :param spikes: timestamps of the spike train
    :param pre_isi: the criterion of the pre burst ISI. all burst should have pre ISIs within this duration.
         unit: second. default: [-inf, -0.1]
    :param inter_isi:  the criterion of the inter burst ISI. The spikes within a burst should have ISIs no longer than
        this duration. unit: second. default: 0.004
    :param spk_num_thr: the criterion of the number of spike within a burst. All bursts should have no less than this
        number of spikes, int (larger than 1), default: 2
    :return:
        burst_ts: timestamps of each burst
        burst_ind: N x 2 np.array, np.uint32, each row is a burst, first column is the onset index if this burst in the
                   spike train, second column is the number of spikes in this burst
    """

    if inter_isi >= -pre_isi[1]:
        raise ValueError('inter_isi should be way shorter than pre_isi threshold.')

    burst_ts = []
    burst_ind = []

    i = 1

    while i <= len(spikes)-2:

        curr_pre_isi = spikes[i-1] - spikes[i]
        curr_post_isi = spikes[i+1] - spikes[i]

        if pre_isi[0] <= curr_pre_isi <= pre_isi[1] and curr_post_isi<=inter_isi:
            burst_ts.append(spikes[i])

            j = 2

            while (i + j) <= len(spikes) - 1:
                next_isi = spikes[i + j] - spikes[i + j - 1]
                if next_isi <= inter_isi:
                    j += 1
                else:
                    break

            burst_ind.append([i, j])

            i += j

        else:

            i += 1

    burst_ts = np.array(burst_ts, dtype=np.float)
    burst_ind = np.array(burst_ind, dtype=np.uint)

    burst_ts = burst_ts[burst_ind[:, 1] >= spk_num_thr]
    burst_ind = burst_ind[burst_ind[:, 1] >= spk_num_thr]

    return burst_ts, burst_ind


def possion_event_ts(duration=600., firing_rate = 1., refractory_dur=0.001, is_plot=False):
    """
    return possion event timestamps given firing rate and durantion
    """

    curr_t = 0.
    ts = []
    isi = []

    while curr_t < duration:
        curr_isi = np.random.exponential(1. / firing_rate)

        while curr_isi <= refractory_dur:
            curr_isi = np.random.exponential(1. / firing_rate)

        ts.append(curr_t + curr_isi)
        isi.append(curr_isi)
        curr_t += curr_isi

    if is_plot:
        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax.hist(isi, bins=1000)

    return np.array(ts)


def check_monotonicity(arr, direction='increasing'):
    """
    check monotonicity of a 1-d array, usually a time series

    :param arr: input array, should be 1 dimensional
    :param direction: 'increasing', 'decreasing', 'non-increasing', 'non-decreasing'
    :return: True or False
    """

    if len(arr.shape) != 1:
        raise ValueError('Input array should be one dimensional!')

    if arr.shape[0] < 2:
        raise ValueError('Input array should have at least two elements!')

    diff = np.diff(arr)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    if direction == 'increasing':
        if min_diff > 0:
            return True
        else:
            return False

    elif direction == 'decreasing':
        if max_diff < 0:
            return True
        else:
            return False

    elif direction == 'non-increasing':
        if max_diff <= 0:
            return True
        else:
            return False

    elif direction == 'non-decreasing':
        if min_diff >= 0:
            return True
        else:
            return False

    else:
        raise LookupError('direction should one of the following: "increasing", "decreasing", '
                          '"non-increasing", "non-decreasing"!')


def butter_bandpass_filter(cutoffs=(300., 6000.), fs=30000., order=5, is_plot=False):
    """
    bandpass digital butterworth filter design
    :param cutoffs: [low cutoff frequency, high cutoff frequency], Hz
    :param fs: sampling rate, Hz
    :param order:
    :param is_plot:
    :return: b, a
    """
    nyq = 0.5 * fs
    low = cutoffs[0] / nyq
    high = cutoffs[1] / nyq

    b, a = sig.butter(order, [low, high], btype='band', analog=False, output='ba')

    if is_plot:
        w, h = sig.freqz(b, a, worN=2000)
        f = plt.figure(figsize=(10, 10))
        plt.loglog((fs * 0.5 / np.pi) * w, abs(h))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude')
        # plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoffs[0], color='green')
        plt.axvline(cutoffs[1], color='red')
        # plt.xlim([0, 10000])
        plt.show()

    return b, a


def butter_lowpass_filter(cutoff=300., fs=30000., order=5, is_plot=False):
    """
    bandpass digital butterworth filter design
    :param cutoffs: cutoff frequency, Hz
    :param fs: sampling rate, Hz
    :param order:
    :param is_plot:
    :return: b, a
    """
    nyq = 0.5 * fs
    low = cutoff / nyq

    b, a = sig.butter(order, low, btype='low', analog=False, output='ba')

    if is_plot:
        w, h = sig.freqz(b, a, worN=2000)
        f = plt.figure(figsize=(10, 10))
        plt.loglog((fs * 0.5 / np.pi) * w, abs(h))
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude')
        # plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff, color='red')
        # plt.xlim([0, 10000])
        plt.show()

    return b, a


def butter_bandpass(trace, fs=30000., cutoffs=(300., 6000.), order=5):
    """
    band pass filter a 1-d signal using digital butterworth filter design

    :param trace: input signal
    :param cutoffs: [low cutoff frequency, high cutoff frequency], Hz
    :param fs: sampling rate, Hz
    :param order:
    :return: filtered signal
    """

    b, a = butter_bandpass_filter(cutoffs=cutoffs, fs=fs, order=order)
    filtered = sig.lfilter(b, a, trace)
    return filtered


def butter_lowpass(trace, fs=30000., cutoff=300., order=5):
    """
    lowpass filter a 1-d signal using digital butterworth filter design

    :param trace: input signal
    :param cutoff: cutoff frequency, Hz
    :param fs: sampling rate, Hz
    :param order:
    :return: filtered signal
    """
    b, a = butter_lowpass_filter(cutoff=cutoff, fs=fs, order=order)
    filtered = sig.lfilter(b, a, trace)
    return filtered


def notch_filter(trace, fs=30000., freq_base=60., bandwidth=1., harmonics=4, order=2):
    """
    filter out signal at power frequency band and its harmonics. for each harmonic, signal at this band is extracted
    by using butter_bandpass function. Then the extraced signal was subtracted from the original signal

    :param trace: 1-d array, input trace
    :param fs: float, sampling rate, Hz
    :param freq_base: float, Hz, base frequency of contaminating signal
    :param bandwidth: float, Hz, filter bandwidth at each side of center frequency
    :param harmonics: int, number of harmonics to filter out
    :param order: int, order of butterworth filter, for a narrow band, shouldn't be larger than 2
    :return: filtered signal
    """

    sig_extracted = np.zeros(trace.shape, dtype=np.float32)

    for har in (np.arange(harmonics) + 1):
        cutoffs = [freq_base * har - bandwidth, freq_base * har + bandwidth]
        curr_sig = butter_bandpass(trace, fs=fs, cutoffs=cutoffs, order=order)
        sig_extracted = sig_extracted + curr_sig

    trace_filtered = trace.astype(np.float32) - sig_extracted

    return trace_filtered.astype(trace.dtype)


def event_triggered_average(ts_event, continuous, ts_continuous, t_range=(-1., 1.), bins=100, is_plot=False):
    """
    event triggered average of an analog signal trigger by discrete events. The timestamps of the analog signal may not
    be regular

    :param ts_event: 1-d array, float, timestamps of trigging event
    :param continuous: 1-d array, float, value of the analog signal
    :param ts_continuous: 1-d array, float, timestamps of the analog signal, should have same length as continuous
    :param t_range: tuple of 2 floats, temporal range of calculated average
    :param bins: int, number of bins of calculated average
    :param is_plot:
    :return: eta: 1-d array, float, event triggered average
             n: 1-d array, unit, number of time point of each bin
             t: 1-d array, float, time axis of event triggered average
             std: 1-d array, float, standard deviation of each bin in the event triggered average

             all four returned arrays should have same length
    """

    if t_range[0] >= t_range[1]:
        raise ValueError('t_range[0] should be smaller than t_range[1].')

    # sort continuous channel to be monotonic increasing in temporal domain
    sort_ind = np.argsort(ts_continuous)
    ts_continuous = ts_continuous[sort_ind]
    continuous = continuous[sort_ind]

    # initiation
    bin_width = (t_range[1] - t_range[0]) / bins
    t = t_range[0] + np.arange(bins, dtype=np.float32) * bin_width

    eta_list = [[] for bin in t]
    n = np.zeros(t.shape, dtype=np.uint64)
    std = np.zeros(t.shape, dtype=np.float32)
    eta = np.zeros(t.shape, dtype=np.float32)
    eta[:] = np.nan

    print '\nStart calculating event triggered average ...'
    percentage = None

    for ind_eve, eve in enumerate(ts_event):

        # for display
        curr_percentage =  int((float(ind_eve) * 100. / float(len(ts_event))) // 10) * 10
        if curr_percentage != percentage:
            print 'progress: ' + str(curr_percentage) + '%'
            # print eve, ':', ts_continuous[-1]
            percentage = curr_percentage

        if ((eve + t_range[0]) > ts_continuous[0]) and ((eve + t_range[1]) < ts_continuous[-1]):

            # slow algorithm
            # bin_starts = t + eve
            # for i, bin_start in enumerate(bin_starts):
            #     curr_datapoint =  continuous[(ts_continuous >= bin_start) & (ts_continuous < (bin_start + bin_width))]
            #     if len(curr_datapoint) != 0:
            #         eta_list[i] += list(curr_datapoint)

            # fast algorithm
            all_bin_start = eve + t_range[0]
            all_bin_end = eve + t_range[1]# - 1e-10
            for i, curr_t in enumerate(ts_continuous):
                if curr_t < all_bin_start:
                    continue
                elif curr_t >= all_bin_end:
                    break
                else:
                    bin_ind = int((curr_t - all_bin_start) // bin_width)
                    eta_list[bin_ind].append(continuous[i])

    for j, datapoints in enumerate(eta_list):
        if len(datapoints) > 0:
            eta[j] = np.mean(datapoints)
            n[j] = len(datapoints)
        if len(datapoints) > 1:
            std[j] = np.std(datapoints)

    if is_plot:

        f = plt.figure(figsize=(10, 10))
        ax = f.add_subplot(111)
        ax.plot(t, eta)
        ax.set_title('event triggered average')
        ax.set_xlabel('time (sec)')
        ax.set_xlim(t_range)
        plt.show()

    return eta, t, n, std


def event_triggered_event_trains(event_ts, triggers, t_range=(-1., 2.)):
    """
    calculate peri-trigger event timestamp trains.

    :param event_ts: array of float, discrete event timestamps
    :param triggers: array of float, trigger timestamps
    :param t_range: tuple of two floats, start and end time of the time window around the trigger
    :return: list of arrays, each array is a triggered train of event timestamps (relative to trigger time)
    """

    # event triggered timestamps
    etts = []

    for trigger in triggers:
        curr_st = trigger + t_range[0]
        curr_et = trigger + t_range[1]
        curr_train = event_ts[(event_ts >= curr_st) & (event_ts < curr_et)]
        etts.append(curr_train - trigger)

    return etts, t_range


if __name__=='__main__':

    #============================================================================================================
    # a=np.arange(100,dtype=np.float)
    # b=a+0.5+(np.random.rand(100)-0.5)*0.1
    # c=discrete_cross_correlation(a,b,range=(0,1),bins=50,isPlot=True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    # trace = np.array(([0.] * 5 + [5.] * 5) * 5)
    # ts = get_onset_timeStamps(trace, Fs=10000., onsetType='raising')
    # assert(ts[2] == 0.0025)
    # ts2 = get_onset_timeStamps(trace, Fs=10000., onsetType='falling')
    # assert(ts2[2] == 0.0030)
    #============================================================================================================

    #============================================================================================================
    # trace = np.random.rand(300) - 0.5
    # _, _ = power_spectrum(trace, 0.1, True)
    # plt.show()
    #============================================================================================================

    #============================================================================================================
    # time_line = np.arange(5000) * 0.01
    # trace = np.sin(time_line * (2 * np.pi))
    # trace2 = np.cos(np.arange(2500) * 0.05 * (2 * np.pi))
    # trace3 = np.cos(np.arange(2500) * 0.1 * (2 * np.pi))
    # trace = trace + np.concatenate((trace2, trace3))
    #
    # spectrum, times, freqs = sliding_power_spectrum(trace, 100, 1., is_plot=True)
    # print 'times:',times
    # print 'freqs:', freqs
    #============================================================================================================

    # ============================================================================================================
    # spikes = [0.3, 0.5, 0.501, 0.503, 0.505, 0.65, 0.7, 0.73, 0.733, 0.734, 0.735, 0.9, 1.5, 1.6,
    #           1.602, 1.603, 1.605, 1.94, 1.942]
    #
    # burst_ts, burst_ind = get_burst(spikes,  pre_isi=(-np.inf, -0.1), inter_isi=0.004, spk_num_thr=2)
    #
    # print burst_ts
    # print burst_ind
    # ============================================================================================================

    # ============================================================================================================
    # trace = np.arange(10)
    # print find_nearest(trace, 1.6)
    # ============================================================================================================

    # ============================================================================================================
    # possion_event_ts(firing_rate=1., duration=1000., refractory_dur=0.1, is_plot=True)
    # plt.show()
    # ============================================================================================================

    # ============================================================================================================
    # continuous = np.arange(1000) * 0.1
    # ts_continuous = np.arange(1000)
    # ts_event = [100, 101, 102, 200, 205]
    # eta, t, n, std = event_triggered_average(ts_event, continuous, ts_continuous, t_range=(-10., 10.), bins=20,
    #                                          is_plot=True)
    # print eta
    # print t
    # print n
    # print std
    # ============================================================================================================

    # ============================================================================================================
    np.random.seed(100)
    ts = np.arange(100) + np.random.rand(100) * 0.4
    print ts
    print np.min(np.diff(ts))
    ts2 = get_event_with_pre_iei(ts, iei=0.8)
    print ts2
    print len(ts2)
    print np.min(np.diff(ts2))
    # ============================================================================================================

    print 'for debugging...'