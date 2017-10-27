import os
import numpy as np
import corticalmapping.core.FileTools as ft

def align_visual_display_time(pkl_dict, ts_pd_fall, ts_display_rise, max_mismatch=0.1, verbose=True,
                              refresh_rate=60., allowed_jitter=0.01):

    """
    align photodiode and display frame TTL for Brain Observatory stimulus. During display, sync square
    alternate between black (duration 1 second) and white (duration 1 second) start with black.
    The beginning of the display was indicated by a quick flash of [black, white, black, white, black, white],
    16.6667 ms each. This function will find the frame indices of each onset of black syncsquare during display,
    and compare them to the corresponding photodiode timestamps (digital), and calculate mean display lag.

    :param pkl_dict: the dictionay of display log
    :param ts_pd_fall: 1d array, timestameps of photodiode fall
    :param ts_display_rise: 1d array, timestamps of the rise of display frames
    :param max_mismatch: positive float, second, If any single display lag is larger than 'max_mismatch'.
                         a ValueException will raise
    :param refresh_rate: positive float, monitor refresh rate, Hz
    :param allowed_jitter: positive float, allowed jitter to evaluate time interval, sec
    :return: ts_display_real, onset timestamps of each display frames after correction for display lag
             display_lag: 2d array, with two columns
                          first column: timestamps of black sync square onsets
                          second column: display lag at that time
    """

    if not (len(ts_pd_fall.shape) == 1 and ts_pd_fall.shape[0] > 8):
        raise ValueError('input "ts_pd_fall" should be a 1d array with more than 8 items.')

    if not len(ts_display_rise.shape) == 1:
        raise ValueError('input "ts_display_rise" should be a 1d array.')

    if not pkl_dict['items']['sync_square']['colorSequence'][0] == -1:
        raise ValueError('The visual display did not start with black sync_square!')

    frame_period = pkl_dict['items']['sync_square']['frequency'] * \
                   len(pkl_dict['items']['sync_square']['colorSequence'])

    ts_onset_frame_ttl = ts_display_rise[::frame_period]

    if verbose:
        print('Number of onset frame TTLs of black sync square: {}'.format(len(ts_onset_frame_ttl)))

    # detect display start in photodiode signal
    refresh_rate = float(refresh_rate)
    for i in range(3, len(ts_pd_fall) - 1):
        post_interval = ts_pd_fall[i + 1] - ts_pd_fall[i]
        pre_interval_1 = ts_pd_fall[i] - ts_pd_fall[i - 1]
        pre_interval_2 = ts_pd_fall[i - 1] - ts_pd_fall[i - 2]
        pre_interval_3 = ts_pd_fall[i - 2] - ts_pd_fall[i - 3]

        # print pre_interval_3, pre_interval_2, pre_interval_1, post_interval

        if abs(post_interval - frame_period / refresh_rate) <= allowed_jitter and \
            abs(pre_interval_1 - 20. / refresh_rate <= allowed_jitter) and \
            abs(pre_interval_2 - 20. / refresh_rate <= allowed_jitter) and \
            abs(pre_interval_3 - 20. / refresh_rate <= allowed_jitter):
                pd_start_ind = i
                break
    else:
        raise ValueError('Did not find photodiode signal marking the start of display.')

    for j in range(0, len(ts_pd_fall) - 1)[::-1]:
        pre_interval_1 = ts_pd_fall[j] - ts_pd_fall[j - 1]
        pre_interval_2 = ts_pd_fall[j - 1] - ts_pd_fall[j - 2]

        if abs(pre_interval_1 - 20. / refresh_rate <= allowed_jitter) and \
            abs(pre_interval_2 - 20. / refresh_rate <= allowed_jitter):
                pd_end_ind = j - 2
                break

        raise ValueError('Did not find photodiode signal marking the end of display.')


    ts_onset_frame_pd = ts_pd_fall[pd_start_ind : pd_end_ind]

    if verbose:
        print('Number of onset frame photodiode falls of black sync square: {}'.format(len(ts_onset_frame_pd)))

    if not len(ts_onset_frame_ttl) == len(ts_onset_frame_pd):
        raise ValueError('Number of onset frame TTLs ({}) and Number of onset frame photodiode signals ({}),'
                         'do not match!'.format(len(ts_onset_frame_ttl), len(ts_onset_frame_pd)))

    display_lag = ts_onset_frame_pd - ts_onset_frame_ttl

    display_lag_max = np.max(np.abs(display_lag))
    display_lag_max_ind = np.argmax(np.abs(display_lag))
    if display_lag_max > max_mismatch:
        raise ValueError('Display lag number {} : {}(sec) is greater than allow max_mismatch {} sec.'
                         .foramt(display_lag_max_ind, display_lag_max, max_mismatch))

    display_lag_mean = np.mean(display_lag)
    if verbose:
        print('Average display lag: {} sec.'.format(display_lag_mean))

    return ts_display_rise + display_lag_mean, np.array([ts_onset_frame_pd, display_lag]).transpose()

def get_stim_dict_drifting_grating(input_dict, stim_name):

    sweep_table = input_dict['sweep_table']
    sweep_order = input_dict['sweep_order']
    sweep_frames = input_dict['sweep_frames']

    # get sweep table
    sweeps = []
    blank_sweep = np.array([np.nan, np.nan, np.nan, np.nan, 1.], dtype=np.float32)
    for sweep_i in sweep_order:
        if sweep_i == -1:
            sweeps.append(blank_sweep)
        else:
            curr_s = sweep_table[sweep_i]
            sweeps.append(np.array([curr_s[0], curr_s[1], curr_s[2], curr_s[3], 0.], dtype=np.float32))
    sweeps = np.array(sweeps, dtype=np.float32)

    # get sweep onset frames
    sweep_onset_frames = [int(sf[0]) for sf in sweep_frames]
    sweep_onset_frames = np.array(sweep_onset_frames, dtype=np.uint64)

    stim_dict = {}
    stim_dict['stim_name'] = stim_name
    stim_dict['sweeps'] = sweeps
    stim_dict['sweep_onset_frames'] = sweep_onset_frames
    stim_dict['data_formatting'] = ['contrast', 'temporal_frequency', 'spatial_frequency', 'direction', 'is_blank']
    stim_dict['iterations'] = input_dict['runs']
    stim_dict['temporal_frequency_list'] = input_dict['sweep_params']['TF'][0]
    stim_dict['spatial_frequency_list'] = input_dict['sweep_params']['SF'][0]
    stim_dict['contrast_list'] = input_dict['sweep_params']['Contrast'][0]
    stim_dict['direction_list'] = input_dict['sweep_params']['Ori'][0]
    stim_dict['sweep_dur_sec'] = input_dict['sweep_length']
    stim_dict['midgap_dur_sec'] = input_dict['blank_length']
    stim_dict['num_of_blank_sweeps'] = input_dict['blank_sweeps']
    stim_dict['stim_text'] = input_dict['stim_text']
    stim_dict['frame_rate_hz'] = input_dict['fps']

    stim_dict['source'] = 'camstim'
    stim_dict['comments'] = 'The timestamps of this stimulus is the display frame index, not the actual time in seconds. ' \
                            'To get the real timestamps in seconds, please use these indices to find the timestamps ' \
                            'of displayed frames in "/processing/visual_display/frame_timestamps".'
    stim_dict['description'] = 'This stimulus is extracted from the pkl file saved by camstim software.'

    return stim_dict


def get_stim_dict_list(pkl_path):
    pkl_dict = ft.loadFile(pkl_path)
    stimuli = pkl_dict['stimuli']
    pre_blank_sec = pkl_dict['pre_blank_sec']
    post_blank_sec = pkl_dict['post_blank_sec']
    total_fps = pkl_dict['fps']

    start_frame_num = int(total_fps * pre_blank_sec)

    assert(pkl_dict['vsynccount'] == pkl_dict['total_frames'] + (pre_blank_sec + post_blank_sec) * total_fps)

    # print('\n'.join(pkl_dict.keys()))

    stim_dict_lst = []


    for stim_ind, stim in enumerate(stimuli):
        # print('\n'.join(stim.keys()))
        # print stim['stim_path']

        # get stim_type
        stim_str = stim['stim']
        if '(' in stim_str:

            if stim_str[0:stim_str.index('(')] == 'GratingStim':

                if 'Phase' in stim['sweep_params'].keys():
                    stim_type = 'static_grating_brain_observatory'
                elif 'TF' in stim['sweep_params'].keys():
                    stim_type = 'drifting_grating_brain_observatory'
                else:
                    print('\n\nunknow stimulus type:')
                    print(stim['stim_path'])
                    print(stim['stim_text'])
                    stim_type = None

            else:
                print('\n\nunknow stimulus type:')
                print(stim['stim_path'])
                print(stim['stim_text'])
                stim_type = None
        else:
            print('\n\nunknow stimulus type:')
            print(stim['stim_path'])
            print(stim['stim_text'])
            stim_type = None

        if stim_type == 'drifting_grating_brain_observatory':
            stim_name = '{:02d}_DriftingGratingBrainObservatory'.format(stim_ind)
            print('\n\nextracting stimulus: ' + stim_name)
            stim_dict = get_stim_dict_drifting_grating(input_dict=stim, stim_name=stim_name)
            stim_dict['sweep_onset_frames'] = stim_dict['sweep_onset_frames'] + start_frame_num
            stim_dict.update({'stim_type': 'drifting_grating_brain_observatory'})
        elif stim_type == 'static_gratings':
            print('\n\nskip static_gratings stimulus. stim index: {}.'.format(stim_ind))
            stim_dict = None
        else:
            print('\nskip unknow stimstimulus. stim index: {}.'.format(stim_ind))
            stim_dict = None

        stim_dict_lst.append(stim_dict)

    return stim_dict_lst



if __name__ == '__main__':

    pkl_path = '/media/junz/m2ssd/2017-09-25-preprocessing-test/m255_presynapticpop_vol1_bessel_DriftingGratingsTemp.pkl'

    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642817351_338502_20171010_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642244262_338502_20171006_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642499066_338502_20171009_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/642817351_338502_20171010_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/643543433_338502_20171016_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/643646020_338502_20171017_stim.pkl'
    # pkl_path = '/media/junz/m2ssd/2017-10-24-camstim-analysis/643792098_338502_20171018_stim.pkl'

    stim_dicts = get_stim_dict_list(pkl_path=pkl_path)