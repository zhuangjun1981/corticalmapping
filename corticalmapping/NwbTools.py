import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import corticalmapping.ephys.OpenEphysWrapper as oew
import corticalmapping.ephys.KilosortWrapper as kw
import corticalmapping.HighLevel as hl
import corticalmapping.core.FileTools as ft
import corticalmapping.core.TimingAnalysis as ta
import corticalmapping.core.PlottingTools as pt
try:
    from nwb.nwb import NWB
except ImportError:
    print('no Allen Institute NWB API. get this from ' \
          'http://stimash.corp.alleninstitute.org/projects/INF/repos/ainwb/browse')

DEFAULT_GENERAL = {
                   'session_id': '',
                   'experimenter': '',
                   'institution': 'Allen Institute for Brain Science',
                   # 'lab': '',
                   # 'related_publications': '',
                   'notes': '',
                   'experiment_description': '',
                   # 'data_collection': '',
                   'stimulus': '',
                   # 'pharmacology': '',
                   # 'surgery': '',
                   # 'protocol': '',
                   'subject': {
                               'subject_id': '',
                               # 'description': '',
                               'species': 'Mus musculus',
                               'genotype': '',
                               'sex': '',
                               'age': '',
                               # 'weight': '',
                               },
                   # 'virus': '',
                   # 'slices': '',
                   'extracellular_ephys': {
                                           'electrode_map': '',
                                           'sampling_rate': 30000.,
                                           # 'electrode_group': [],
                                           # 'impedance': [],
                                           # 'filtering': []
                                           },
                   'optophysiology': {
                                      # 'indicator': '',
                                      # 'excitation_lambda': '',
                                      # 'imaging_rate': '',
                                      # 'location': '',
                                      # 'device': '',
                                      },
                   # 'optogenetics': {},
                   'devices': {}
                   }
SPIKE_WAVEFORM_TIMEWINDOW = (-0.002, 0.002)

def plot_waveforms(waveforms, ch_locations=None, stds=None, waveforms_filtered=None, stds_filtered=None,
                   f=None, ch_ns=None, axes_size=(0.2, 0.2), **kwargs):
    """
    plot spike waveforms at specified channel locations

    :param waveforms: 2-d array, time point x channel, mean spike waveforms at each channel
    :param ch_locations: list of tuples, (x_location, y_location) for each channel, if None, waveform will be plotted
                         in a linear fashion
    :param stds: 2-d array, same size as waveform, measurement of variance of each time point at each channel
    :param waveforms_filtered: 2-d array, same size as waveform, waveforms from unfiltered analog signal
    :param stds_filtered: 2-d array, same size as waveform, measurement of variance of each time point at
                            each channel of unfiltered analog signal
    :param f: matplotlib figure object
    :param ch_ns: list of strings, names of each channel
    :param axes_size: tuple, sise of subplot, (width, height), only implemented when ch_locations is not None
    :return: f
    """

    if f is None:
        f = plt.figure(figsize=(10, 10))

    ch_num = waveforms.shape[1]

    if ch_locations is None:
        ax_s = []
        for i in range(ch_num):
            ax_s.append(f.add_subplot(1, ch_num, i + 1))
    else:
        ax_s = pt.distributed_axes(f, axes_pos=ch_locations, axes_size=axes_size)

    #  detect uniform y axis scale
    max_trace = waveforms.flatten()
    min_trace = waveforms.flatten()
    if stds is not None:
        max_trace = np.concatenate((max_trace, waveforms.flatten() + stds.flatten()))
        min_trace = np.concatenate((min_trace, waveforms.flatten() - stds.flatten()))
    if waveforms_filtered is not None:
        max_trace = np.concatenate((max_trace, waveforms_filtered.flatten()))
        min_trace = np.concatenate((min_trace, waveforms_filtered.flatten()))
        if stds_filtered is not None:
            max_trace = np.concatenate((max_trace, waveforms_filtered.flatten() + stds_filtered.flatten()))
            min_trace = np.concatenate((min_trace, waveforms_filtered.flatten() - stds_filtered.flatten()))

    peak_min = np.min(min_trace)
    peak_max = np.max(max_trace)

    for j, ax in enumerate(ax_s):

        # plot unfiltered data
        if waveforms_filtered is not None:
            curr_wf_f = waveforms_filtered[:, j]
            if stds_filtered is not None:
                curr_std_f = stds_filtered[:, j]
                ax.fill_between(list(range(waveforms_filtered.shape[0])), curr_wf_f - curr_std_f,
                                curr_wf_f + curr_std_f, color='#888888', alpha=0.5, edgecolor='none')
            ax.plot(curr_wf_f, '-', color='#555555', label='filtered', **kwargs)

        # plot filtered data
        curr_wf = waveforms[:, j]
        if stds is not None:
            curr_std = stds[:, j]
            ax.fill_between(list(range(waveforms.shape[0])), curr_wf - curr_std, curr_wf + curr_std,
                            color='#8888ff',alpha=0.5, edgecolor='none')
        ax.plot(curr_wf, '-', color='#3333ff', label='unfiltered', **kwargs)

        # plot title
        if ch_ns is not None:
            ax.set_title(ch_ns[j], y=0.9)

        ax.set_xlim([0, waveforms.shape[0] - 1])
        ax.set_ylim([peak_min, peak_max])
        ax.set_axis_off()

        if waveforms_filtered is not None and j == 0:
            ax.legend(frameon=False, loc='upper left', fontsize='small', bbox_to_anchor=(0, 0.9))

    return f


class RecordedFile(NWB):
    """
    Jun's wrapper of nwb file. Designed for LGN-ephys/V1-ophys dual recording experiments. Should be able to save
    ephys, wide field, 2-photon data in a single file.
    """

    def __init__(self, filename, is_manual_check=False, **kwargs):

        if os.path.isfile(filename):
            if is_manual_check:
                keyboard_input = ''
                while keyboard_input != 'y' and keyboard_input != 'n':
                    keyboard_input = input('\nthe path "' + filename + '" already exists. Modify it? (y/n) \n')
                    if keyboard_input == 'y':
                        super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
                    elif keyboard_input == 'n':
                        raise IOError('file already exists.')
            else:
                print(('\nModifying existing nwb file: ' + filename))
                super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
        else:
            print(('\nCreating a new nwb file: ' + filename))
            super(RecordedFile, self).__init__(filename=filename, modify=False, **kwargs)

    def add_general(self, general=DEFAULT_GENERAL, is_overwrite=True):
        """
        add general dictionary to the general filed
        """
        slf = self.file_pointer
        ft.write_dictionary_to_h5group_recursively(target=slf['general'], source=general, is_overwrite=is_overwrite)

    def add_open_ephys_data(self, folder, prefix, digital_channels=()):
        """
        add open ephys raw data to self, in acquisition group, less useful, because the digital events needs to be
        processed before added in
        :param folder: str, the folder contains open ephys raw data
        :param prefix: str, prefix of open ephys files
        :param digital_channels: list of str, digital channel
        :return:
        """
        output = oew.pack_folder_for_nwb(folder=folder, prefix=prefix, digital_channels=digital_channels)

        for key, value in list(output.items()):

            if 'CH' in key:  # analog channel for electrode recording
                ch_ind = int(key[key.find('CH') + 2:])
                ch_name = 'ch_' + ft.int2str(ch_ind, 4)
                ch_trace = value['trace']
                ch_series = self.create_timeseries('ElectricalSeries', ch_name, 'acquisition')
                ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
                                   resolution=1.)
                ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
                                           rate=float(value['header']['sampleRate']))
                ch_series.set_value('electrode_idx', ch_ind)
                ch_series.set_value('num_samples', len(ch_trace))
                ch_series.set_comments('continuous')
                ch_series.set_description('extracellular continuous voltage recording from tetrode')
                ch_series.set_source('open ephys')
                ch_series.finalize()

            elif key != 'events':  # other continuous channels
                ch_name = key[len(prefix) + 1:]
                ch_trace = value['trace']
                ch_series = self.create_timeseries('AbstractFeatureSeries', ch_name, 'acquisition')
                ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
                                   resolution=1.)
                ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
                                           rate=float(value['header']['sampleRate']))
                ch_series.set_value('features', ch_name)
                ch_series.set_value('feature_units', 'bit')
                ch_series.set_value('num_samples', len(ch_trace))
                ch_series.set_value('help', 'continuously recorded analog channels with same sampling times as '
                                            'of electrode recordings')
                ch_series.set_comments('continuous')
                ch_series.set_description('continuous voltage recording from IO board')
                ch_series.set_source('open ephys')
                ch_series.finalize()

            else:  # digital events

                for key2, value2 in list(value.items()):

                    ch_rise_ts = value2['rise']
                    ch_series_rise = self.create_timeseries('TimeSeries', key2+'_rise', 'acquisition')
                    ch_series_rise.set_data([], unit='', conversion=np.nan, resolution=np.nan)
                    if len(ch_rise_ts) == 0:
                        ch_rise_ts = np.array([np.nan])
                        ch_series_rise.set_time(ch_rise_ts)
                        ch_series_rise.set_value('num_samples', 0)
                    else:
                        ch_series_rise.set_time(ch_rise_ts)
                    ch_series_rise.set_description('timestamps of rise cross of digital channel: ' + key2)
                    ch_series_rise.set_source('open ephys')
                    ch_series_rise.set_comments('digital')
                    ch_series_rise.finalize()

                    ch_fall_ts = value2['fall']
                    ch_series_fall = self.create_timeseries('TimeSeries', key2 + '_fall', 'acquisition')
                    ch_series_fall.set_data([], unit='', conversion=np.nan, resolution=np.nan)
                    if len(ch_fall_ts) == 0:
                        ch_fall_ts = np.array([np.nan])
                        ch_series_fall.set_time(ch_fall_ts)
                        ch_series_fall.set_value('num_samples', 0)
                    else:
                        ch_series_fall.set_time(ch_fall_ts)
                    ch_series_fall.set_description('timestamps of fall cross of digital channel: ' + key2)
                    ch_series_fall.set_source('open ephys')
                    ch_series_fall.set_comments('digital')
                    ch_series_fall.finalize()

    def add_open_ephys_continuous_data(self, folder, prefix):
        """
        add open ephys raw continuous data to self, in acquisition group
        :param folder: str, the folder contains open ephys raw data
        :param prefix: str, prefix of open ephys files
        :param digital_channels: list of str, digital channel
        :return:
        """
        output = oew.pack_folder_for_nwb(folder=folder, prefix=prefix)

        for key, value in list(output.items()):

            if 'CH' in key:  # analog channel for electrode recording
                ch_ind = int(key[key.find('CH') + 2:])
                ch_name = 'ch_' + ft.int2str(ch_ind, 4)
                ch_trace = value['trace']
                ch_series = self.create_timeseries('ElectricalSeries', ch_name, 'acquisition')
                ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
                                   resolution=1.)
                ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
                                           rate=float(value['header']['sampleRate']))
                ch_series.set_value('electrode_idx', ch_ind)
                ch_series.set_value('num_samples', len(ch_trace))
                ch_series.set_comments('continuous')
                ch_series.set_description('extracellular continuous voltage recording from tetrode')
                ch_series.set_source('open ephys')
                ch_series.finalize()

            elif key != 'events':  # other continuous channels
                ch_name = key[len(prefix) + 1:]
                ch_trace = value['trace']
                ch_series = self.create_timeseries('AbstractFeatureSeries', ch_name, 'acquisition')
                ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
                                   resolution=1.)
                ch_series.set_time_by_rate(time_zero=0.0,  # value['header']['start_time'],
                                           rate=float(value['header']['sampleRate']))
                ch_series.set_value('features', ch_name)
                ch_series.set_value('feature_units', 'bit')
                ch_series.set_value('num_samples', len(ch_trace))
                ch_series.set_value('help', 'continuously recorded analog channels with same sampling times as '
                                            'of electrode recordings')
                ch_series.set_comments('continuous')
                ch_series.set_description('continuous voltage recording from IO board')
                ch_series.set_source('open ephys')
                ch_series.finalize()

    def add_acquisition_image(self, name, img, format='array', description=''):
        """
        add arbitrarily recorded image into acquisition group, mostly surface vasculature image
        :param name:
        :param img:
        :param format:
        :param description:
        :return:
        """
        img_dset = self.file_pointer['acquisition/images'].create_dataset(name, data=img)
        img_dset.attrs['format'] = format
        img_dset.attrs['description'] = description

    def add_acquired_image_series_as_remote_link(self, name, image_file_path, dataset_path, timestamps,
                                                 description='', comments='', data_format='zyx', pixel_size=np.nan,
                                                 pixel_size_unit=''):
        """
        add a required image series in to acquisition field as a link to an external hdf5 file.
        :param name: str, name of the image series
        :param image_file_path: str, the full file system path to the hdf5 file containing the raw image data
        :param dataset_path: str, the path within the hdf5 file pointing to the raw data. the object should have at
                             least 3 attributes: 'conversion', resolution, unit
        :param timestamps: 1-d array, the length of this array should be the same as number of frames in the image data
        :param data_format: str, required field for ImageSeries object
        :param pixel_size: array, size of pixel
        :param pixel_size_unit: str, unit of pixel size
        :return:
        """

        img_file = h5py.File(image_file_path)
        img_data = img_file[dataset_path]
        if timestamps.shape[0] != img_data.shape[0]:
            raise ValueError('Number of frames does not equal to the length of timestamps!')
        img_series = self.create_timeseries(ts_type='ImageSeries', name=name, modality='acquisition')
        img_series.set_data_as_remote_link(image_file_path, dataset_path)
        img_series.set_time(timestamps)
        img_series.set_description(description)
        img_series.set_comments(comments)
        img_series.set_value('bits_per_pixel', img_data.dtype.itemsize * 8)
        img_series.set_value('format', data_format)
        img_series.set_value('dimension', img_data.shape)
        img_series.set_value('image_file_path', image_file_path)
        img_series.set_value('image_data_path_within_file', dataset_path)
        img_series.set_value('pixel_size', pixel_size)
        img_series.set_value('pixel_size_unit', pixel_size_unit)
        img_series.finalize()

    def add_phy_template_clusters(self, folder, module_name, ind_start=None, ind_end=None,
                                  is_add_artificial_unit=False, artificial_unit_firing_rate=2.,
                                  spike_sorter=None):
        """
        extract phy-template clustering results to nwb format. Only extract spike times, no template for now.
        Usually the continuous channels of multiple files are concatenated for kilosort. ind_start and ind_end are
        Used to extract the data of this particular file.

        :param folder: folder containing phy template results.
                       expects cluster_groups.csv, spike_clusters.npy and spike_times.npy in the folder.
        :param module_name: str, name of clustering module group
        :param ind_start: int, the start index of continuous channel of the current file in the concatenated file.
        :param ind_end: int, the end index of continuous channel of the current file in the concatenated file.
        :param is_add_artificial_unit: bool, if True: a artificial unit with possion event will be added, this unit
                                       will have name 'aua' and refractory period 1 ms.
        :param artificial_unit_firing_rate: float, firing rate of the artificial unit
        :param spike_sorter: string, user ID of who manually sorted the clusters
        :return:
        """

        #  check integrity
        if ind_start == None:
            ind_start = 0

        if ind_end == None:
            ind_end = self.file_pointer['acquisition/timeseries/photodiode/num_samples'].value

        if ind_start >= ind_end:
            raise ValueError('ind_end should be larger than ind_start.')

        try:
            fs = self.file_pointer['general/extracellular_ephys/sampling_rate'].value
        except KeyError:
            print('\nCannot find "general/extracellular_ephys/sampling_rate" field. Abort process.')
            return

        #  get spike sorter
        if spike_sorter is None:
            spike_sorter = self.file_pointer['general/experimenter'].value

        #  set kilosort output file paths
        clusters_path = os.path.join(folder, 'spike_clusters.npy')
        spike_times_path = os.path.join(folder, 'spike_times.npy')

        #  generate dictionary of cluster timing indices
        try:
            #  for new version of kilosort
            phy_template_output = kw.get_clusters(kw.read_csv(os.path.join(folder, 'cluster_group.tsv')))
        except IOError:
            # for old version of kilosort
            phy_template_output = kw.get_clusters(kw.read_csv(os.path.join(folder, 'cluster_groups.csv')))

        spike_ind = kw.get_spike_times_indices(phy_template_output, spike_clusters_path=clusters_path,
                                               spike_times_path=spike_times_path)

        #  add artificial random unit
        if is_add_artificial_unit:
            file_length = (ind_end - ind_start) / fs
            au_ts = ta.possion_event_ts(duration=file_length, firing_rate=artificial_unit_firing_rate,
                                        refractory_dur=0.001, is_plot=False)
            spike_ind.update({'unit_aua': (au_ts * fs).astype(np.uint64) + ind_start})

        #  get channel related infomation
        ch_ns = self._get_channel_names()
        file_starting_time = self.get_analog_data(ch_ns[0])[1][0]
        channel_positions = kw.get_channel_geometry(folder, channel_names=ch_ns)

        #  create specificed module
        mod = self.create_module(name=module_name)
        mod.set_description('phy-template manual clustering after kilosort')
        mod.set_value('channel_list', [ch.encode('Utf8') for ch in ch_ns])
        mod.set_value('channel_xpos', [channel_positions[ch][0] for ch in ch_ns])
        mod.set_value('channel_ypos', [channel_positions[ch][1] for ch in ch_ns])

        #  create UnitTimes interface
        unit_times = mod.create_interface('UnitTimes')
        for unit in list(spike_ind.keys()):

            #  get timestamps of current unit
            curr_ts = spike_ind[unit]
            curr_ts = curr_ts[np.logical_and(curr_ts >= ind_start, curr_ts < ind_end)] - ind_start
            curr_ts = curr_ts / fs + file_starting_time

            # array to store waveforms from all channels
            template = []
            template_uf = []  # wavefrom from unfiltered analog signal

            # array to store standard deviations of waveform from all channels
            std = []
            std_uf = []  # standard deviations of unfiltered analog signal

            # temporary variables to detect peak channels
            peak_channel = None
            peak_channel_ind = None
            peak_amp = 0

            for i, ch_n in enumerate(ch_ns):

                #  get current analog signals of a given channel
                curr_ch_data, curr_ch_ts = self.get_analog_data(ch_n)

                #  band pass this analog signal
                curr_ch_data_f = ta.butter_highpass(curr_ch_data, cutoff=300., fs=fs)

                #  calculate spike triggered average filtered signal
                curr_waveform_results = ta.event_triggered_average_regular(ts_event=curr_ts,
                                                                           continuous=curr_ch_data_f,
                                                                           fs_continuous=fs,
                                                                           start_time_continuous=file_starting_time,
                                                                           t_range=SPIKE_WAVEFORM_TIMEWINDOW,
                                                                           is_normalize=True,
                                                                           is_plot=False)
                curr_waveform, curr_n, curr_t, curr_std = curr_waveform_results

                curr_waveform_results_uf = ta.event_triggered_average_regular(ts_event=curr_ts,
                                                                              continuous=curr_ch_data,
                                                                              fs_continuous=fs,
                                                                              start_time_continuous=file_starting_time,
                                                                              t_range=SPIKE_WAVEFORM_TIMEWINDOW,
                                                                              is_normalize=True,
                                                                              is_plot=False)
                curr_waveform_uf, _, _, curr_std_uf = curr_waveform_results_uf

                #  append waveform and std for current channel
                template.append(curr_waveform)
                std.append(curr_std)
                template_uf.append(curr_waveform_uf)
                std_uf.append(curr_std_uf)

                #  detect the channel with peak amplitude
                if peak_channel is not None:
                    peak_channel = ch_n
                    peak_channel_ind = i
                    peak_amp = np.max(curr_waveform) - np.min(curr_waveform)
                else:
                    if np.max(curr_waveform) - np.min(curr_waveform) > peak_amp:
                        peak_channel = ch_n
                        peak_channel_ind = i
                        peak_amp = np.max(curr_waveform) - np.min(curr_waveform)

            #  add 'UnitTimes' field
            if unit == 'unit_aua':
                unit_times.add_unit(unit_name='unit_aua', unit_times=curr_ts,
                                    source='electrophysiology extracellular recording',
                                    description='Artificial possion unit for control. Spike time unit: seconds. '
                                                'Spike waveforms are band-pass filtered with cutoff frequency'
                                                ' (300, 6000) Hz.')
            else:
                unit_times.add_unit(unit_name=unit, unit_times=curr_ts,
                                    source='electrophysiology extracellular recording',
                                    description="Data spike-sorted by: " + spike_sorter +
                                                ' using phy-template. Spike time unit: seconds. Spike waveforms are'
                                                'band-pass filtered with cutoff frequency (300, 6000) Hz.')

            #  add relevant information to current UnitTimes field
            unit_times.append_unit_data(unit_name=unit, key='channel_name', value=peak_channel)
            unit_times.append_unit_data(unit_name=unit, key='channel', value=peak_channel_ind)
            unit_times.append_unit_data(unit_name=unit, key='template_filtered', value=np.array(template).transpose())
            unit_times.append_unit_data(unit_name=unit, key='template',
                                        value=np.array(template_uf).transpose())
            unit_times.append_unit_data(unit_name=unit, key='template_std_filtered', value=np.array(std).transpose())
            unit_times.append_unit_data(unit_name=unit, key='template_std',
                                        value=np.array(std_uf).transpose())
            unit_times.append_unit_data(unit_name=unit, key='waveform', value=template_uf[peak_channel_ind])
            unit_times.append_unit_data(unit_name=unit, key='waveform_std', value=std_uf[peak_channel_ind])
            unit_times.append_unit_data(unit_name=unit, key='xpos_probe',
                                        value=[channel_positions[ch][0] for ch in ch_ns][peak_channel_ind])
            unit_times.append_unit_data(unit_name=unit, key='ypos_probe',
                                        value=[channel_positions[ch][1] for ch in ch_ns][peak_channel_ind])

        #  finalize
        unit_times.finalize()
        mod.finalize()

    def add_external_LFP(self,  traces, fs=30000., module_name=None, notch_base=60., notch_bandwidth=1., notch_harmonics=4,
                         notch_order=2, lowpass_cutoff=300., lowpass_order=5, resolution=0, conversion=0, unit='',
                        comments='', source=''):
        """
        add LFP of raw arbitrary electrical traces into LFP module into /procession field. the trace will be filtered
        by corticalmapping.HighLevel.get_lfp() function. All filters are butterworth digital filters

        :param module_name: str, name of module to be added
        :param traces: dict, {str: 1d-array}, {name: trace}, input raw traces
        :param fs: float, sampling rate, Hz
        :param notch_base: float, Hz, base frequency of powerline contaminating signal
        :param notch_bandwidth: float, Hz, filter bandwidth at each side of center frequency
        :param notch_harmonics: int, number of harmonics to filter out
        :param notch_order: int, order of butterworth bandpass notch filter, for a narrow band, shouldn't be larger than 2
        :param lowpass_cutoff: float, Hz, cutoff frequency of lowpass filter
        :param lowpass_order: int, order of butterworth lowpass filter
        :param resolution: float, resolution of LFP time series
        :param conversion: float, conversion of LFP time series
        :param unit: str, unit of LFP time series
        :param comments: str, interface comments
        :param source: str, interface source
        """

        if module_name is None or module_name=='':
            module_name = 'external_LFP'

        lfp = {}
        for tn, trace in list(traces.items()):
            curr_lfp = hl.get_lfp(trace,fs=fs, notch_base=notch_base, notch_bandwidth=notch_bandwidth,
                                  notch_harmonics=notch_harmonics, notch_order=notch_order,
                                  lowpass_cutoff=lowpass_cutoff, lowpass_order=lowpass_order)
            lfp.update({tn: curr_lfp})

        lfp_mod = self.create_module(module_name)
        lfp_mod.set_description('LFP from external traces')
        lfp_interface = lfp_mod.create_interface('LFP')
        lfp_interface.set_value('description', 'LFP of raw arbitrary electrical traces. The traces were filtered by '
                                'corticalmapping.HighLevel.get_lfp() function. First, the powerline contamination at '
                                'multiplt harmonics were filtered out by a notch filter. Then the resulting traces were'
                                ' filtered by a lowpass filter. All filters are butterworth digital filters')
        lfp_interface.set_value('comments', comments)
        lfp_interface.set_value('notch_base', notch_base)
        lfp_interface.set_value('notch_bandwidth', notch_bandwidth)
        lfp_interface.set_value('notch_harmonics', notch_harmonics)
        lfp_interface.set_value('notch_order', notch_order)
        lfp_interface.set_value('lowpass_cutoff', lowpass_cutoff)
        lfp_interface.set_value('lowpass_order', lowpass_order)
        lfp_interface.set_source(source)
        for tn, t_lfp in list(lfp.items()):
            curr_ts = self.create_timeseries('ElectricalSeries', tn, modality='other')
            curr_ts.set_data(t_lfp, conversion=conversion, resolution=resolution, unit=unit)
            curr_ts.set_time_by_rate(time_zero=0., rate=fs)
            curr_ts.set_value('num_samples', len(t_lfp))
            curr_ts.set_value('electrode_idx', 0)
            lfp_interface.add_timeseries(curr_ts)
            lfp_interface.finalize()

        lfp_mod.finalize()

    def add_internal_LFP(self, continuous_channels, module_name=None, notch_base=60., notch_bandwidth=1.,
                         notch_harmonics=4, notch_order=2, lowpass_cutoff=300., lowpass_order=5, comments='',
                         source=''):
        """
        add LFP of acquired electrical traces into LFP module into /procession field. the trace will be filtered
        by corticalmapping.HighLevel.get_lfp() function. All filters are butterworth digital filters.

        :param continuous_channels: list of strs, name of continuous channels saved in '/acquisition/timeseries'
                                    folder, the time axis of these channels should be saved by rate
                                    (ephys sampling rate).
        :param module_name: str, name of module to be added
        :param notch_base: float, Hz, base frequency of powerline contaminating signal
        :param notch_bandwidth: float, Hz, filter bandwidth at each side of center frequency
        :param notch_harmonics: int, number of harmonics to filter out
        :param notch_order: int, order of butterworth bandpass notch filter, for a narrow band, shouldn't be larger than 2
        :param lowpass_cutoff: float, Hz, cutoff frequency of lowpass filter
        :param lowpass_order: int, order of butterworth lowpass filter
        :param comments: str, interface comments
        :param source: str, interface source
        """

        if module_name is None or module_name=='':
            module_name = 'LFP'

        lfp_mod = self.create_module(module_name)
        lfp_mod.set_description('LFP from acquired electrical traces')
        lfp_interface = lfp_mod.create_interface('LFP')
        lfp_interface.set_value('description', 'LFP of acquired electrical traces. The traces were filtered by '
                                               'corticalmapping.HighLevel.get_lfp() function. First, the powerline '
                                               'contamination at multiplt harmonics were filtered out by a notch '
                                               'filter. Then the resulting traces were filtered by a lowpass filter. '
                                               'All filters are butterworth digital filters')
        lfp_interface.set_value('comments', comments)
        lfp_interface.set_value('notch_base', notch_base)
        lfp_interface.set_value('notch_bandwidth', notch_bandwidth)
        lfp_interface.set_value('notch_harmonics', notch_harmonics)
        lfp_interface.set_value('notch_order', notch_order)
        lfp_interface.set_value('lowpass_cutoff', lowpass_cutoff)
        lfp_interface.set_value('lowpass_order', lowpass_order)
        lfp_interface.set_source(source)

        for channel in continuous_channels:

            print('\n', channel, ': start adding LFP ...')

            trace = self.file_pointer['acquisition/timeseries'][channel]['data'].value
            fs = self.file_pointer['acquisition/timeseries'][channel]['starting_time'].attrs['rate']
            start_time = self.file_pointer['acquisition/timeseries'][channel]['starting_time'].value
            conversion = self.file_pointer['acquisition/timeseries'][channel]['data'].attrs['conversion']
            resolution = self.file_pointer['acquisition/timeseries'][channel]['data'].attrs['resolution']
            unit = self.file_pointer['acquisition/timeseries'][channel]['data'].attrs['unit']
            ts_source = self.file_pointer['acquisition/timeseries'][channel].attrs['source']

            print(channel, ': calculating LFP ...')

            t_lfp = hl.get_lfp(trace, fs=fs, notch_base=notch_base, notch_bandwidth=notch_bandwidth,
                               notch_harmonics=notch_harmonics, notch_order=notch_order, lowpass_cutoff=lowpass_cutoff,
                               lowpass_order=lowpass_order)

            curr_ts = self.create_timeseries('ElectricalSeries', channel, modality='other')
            curr_ts.set_data(t_lfp, conversion=conversion, resolution=resolution, unit=unit)
            curr_ts.set_time_by_rate(time_zero=start_time, rate=fs)
            curr_ts.set_value('num_samples', len(t_lfp))
            curr_ts.set_value('electrode_idx', int(channel.split('_')[1]))
            curr_ts.set_source(ts_source)
            lfp_interface.add_timeseries(curr_ts)
            print(channel, ': finished adding LFP.')

        lfp_interface.finalize()

        lfp_mod.finalize()

    def add_visual_stimulation(self, log_path, display_order=0):
        """
        load visual stimulation given saved display log pickle file
        :param log_path: the path to the display log generated by corticalmapping.VisualStim
        :param display_order: int, in case there is more than one visual display in the file.
                              This value records the order of the displays
        :return:
        """
        self._check_display_order(display_order)

        log_dict = ft.loadFile(log_path)

        stim_name = log_dict['stimulation']['stimName']

        display_frames = log_dict['presentation']['displayFrames']
        time_stamps = log_dict['presentation']['timeStamp']

        if len(display_frames) != len(time_stamps):
            print(('\nWarning: {}'.format(log_path)))
            print(('Unequal number of displayFrames ({}) and timeStamps ({}).'.format(len(display_frames),
                                                                                     len(time_stamps))))

        if stim_name == 'SparseNoise':
            self._add_sparse_noise_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'FlashingCircle':
            self._add_flashing_circle_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'UniformContrast':
            self._add_uniform_contrast_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'DriftingGratingCircle':
            self._add_drifting_grating_circle_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'KSstimAllDir':
            self._add_drifting_checker_board_stimulation(log_dict, display_order=display_order)
        else:
            raise ValueError('stimulation name {} unrecognizable!'.format(stim_name))

    def add_visual_stimulations(self, log_paths):

        exist_stimuli = list(self.file_pointer['stimulus/presentation'].keys())

        for i, log_path in enumerate(log_paths):
            self.add_visual_stimulation(log_path, i + len(exist_stimuli))

    @staticmethod
    def _analyze_sparse_noise_frames(sn_grp):
        """
        analyze sparse noise display frames saved in '/stimulus/presentation', extract information about onset of
        each displayed square:

        return: all_squares: 2-d array, each line is a displayed square in sparse noise, each column is a feature of
                             a particular square, squares follow display order
                data_format: str, description of the column structure of each square
                description: str,
                pooled_squares: dict, squares with same location and sign are pooled together.
                                keys: 'square_00000', 'square_00001', 'square_00002' ... each represents a unique
                                      square.
                                values: dict, {
                                               'azi': <azimuth of the square>,
                                               'alt': <altitude of the square>,
                                               'sign': <sign of the square>,
                                               'onset_ind': list of indices of the appearances of current square in
                                                            in "all_squares", to be aligned with to photodiode onset
                                                            timestamps
                                               }
        """

        if sn_grp['stim_name'].value != 'SparseNoise':
            raise NameError('The input stimulus should be "SparseNoise".')

        frames = sn_grp['data'].value
        frames = [tuple(x) for x in frames]
        dtype = [('isDisplay', int), ('azimuth', float), ('altitude', float), ('sign', int), ('isOnset', int)]
        frames = np.array(frames, dtype=dtype)

        all_squares = []
        for i in range(len(frames)):
            if frames[i]['isDisplay'] == 1 and \
                    (i == 0 or (frames[i - 1]['isOnset'] == -1 and frames[i]['isOnset'] == 1)):
                all_squares.append(np.array((i, frames[i]['azimuth'], frames[i]['altitude'], frames[i]['sign']),
                                             dtype=np.float32))

        all_squares = np.array(all_squares)

        pooled_squares = {}
        unique_squares = list(set([tuple(x[1:]) for x in all_squares]))
        for i, unique_square in enumerate(unique_squares):
            curr_square_n = 'square_' + ft.int2str(i, 5)
            curr_azi = unique_square[0]
            curr_alt = unique_square[1]
            curr_sign = unique_square[2]
            curr_onset_ind = []
            for j, give_square in enumerate(all_squares):
                if np.array_equal(give_square[1:], unique_square):
                    curr_onset_ind.append(j)
            pooled_squares.update({curr_square_n: {'azi': curr_azi,
                                                   'alt': curr_alt,
                                                   'sign': curr_sign,
                                                   'onset_ind': curr_onset_ind}})
        all_squares = np.array(all_squares)
        data_format = ['display frame indices for the onset of each square', 'azimuth of each square',
                       'altitude of each square', 'sign of each square']
        description = 'TimeSeries of sparse noise square onsets. Stimulus generated by ' \
                      'corticalmapping.VisualStim.SparseNoise class.'
        return all_squares, data_format, description, pooled_squares

    @staticmethod
    def _analyze_driftig_grating_frames(dg_grp):
        """
        analyze drifting grating display frames saved in '/stimulus/presentation', extract information about onset of
        each displayed grating:

        return: all_gratings: 2-d array, each line is a displayed square in sparse noise, each column is a feature of
                             a particular square, squares follow display order
                data_format: str, description of the column structure of each grating
                description: str,
                pooled_squares: dict, gratings with same parameters are pooled together.
                                keys: 'grating_00000', 'grating_00001', 'grating_00002' ... each represents a unique
                                      grating.
                                values: dict, {
                                               'sf': <spatial frequency of the grating>,
                                               'tf': <temporal frequency of the grating>,
                                               'direction': <moving direction of the grating>,
                                               'contrast': <contrast of the grating>,
                                               'radius': <radius of the grating>,
                                               'azi': <azimuth of the grating center>
                                               'alt': <altitude of the grating center>
                                               'onset_ind': list of indices of the appearances of current square in
                                                            in "all_squares", to be aligned with to photodiode onset
                                                            timestamps
                                               }
        """
        if dg_grp['stim_name'].value != 'DriftingGratingCircle':
            raise NameError('The input stimulus should be "DriftingGratingCircle".')

        frames = dg_grp['data'].value

        all_gratings = []
        for i in range(len(frames)):
            if frames[i][8] == 1 and (i == 0 or (frames[i - 1][8] == -1)):
                all_gratings.append(np.array((i, frames[i][2], frames[i][3], frames[i][4], frames[i][5], frames[i][6]),
                                            dtype=np.float32))

        all_gratings = np.array(all_gratings)

        pooled_gratings = {}
        unique_gratings = list(set([tuple(x[1:]) for x in all_gratings]))
        for i, unique_grating in enumerate(unique_gratings):
            curr_grating_n = 'grating_' + ft.int2str(i, 5)
            curr_sf = unique_grating[0]
            curr_tf = unique_grating[1]
            curr_dir = unique_grating[2]
            curr_con = unique_grating[3]
            curr_r = unique_grating[4]
            curr_onset_ind = []
            for j, given_grating in enumerate(all_gratings):
                if np.array_equal(given_grating[1:], unique_grating):
                    curr_onset_ind.append(j)
            pooled_gratings.update({curr_grating_n: {'sf': curr_sf,
                                                     'tf': curr_tf,
                                                     'dir': curr_dir,
                                                     'con': curr_con,
                                                     'r': curr_r,
                                                     'onset_ind': curr_onset_ind}})
        data_format = ['display frame indices for the onset of each square', 'spatial frequency (cyc/deg)',
                       'temporal frequency (Hz)', 'moving direction (arc)', 'contrast (%)', 'radius (deg)']
        description = 'TimeSeries of drifting grating circle onsets. Stimulus generated by ' \
                      'corticalmapping.VisualStim.SparseNoise class.'
        return all_gratings, data_format, description, pooled_gratings

    @staticmethod
    def _analyze_flashing_circle_frames(fc_grp):
        """
        analyze flashing circle display frames saved in '/stimulus/presentation', extract information about onset of
        each displayed circle:

        return: all_circles: 2-d array, each line is the onset of displayed circle, each column is a feature of
                             that circle, circles follow the display order
                data_format: str, description of the column structure of each circle
                description: str,
                pooled_circles: None
        """

        if fc_grp['stim_name'].value != 'FlashingCircle':
            raise NameError('The input stimulus should be "FlashingCircle".')

        frames = fc_grp['data'][:, 0]
        azi = fc_grp['center_azimuth_deg'].value
        alt = fc_grp['center_altitude_deg'].value
        color_c = fc_grp['center_color'].value
        color_b = fc_grp['background_color'].value
        radius = fc_grp['radius_deg'].value

        all_cirlces = []
        for i in range(len(frames)):
            if frames[i] == 1 and (i == 0 or (frames[i - 1] == 0)):
                all_cirlces.append(np.array((i, azi, alt, color_c, color_b, radius), dtype=np.float32))

        all_cirlces = np.array(all_cirlces)
        data_format = ['display frame indices for the onset of each circle', 'center_azimuth_deg',
                       'center_altitude_deg', 'center_color', 'background_color', 'radius_deg']
        description = 'TimeSeries of flashing circle onsets. Stimulus generated by ' \
                      'corticalmapping.VisualStim.SparseNoise class.'
        return all_cirlces, data_format, description, None

    @staticmethod
    def _analyze_uniform_contrast_frames(uc_grp):

        if uc_grp['stim_name'].value != 'UniformContrast':
            raise NameError('The input stimulus should be "UniformContrast".')

        onset_array = np.array([])
        data_format = ''
        description = 'TimeSeries of uniform contrast stimulus. No onset information. Stimulus generated by ' \
                      'corticalmapping.VisualStim.UniformContrast class.'
        return onset_array, data_format, description, {}

    def analyze_visual_stimuli(self, onsets_ts=None):
        """
        add stimuli onset timestamps of all saved stimulus presentations to 'processing/stimulus_onsets' module

        :param onsets_ts: 1-d array, timestamps of stimuli onsets. if None, it will look for
                          ['processing/photodiode/timestemps'] as onset_ts
        """

        if onsets_ts is None:
            print('input onsets_ts is None, try to use photodiode onsets as onsets_ts.')
            onsets_ts = self.file_pointer['processing/PhotodiodeOnsets/photodiode_onsets/timestamps'].value

        stim_ns = list(self.file_pointer['stimulus/presentation'].keys())
        stim_ns.sort()

        total_onsets = 0

        for stim_ind, stim_n in enumerate(stim_ns):

            if int(stim_n[0: 2]) != stim_ind:
                raise ValueError('Stimulus name: {} does not follow the order: {}'.format(stim_n, stim_ind))

            curr_stim_grp = self.file_pointer['stimulus/presentation'][stim_n]
            if curr_stim_grp['stim_name'].value == 'SparseNoise':
                _ = self._analyze_sparse_noise_frames(curr_stim_grp)
            elif curr_stim_grp['stim_name'].value == 'FlashingCircle':
                _ = self._analyze_flashing_circle_frames(curr_stim_grp)
            elif curr_stim_grp['stim_name'].value == 'DriftingGratingCircle':
                _ = self._analyze_driftig_grating_frames(curr_stim_grp)
            elif curr_stim_grp['stim_name'].value == 'UniformContrast':
                _ = self._analyze_uniform_contrast_frames(curr_stim_grp)
            else:
                raise LookupError('Do not understand stimulus type: {}.'.format(stim_n))

            curr_onset_arr, curr_data_format, curr_description, pooled_onsets = _

            total_onsets += curr_onset_arr.shape[0]

        if total_onsets != len(onsets_ts):
            raise ValueError('Number of stimuli onsets ({}) do not match the number of given onsets_ts ({}).'
                             .format(total_onsets, len(onsets_ts)))

        curr_onset_start_ind = 0

        for stim_ind, stim_n in enumerate(stim_ns):
            curr_stim_grp = self.file_pointer['stimulus/presentation'][stim_n]

            if curr_stim_grp['stim_name'].value == 'SparseNoise':
                _ = self._analyze_sparse_noise_frames(curr_stim_grp)
            elif curr_stim_grp['stim_name'].value == 'FlashingCircle':
                _ = self._analyze_flashing_circle_frames(curr_stim_grp)
            elif curr_stim_grp['stim_name'].value == 'DriftingGratingCircle':
                _ = self._analyze_driftig_grating_frames(curr_stim_grp)
            elif curr_stim_grp['stim_name'].value == 'UniformContrast':
                _ = self._analyze_uniform_contrast_frames(curr_stim_grp)
            else:
                raise LookupError('Do not understand stimulus type: {}.'.format(stim_n))

            curr_onset_arr, curr_data_format, curr_description, pooled_onsets = _

            curr_onset_ts = onsets_ts[curr_onset_start_ind: curr_onset_start_ind + curr_onset_arr.shape[0]]

            curr_onset = self.create_timeseries('TimeSeries', stim_n, modality='other')
            curr_onset.set_data(curr_onset_arr, unit='', conversion=np.nan, resolution=np.nan)
            curr_onset.set_time(curr_onset_ts)
            curr_onset.set_description(curr_description)
            curr_onset.set_value('data_format', curr_data_format)
            curr_onset.set_path('processing/StimulusOnsets')
            curr_onset.set_value('stim_name', curr_stim_grp['stim_name'].value)
            curr_onset.set_value('background_color', curr_stim_grp['background_color'].value)
            curr_onset.set_value('pre_gap_dur_sec', curr_stim_grp['pre_gap_dur_sec'].value)
            curr_onset.set_value('post_gap_dur_sec', curr_stim_grp['post_gap_dur_sec'].value)

            if curr_stim_grp['stim_name'].value == 'UniformContrast':
                curr_onset.set_value('color', curr_stim_grp['color'].value)
                curr_onset.finalize()

            elif curr_stim_grp['stim_name'].value == 'FlashingCircle':
                curr_onset.set_value('iteration', curr_stim_grp['iteration'].value)
                curr_onset.finalize()

            elif curr_stim_grp['stim_name'].value == 'SparseNoise':
                curr_onset.set_value('grid_space', curr_stim_grp['grid_space'].value)
                curr_onset.set_value('iteration', curr_stim_grp['iteration'].value)
                curr_onset.set_value('probe_frame_num', curr_stim_grp['probe_frame_num'].value)
                curr_onset.set_value('probe_height_deg', curr_stim_grp['probe_height_deg'].value)
                curr_onset.set_value('probe_orientation', curr_stim_grp['probe_orientation'].value)
                curr_onset.set_value('probe_width_deg', curr_stim_grp['probe_width_deg'].value)
                curr_onset.set_value('sign', curr_stim_grp['sign'].value)
                curr_onset.set_value('subregion_deg', curr_stim_grp['subregion_deg'].value)
                curr_onset.finalize()
                for curr_sn, curr_sd in list(pooled_onsets.items()):
                    curr_s_ts = self.create_timeseries('TimeSeries', curr_sn, modality='other')
                    curr_s_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
                    curr_s_ts.set_time(curr_onset_ts[curr_sd['onset_ind']])
                    curr_s_ts.set_value('azimuth_deg', curr_sd['azi'])
                    curr_s_ts.set_value('altitude_deg', curr_sd['alt'])
                    curr_s_ts.set_value('sign', curr_sd['sign'])
                    curr_s_ts.set_path('processing/StimulusOnsets/' + stim_n + '/square_timestamps')
                    curr_s_ts.finalize()

            elif curr_stim_grp['stim_name'].value == 'DriftingGratingCircle':
                curr_onset.set_value('iteration', curr_stim_grp['iteration'].value)
                curr_onset.set_value('mid_gap_dur_sec', curr_stim_grp['mid_gap_dur_sec'].value)
                curr_onset.set_value('center_altitude_deg', curr_stim_grp['center_altitude_deg'].value)
                curr_onset.set_value('center_azimuth_deg', curr_stim_grp['center_azimuth_deg'].value)
                curr_onset.set_value('contrast_list', curr_stim_grp['contrast_list'].value)
                curr_onset.set_value('direction_list', curr_stim_grp['direction_list'].value)
                curr_onset.set_value('radius_list', curr_stim_grp['radius_list'].value)
                curr_onset.set_value('spatial_frequency_list', curr_stim_grp['spatial_frequency_list'].value)
                curr_onset.set_value('temporal_frequency_list', curr_stim_grp['temporal_frequency_list'].value)
                curr_onset.finalize()
                for curr_gn, curr_gd in list(pooled_onsets.items()):
                    curr_g_ts = self.create_timeseries('TimeSeries', curr_gn, modality='other')
                    curr_g_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
                    curr_g_ts.set_time(curr_onset_ts[curr_gd['onset_ind']])
                    curr_g_ts.set_value('sf_cyc_per_deg', curr_gd['sf'])
                    curr_g_ts.set_value('tf_hz', curr_gd['tf'])
                    curr_g_ts.set_value('direction_arc', curr_gd['dir'])
                    curr_g_ts.set_value('contrast', curr_gd['con'])
                    curr_g_ts.set_value('radius_deg', curr_gd['r'])
                    curr_g_ts.set_path('processing/StimulusOnsets/' + stim_n + '/grating_timestamps')
                    curr_g_ts.finalize()

            curr_onset_start_ind = curr_onset_start_ind + curr_onset_arr.shape[0]

    def add_photodiode_onsets(self, digitizeThr=0.9, filterSize=0.01, segmentThr=0.01, smallestInterval=0.03,
                              expected_onsets_number=None):
        """
        intermediate processing step for analysis of visual display. Containing the information about the onset of
        photodiode signal. Timestamps are extracted from photodiode signal, should be aligned to the master clock.
        extraction is done by corticalmapping.HighLevel.segmentPhotodiodeSignal() function. The raw signal
        was first digitized by the digitize_threshold, then filtered by a gaussian fileter with filter_size. Then
        the derivative of the filtered signal was calculated by numpy.diff. The derivative signal was then timed
        with the digitized signal. Then the segmentation_threshold was used to detect rising edge of the resulting
        signal. Any onset with interval from its previous onset smaller than smallest_interval will be discarded.
        the resulting timestamps of photodiode onsets will be saved in 'processing/photodiode_onsets' timeseries

        :param digitizeThr: float
        :param filterSize: float
        :param segmentThr: float
        :param smallestInterval: float
        :param expected_onsets_number: int, expected number of photodiode onsets, may extract from visual display
                                       log. if extracted onset number does not match this number, the process will
                                       be abort. If None, no such check will be performed.
        :return:
        """
        fs = self.file_pointer['acquisition/timeseries/photodiode/starting_time'].attrs['rate']
        pd = self.file_pointer['acquisition/timeseries/photodiode/data'].value * \
             self.file_pointer['acquisition/timeseries/photodiode/data'].attrs['conversion']

        pd_onsets = hl.segmentPhotodiodeSignal(pd, digitizeThr=digitizeThr, filterSize=filterSize,
                                               segmentThr=segmentThr, Fs=fs, smallestInterval=smallestInterval)

        if pd_onsets.shape[0] == 0:
            return

        if expected_onsets_number is not None:
            if len(pd_onsets) != expected_onsets_number:
                raise ValueError('The number of photodiode onsets (' + str(len(pd_onsets)) + ') and the expected '
                                 'number of sweeps ' + str(expected_onsets_number) + ' do not match. Abort.')

        pd_ts = self.create_timeseries('TimeSeries', 'photodiode_onsets', modality='other')
        pd_ts.set_time(pd_onsets)
        pd_ts.set_data([], unit='', conversion=np.nan, resolution=np.nan)
        pd_ts.set_description('intermediate processing step for analysis of visual display. '
                              'Containing the information about the onset of photodiode signal. Timestamps '
                              'are extracted from photodiode signal, should be aligned to the master clock.'
                              'extraction is done by corticalmapping.HighLevel.segmentPhotodiodeSignal()'
                              'function. The raw signal was first digitized by the digitize_threshold, then '
                              'filtered by a gaussian fileter with filter_size. Then the derivative of the filtered '
                              'signal was calculated by numpy.diff. The derivative signal was then timed with the '
                              'digitized signal. Then the segmentation_threshold was used to detect rising edge of '
                              'the resulting signal. Any onset with interval from its previous onset smaller than '
                              'smallest_interval will be discarded.')
        pd_ts.set_path('/processing/PhotodiodeOnsets')
        pd_ts.set_value('digitize_threshold', digitizeThr)
        pd_ts.set_value('fileter_size', filterSize)
        pd_ts.set_value('segmentation_threshold', segmentThr)
        pd_ts.set_value('smallest_interval', smallestInterval)
        pd_ts.finalize()

    def plot_spike_waveforms(self, modulen, unitn, is_plot_filtered=False, fig=None, axes_size=(0.2, 0.2), **kwargs):
        """
        plot spike waveforms

        :param modulen: str, name of the module containing ephys recordings
        :param unitn: str, name of ephys unit, should be in '/processing/ephys_units/UnitTimes'
        :param is_plot_filtered: bool, plot unfiltered waveforms or not
        :param channel_names: list of strs, channel names in continuous recordings, should be in '/acquisition/timeseries'
        :param fig: matplotlib figure object
        :param t_range: tuple of two floats, time range to plot along spike time stamps
        :param kwargs: inputs to matplotlib.axes.plot() function
        :return: fig
        """
        if modulen not in list(self.file_pointer['processing'].keys()):
            raise LookupError('Can not find module for ephys recording: ' + modulen + '.')

        if unitn not in list(self.file_pointer['processing'][modulen]['UnitTimes'].keys()):
            raise LookupError('Can not find ephys unit: ' + unitn + '.')

        ch_ns = self._get_channel_names()

        unit_grp = self.file_pointer['processing'][modulen]['UnitTimes'][unitn]
        waveforms = unit_grp['template'].value

        if 'template_std' in list(unit_grp.keys()):
            stds = unit_grp['template_std'].value
        else:
            stds = None

        if is_plot_filtered:
            if 'template_filtered' in list(unit_grp.keys()):
                waveforms_f = unit_grp['template_filtered'].value
                if 'template_std_filtered' in list(unit_grp.keys()):
                    stds_f = unit_grp['template_std_filtered'].value
                else:
                    stds_f = None
            else:
                print(('can not find unfiltered spike waveforms for unit: ' + unitn))
                waveforms_f = None
                stds_f = None
        else:
            waveforms_f = None
            stds_f = None

        if 'channel_xpos' in list(self.file_pointer['processing'][modulen].keys()):
            ch_xpos = self.file_pointer['processing'][modulen]['channel_xpos']
            ch_ypos = self.file_pointer['processing'][modulen]['channel_ypos']
            ch_locations = list(zip(ch_xpos, ch_ypos))
        else:
            ch_locations = None

        fig = plot_waveforms(waveforms, ch_locations=ch_locations, stds=stds, waveforms_filtered=waveforms_f,
                             stds_filtered=stds_f, f=fig, ch_ns=ch_ns, axes_size=axes_size, **kwargs)

        fig.suptitle(self.file_pointer['identifier'].value + ' : ' + unitn)

        return fig

    def add_motion_correction_module(self, module_name, original_timeseries_path, corrected_file_path,
                                     corrected_dataset_path, xy_translation_offsets, interface_name='MotionCorrection',
                                     mean_projection=None, max_projection=None, description='', comments='',
                                     source=''):
        """
        add a motion corrected image series in to processing field as a module named 'motion_correction' and create a
        link to an external hdf5 file which contains the images.
        :param module_name: str, module name to be created
        :param interface_name: str, interface name of the image series
        :param original_timeseries_path: str, the path to the timeseries of the original images
        :param corrected_file_path: str, the full file system path to the hdf5 file containing the raw image data
        :param corrected_dataset_path: str, the path within the hdf5 file pointing to the raw data. the object should have at
                             least 3 attributes: 'conversion', resolution, unit
        :param xy_translation_offsets: 2d array with two columns,
        :param mean_projection: 2d array, mean_projection of corrected image, if None, no dataset will be
                                created
        :param max_projection: 2d array,  max_projection of corrected image, if None, no dataset will be
                                created
        :return:
        """

        orig = self.file_pointer[original_timeseries_path]
        timestamps = orig['timestamps'].value

        img_file = h5py.File(corrected_file_path)
        img_data = img_file[corrected_dataset_path]
        if timestamps.shape[0] != img_data.shape[0]:
            raise ValueError('Number of frames does not equal to the length of timestamps!')

        if xy_translation_offsets.shape[0] != timestamps.shape[0]:
            raise ValueError('Number of offsets does not equal to the length of timestamps!')

        corrected = self.create_timeseries(ts_type='ImageSeries', name='corrected', modality='other')
        corrected.set_data_as_remote_link(corrected_file_path, corrected_dataset_path)
        corrected.set_time_as_link(original_timeseries_path)
        corrected.set_description(description)
        corrected.set_comments(comments)
        corrected.set_source(source)
        for value_n in list(orig.keys()):
            if value_n not in ['image_data_path_within_file', 'image_file_path', 'data', 'timestamps']:
                corrected.set_value(value_n, orig[value_n].value)

        xy_translation = self.create_timeseries(ts_type='TimeSeries', name='xy_translation', modality='other')
        xy_translation.set_data(xy_translation_offsets, unit='pixel', conversion=np.nan, resolution=np.nan)
        xy_translation.set_time_as_link(original_timeseries_path)
        xy_translation.set_value('num_samples', xy_translation_offsets.shape[0])
        xy_translation.set_description('Time series of x, y shifts applied to create motion stabilized image series')
        xy_translation.set_value('feature_description', ['x_motion', 'y_motion'])

        mc_mod = self.create_module(module_name)
        mc_interf = mc_mod.create_interface("MotionCorrection")
        mc_interf.add_corrected_image(interface_name, orig=original_timeseries_path, xy_translation=xy_translation,
                                      corrected=corrected)

        if mean_projection is not None:
            mc_interf.set_value('mean_projection', mean_projection)

        if max_projection is not None:
            mc_interf.set_value('max_projection', max_projection)

        mc_interf.finalize()
        mc_mod.finalize()

    def generate_dat_file_for_kilosort(self, output_folder, output_name, ch_ns, is_filtered=True, cutoff_f_low=300.,
                                       cutoff_f_high=6000.):
        """
        generate .dat file for kilolsort: "https://github.com/cortex-lab/KiloSort", it is binary raw code, with
        structure: ch0_t0, ch1_t0, ch2_t0, ...., chn_t0, ch0_t1, ch1_t1, ch2_t1, ..., chn_t1, ..., ch0_tm, ch1_tm,
        ch2_tm, ..., chn_tm

        :param output_folder: str, path to output directory
        :param output_name: str, output file name, an extension of '.dat' will be automatically added.
        :param ch_ns: list of strings, name of included analog channels
        :param is_filtered: bool, if Ture, another .dat file with same size will be generated in the output folder.
                            this file will contain temporally filtered data (filter done by
                            corticalmapping.core.TimingAnalysis.butter_... functions). '_filtered' will be attached
                            to the filtered file name.
        :param cutoff_f_low: float, low cutoff frequency, Hz. if None, it will be low-pass
        :param cutoff_f_high: float, high cutoff frequency, Hz, if None, it will be high-pass
        :return: None
        """

        save_path = os.path.join(output_folder, output_name + '.dat')
        if os.path.isfile(save_path):
            raise IOError('Output file already exists.')

        data_lst = []
        for ch_n in ch_ns:
            data_lst.append(self.file_pointer['acquisition/timeseries'][ch_n]['data'].value)

        dtype = data_lst[0].dtype
        data = np.array(data_lst, dtype=dtype).flatten(order='F')
        data.tofile(save_path)

        if is_filtered:

            if cutoff_f_low is None and cutoff_f_high is None:
                print ('both low cutoff frequency and high cutoff frequency are None. Do nothing.')
                return

            save_path_f = os.path.join(output_folder, output_name + '_filtered.dat')
            if os.path.isfile(save_path_f):
                raise IOError('Output file for filtered data already existes.')

            fs = self.file_pointer['general/extracellular_ephys/sampling_rate'].value
            data_lst_f = []
            for data_r in data_lst:
                if cutoff_f_high is None:
                    data_lst_f.append(ta.butter_lowpass(data_r, fs=fs, cutoff=cutoff_f_low).astype(dtype))
                elif cutoff_f_low is None:
                    data_lst_f.append(ta.butter_highpass(data_r, fs=fs, cutoff=cutoff_f_high).astype(dtype))
                else:
                    data_lst_f.append(ta.butter_bandpass(data_r,
                                                         fs=fs,
                                                         cutoffs=(cutoff_f_low, cutoff_f_high)).astype(dtype))
            data_f = np.array(data_lst_f, dtype=dtype).flatten(order='F')
            data_f.tofile(save_path_f)

    def _get_channel_names(self):
        """
        :return: sorted list of channel names, each channel name should have prefix 'ch_'
        """
        analog_chs = list(self.file_pointer['acquisition/timeseries'].keys())
        channel_ns = [cn for cn in analog_chs if cn[0:3] == 'ch_']
        channel_ns.sort()
        return channel_ns

    def get_analog_data(self, ch_n):
        """
        :param ch_n: string, analog channel name
        :return: 1-d array, analog data, data * conversion
                 1-d array, time stamps
        """
        grp = self.file_pointer['acquisition/timeseries'][ch_n]
        data = grp['data'].value
        if not np.isnan(grp['data'].attrs['conversion']):
            data = data.astype(np.float32) * grp['data'].attrs['conversion']
        if 'timestamps' in list(grp.keys()):
            t = grp['timestamps']
        elif 'starting_time' in list(grp.keys()):
            fs = self.file_pointer['general/extracellular_ephys/sampling_rate'].value
            sample_num = grp['num_samples'].value
            t = np.arange(sample_num) / fs + grp['starting_time'].value
        else:
            raise ValueError('can not find timing information of channel:' + ch_n)
        return data, t

    def _check_display_order(self, display_order=None):
        """
        check display order make sure each presentation has a unique position, and move from increment order.
        also check the given display_order is of the next number
        """
        stimuli = list(self.file_pointer['stimulus/presentation'].keys())

        print('\nExisting visual stimuli:')
        print(('\n'.join(stimuli)))

        stimuli = [int(s[0:s.find('_')]) for s in stimuli]
        stimuli.sort()
        if stimuli != list(range(len(stimuli))):
            raise ValueError('display order is not incremental.')

        if display_order is not None:

            if display_order != len(stimuli):
                raise ValueError('input display order not the next display.')

    def _add_sparse_noise_stimulation(self, log_dict, display_order):

        stim_name = log_dict['stimulation']['stimName']

        if stim_name != 'SparseNoise':
            raise ValueError('stimulus was not sparse noise.')

        display_frames = log_dict['presentation']['displayFrames']
        time_stamps = log_dict['presentation']['timeStamp']

        frame_array = np.empty((len(display_frames), 5), dtype=np.float32)
        for i, frame in enumerate(display_frames):
            if frame[0] == 0:
                frame_array[i] = np.array([0, np.nan, np.nan, np.nan, frame[3]])
            elif frame[0] == 1:
                frame_array[i] = np.array([1, frame[1][0], frame[1][1], frame[2], frame[3]])
            else:
                raise ValueError('The first value of ' + str(i) + 'th display frame: ' + str(frame) + ' should' + \
                                 ' be only 0 or 1.')
        stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
                                      'stimulus')
        stim.set_time(time_stamps)
        stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
        stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
                          'this particular display, not the master time clock. For more useful timestamps, check'
                          '/processing for aligned photodiode onset timestamps.')
        stim.set_description('data formatting: [isDisplay (0:gap; 1:display), azimuth (deg), altitude (deg), '
                             'polarity (from -1 to 1), indicatorColor (for photodiode, from -1 to 1)]')
        stim.set_value('data_formatting', ['isDisplay', 'azimuth', 'altitude', 'polarity', 'indicatorColor'])
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
        stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
        stim.set_value('iteration', log_dict['stimulation']['iteration'])
        stim.set_value('sign', log_dict['stimulation']['sign'])
        stim.set_value('probe_width_deg', log_dict['stimulation']['probeSize'][0])
        stim.set_value('probe_height_deg', log_dict['stimulation']['probeSize'][1])
        stim.set_value('subregion_deg', log_dict['stimulation']['subregion'])
        try:
            stim.set_value('probe_orientation', log_dict['stimulation']['probeOrientation'])
        except KeyError:
            stim.set_value('probe_orientation', log_dict['stimulation']['probeOrientationt'])
        stim.set_value('stim_name', log_dict['stimulation']['stimName'])
        stim.set_value('grid_space', log_dict['stimulation']['gridSpace'])
        stim.set_value('probe_frame_num', log_dict['stimulation']['probeFrameNum'])
        stim.set_source('corticalmapping.VisualStim.SparseNoise for stimulus; '
                        'corticalmapping.VisualStim.DisplaySequence for display')
        stim.finalize()

    def _add_flashing_circle_stimulation(self, log_dict, display_order):

        stim_name = log_dict['stimulation']['stimName']

        if stim_name != 'FlashingCircle':
            raise ValueError('stimulus should be flashing circle.')

        display_frames = log_dict['presentation']['displayFrames']
        time_stamps = log_dict['presentation']['timeStamp']

        frame_array = np.empty((len(display_frames), 2), dtype=np.int8)
        for i, frame in enumerate(display_frames):
            if frame[0] == 0 or frame[0] == 1:
                frame_array[i] = np.array([frame[0], frame[3]])
            else:
                raise ValueError('The first value of ' + str(i) + 'th display frame: ' + str(frame) + ' should' + \
                                 ' be only 0 or 1.')
        stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
                                      'stimulus')
        stim.set_time(time_stamps)
        stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
        stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
                          'this particular display, not the master time clock. For more useful timestamps, check'
                          '/processing for aligned photodiode onset timestamps.')
        stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
                             'indicatorColor (for photodiode, from -1 to 1)]')
        stim.set_value('data_formatting', ['isDisplay', 'indicatorColor'])
        stim.set_source('corticalmapping.VisualStim.FlashingCircle for stimulus; '
                        'corticalmapping.VisualStim.DisplaySequence for display')
        stim.set_value('radius_deg', log_dict['stimulation']['radius'])
        stim.set_value('center_azimuth_deg', log_dict['stimulation']['center'][0])
        stim.set_value('center_altitude_deg', log_dict['stimulation']['center'][1])
        stim.set_value('center_color', log_dict['stimulation']['color'])
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.set_value('stim_name', log_dict['stimulation']['stimName'])
        stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
        stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
        stim.set_value('iteration', log_dict['stimulation']['iteration'])
        stim.finalize()

    def _add_uniform_contrast_stimulation(self, log_dict, display_order):

        stim_name = log_dict['stimulation']['stimName']

        if stim_name != 'UniformContrast':
            raise ValueError('stimulus should be uniform contrast.')

        display_frames = log_dict['presentation']['displayFrames']
        time_stamps = log_dict['presentation']['timeStamp']

        frame_array = np.array(display_frames, dtype=np.int8)

        stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
                                      'stimulus')
        stim.set_time(time_stamps)
        stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
        stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
                          'this particular display, not the master time clock. For more useful timestamps, check'
                          '/processing for aligned photodiode onset timestamps.')
        stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
                             'indicatorColor (for photodiode, from -1 to 1)]')
        stim.set_value('data_formatting', ['isDisplay', 'indicatorColor'])
        stim.set_source('corticalmapping.VisualStim.UniformContrast for stimulus; '
                        'corticalmapping.VisualStim.DisplaySequence for display')
        stim.set_value('color', log_dict['stimulation']['color'])
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
        stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
        stim.set_value('stim_name', log_dict['stimulation']['stimName'])
        stim.finalize()

    def _add_drifting_grating_circle_stimulation(self, log_dict, display_order):

        stim_name = log_dict['stimulation']['stimName']

        if stim_name != 'DriftingGratingCircle':
            raise ValueError('stimulus should be drifting grating circle.')

        display_frames = log_dict['presentation']['displayFrames']
        time_stamps = log_dict['presentation']['timeStamp']

        frame_array = np.array(display_frames)
        frame_array[np.equal(frame_array, None)] = np.nan
        frame_array = frame_array.astype(np.float32)

        stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
                                      'stimulus')
        stim.set_time(time_stamps)
        stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
        stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
                          'this particular display, not the master time clock. For more useful timestamps, check'
                          '/processing for aligned photodiode onset timestamps.')
        stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
                             'firstFrameInCycle (first frame in cycle:1, rest display frames: 0), '
                             'spatialFrequency (cyc/deg), '
                             'temporalFrequency (Hz), '
                             'direction ([0, 2*pi)), '
                             'contrast ([0, 1]), '
                             'radius (deg), '
                             'phase ([0, 2*pi)'
                             'indicatorColor (for photodiode, from -1 to 1)]. '
                             'for gap frames, the 2ed to 8th elements should be np.nan.')
        stim.set_value('data_formatting', ['isDisplay', 'firstFrameInCycle', 'spatialFrequency', 'temporalFrequency',
                                           'direction', 'contrast', 'radius', 'phase', 'indicatorColor'])
        stim.set_value('spatial_frequency_list', log_dict['stimulation']['sf_list'])
        stim.set_value('temporal_frequency_list', log_dict['stimulation']['tf_list'])
        stim.set_value('direction_list', log_dict['stimulation']['dire_list'])
        stim.set_value('contrast_list', log_dict['stimulation']['con_list'])
        stim.set_value('radius_list', log_dict['stimulation']['size_list'])
        stim.set_value('center_azimuth_deg', log_dict['stimulation']['center'][0])
        stim.set_value('center_altitude_deg', log_dict['stimulation']['center'][1])
        stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
        stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
        stim.set_value('mid_gap_dur_sec', log_dict['stimulation']['midGapDur'])
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.set_value('iteration', log_dict['stimulation']['iteration'])
        stim.set_value('stim_name', log_dict['stimulation']['stimName'])
        stim.set_value('sweep_dur_sec', log_dict['stimulation']['blockDur'])
        stim.set_source('corticalmapping.VisualStim.DriftingGratingCircle for stimulus; '
                        'corticalmapping.VisualStim.DisplaySequence for display')
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.finalize()

    def _add_drifting_checker_board_stimulation(self, log_dict, display_order):

        stim_name = log_dict['stimulation']['stimName']

        if stim_name != 'KSstimAllDir':
            raise ValueError('stimulus should be drifting checker board all directions.')

        display_frames = log_dict['presentation']['displayFrames']
        time_stamps = log_dict['presentation']['timeStamp']

        display_frames = [list(f) for f in display_frames]

        for i in range(len(display_frames)):
            if display_frames[i][4] == 'B2U':
                display_frames[i][4] = 0
            elif display_frames[i][4] == 'U2B':
                display_frames[i][4] = 1
            elif display_frames[i][4] == 'L2R':
                display_frames[i][4] = 2
            elif display_frames[i][4] == 'R2L':
                display_frames[i][4] = 3

        frame_array = np.array(display_frames)
        frame_array[np.equal(frame_array, None)] = np.nan
        frame_array = frame_array.astype(np.float32)

        stim = self.create_timeseries('TimeSeries', ft.int2str(display_order, 2) + '_' + stim_name,
                                      'stimulus')
        stim.set_time(time_stamps)
        stim.set_data(frame_array, unit='', conversion=np.nan, resolution=np.nan)
        stim.set_comments('the timestamps of displayed frames (saved in data) are referenced to the start of'
                          'this particular display, not the master time clock. For more useful timestamps, check'
                          '/processing for aligned photodiode onset timestamps.')
        stim.set_description('data formatting: [isDisplay (0:gap; 1:display), '
                             'square polarity (1: not reversed; -1: reversed), '
                             'sweeps, ind, index in sweep table, '
                             'indicatorColor (for photodiode, from -1 to 1)]. '
                             'direction (B2U: 0, U2B: 1, L2R: 2, R2L: 3), '
                             'for gap frames, the 2ed to 3th elements should be np.nan.')
        stim.set_value('data_formatting', ['isDisplay', 'squarePolarity', 'sweepIndex', 'indicatorColor', 'sweepDirection'])
        stim.set_source('corticalmapping.VisualStim.KSstimAllDir for stimulus; '
                        'corticalmapping.VisualStim.DisplaySequence for display')
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.set_value('pre_gap_dur_sec', log_dict['stimulation']['preGapDur'])
        stim.set_value('post_gap_dur_sec', log_dict['stimulation']['postGapDur'])
        stim.set_value('iteration', log_dict['stimulation']['iteration'])
        stim.set_value('stim_name', log_dict['stimulation']['stimName'])
        stim.finalize()

        display_info = hl.analysisMappingDisplayLog(display_log=log_dict)
        display_grp = self.file_pointer['processing'].create_group('mapping_display_info')
        display_grp.attrs['description'] = 'This group saves the useful infomation about the retiotopic mapping visual' \
                                           'stimulation (drifting checker board sweeps in all directions). Generated ' \
                                           'by the corticalmapping.HighLevel.analysisMappingDisplayLog() function.'
        for direction, value in list(display_info.items()):
            dir_grp = display_grp.create_group(direction)
            dir_grp.attrs['description'] = 'group containing the relative information about all sweeps in a particular' \
                                           'sweep direction. B: bottom, U: up, L: nasal, R: temporal (for stimulus to' \
                                           'the right eye)'
            ind_dset = dir_grp.create_dataset('onset_index', data=value['ind'])
            ind_dset.attrs['description'] = 'indices of sweeps of current direction in the whole experiment'
            st_dset = dir_grp.create_dataset('start_time', data=value['startTime'])
            st_dset.attrs['description'] = 'sweep start time relative to stimulus onset (second)'
            sd_dset = dir_grp.create_dataset('sweep_duration', data=value['sweepDur'])
            sd_dset.attrs['description'] = 'sweep duration (second)'
            equ_dset = dir_grp.create_dataset('phase_retinotopy_equation', data=[value['slope'], value['intercept']])
            equ_dset.attrs['description'] = 'the linear equation to transform fft phase into retinotopy visual degrees.' \
                                            'degree = phase * slope + intercept'
            equ_dset.attrs['data_format'] = ['slope', 'intercept']

    def add_sync_data(self):
        # not for now
        pass

    def add_kilosort_clusters(self, folder, module_name, ind_start=None, ind_end=None):
        """
        expects spike clusters.npy, spike_templates.npy, and spike_times.npy in the folder. use only for the direct outputs of kilosort,
        that haven't been modified with phy-template.
        :param folder:
        :return:
        """

        # if ind_start == None:
        #     ind_start = 0
        #
        # if ind_end == None:
        #     ind_end = self.file_pointer['acquisition/timeseries/photodiode/num_samples'].value
        #
        # if ind_start >= ind_end:
        #     raise ValueError('ind_end should be larger than ind_start.')
        #
        # spike_clusters = np.load(os.path.join(folder, 'spike_clusters.npy'))
        # spike_templates = np.load(os.path.join(folder, 'spike_templates.npy'))
        # spikes_times = np.load(os.path.join(folder, 'spike_times.npy'))
        # templates = np.load(os.path.join(folder, 'templates.npy'))

        # not for now
        pass


if __name__ == '__main__':

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # open_ephys_folder = r"E:\data\2016-07-19-160719-M256896\100_spontaneous_2016-07-19_09-45-06_Jun"
    # rf = RecordedFile(tmp_path, identifier='', description='')
    # rf.add_open_ephys_data(open_ephys_folder, '100', ['wf_read', 'wf_trigger', 'visual_frame'])
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # rf = RecordedFile(tmp_path)
    # rf.add_general()
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # rf = RecordedFile(tmp_path)
    # rf.add_acquisition_image('surface_vas_map', np.zeros((10, 10)), description='surface vasculature map')
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # data_path = r"E:\data\2016-07-25-160722-M256896\processed_1"
    # rf = RecordedFile(tmp_path)
    # rf.add_phy_template_clusters(folder=data_path, module_name='LGN')
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # data_path = r"E:\data\2016-07-25-160722-M256896\processed_1"
    # rf = RecordedFile(tmp_path)
    # rf.add_kilosort_clusters(folder=data_path, module_name='LGN_kilosort')
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_path = r"E:\data\2016-06-29-160610-M240652-Ephys\101_160610172256-SparseNoise-M240652-Jun-0-" \
    #            r"notTriggered-complete.pkl"
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulation(log_path)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_path = r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\161017-M274376-FlashingCircle" \
    #            r"\161017162026-FlashingCircle-M274376-Sahar-101-Triggered-complete.pkl"
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulation(log_path, display_order=1)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_paths = [r"\\aibsdata2\nc-ophys\CorticalMapping\IntrinsicImageData\161017-M274376-FlashingCircle\161017162026-FlashingCircle-M274376-Sahar-101-Triggered-complete.pkl",
    #              r"E:\data\2016-06-29-160610-M240652-Ephys\101_160610172256-SparseNoise-M240652-Jun-0-notTriggered-complete.pkl",]
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulations(log_paths)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # log_paths = [r"C:\data\sequence_display_log\161018164347-UniformContrast-MTest-Jun-255-notTriggered-complete.pkl"]
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulations(log_paths)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # # log_paths = [r"C:\data\sequence_display_log\160205131514-ObliqueKSstimAllDir-MTest-Jun-255-notTriggered-incomplete.pkl"]
    # log_paths = [r"C:\data\sequence_display_log\161018174812-DriftingGratingCircle-MTest-Jun-255-notTriggered-complete.pkl"]
    # rf = RecordedFile(tmp_path)
    # rf.add_visual_stimulations(log_paths)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    # img_data_path = r"E:\data\python_temp_folder\img_data.hdf5"
    # img_data = h5py.File(img_data_path)
    # dset = img_data.create_dataset('data', data=np.random.rand(1000, 1000, 100))
    # dset.attrs['conversion'] = np.nan
    # dset.attrs['resolution'] = np.nan
    # dset.attrs['unit'] = ''
    # img_data.close()

    # ts = np.random.rand(1000)
    #
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # rf = RecordedFile(tmp_path)
    # rf.add_acquired_image_series_as_remote_link('test_img', image_file_path=img_data_path, dataset_path='/data',
    #                                             timestamps=ts)
    # rf.close()
    # =========================================================================================================

    # =========================================================================================================
    rf = RecordedFile(r"D:\data2\thalamocortical_project\method_development\2017-02-25-ephys-software-development"
                      r"\test_folder\170302_M292070_100_SparseNoise.nwb")
    unit = 'unit_00065'
    wfs = rf.file_pointer['processing/tetrode/UnitTimes'][unit]['template'].value
    stds = rf.file_pointer['processing/tetrode/UnitTimes'][unit]['template_std'].value
    x_pos = rf.file_pointer['processing/tetrode/channel_xpos'].value
    y_pos = rf.file_pointer['processing/tetrode/channel_ypos'].value
    rf.close()
    plot_waveforms(wfs, list(zip(x_pos, y_pos)), stds, axes_size=(0.3, 0.3))
    plt.show()
    # =========================================================================================================



    print('for debug ...')