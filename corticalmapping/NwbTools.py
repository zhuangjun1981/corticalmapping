import os
import numpy as np
import corticalmapping.ephys.OpenEphysWrapper as oew
import corticalmapping.ephys.KilosortWrapper as kw
import corticalmapping.core.FileTools as ft
try:
    from nwb.nwb import NWB
except ImportError:
    print 'no Allen Institute NWB API. get this from ' \
          'http://stimash.corp.alleninstitute.org/projects/INF/repos/ainwb/browse'

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
                                      'indicator': '',
                                      # 'excitation_lambda': '',
                                      'imaging_rate': '',
                                      # 'location': '',
                                      # 'device': '',
                                      },
                   # 'optogenetics': {},
                   'devices': {}
                   }


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
                    keyboard_input = raw_input('\nthe path "' + filename + '" already exists. Modify it? (y/n) \n')
                    if keyboard_input == 'y':
                        super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
                    elif keyboard_input == 'n':
                        raise IOError('file already exists.')
            else:
                print('\nModifying existing nwb file: ' + filename)
                super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
        else:
            print('\nCreating a new nwb file: ' + filename)
            super(RecordedFile, self).__init__(filename=filename, modify=False, **kwargs)

    def add_general(self, general=DEFAULT_GENERAL, is_overwrite=True):
        """
        add general dictionary to the general filed
        """
        slf = self.file_pointer

        for key, value in general.items():
            if isinstance(value, dict):
                try:
                    curr_group = slf['general'].create_group(key)
                except ValueError:
                    curr_group = slf['general'][key]
                for key2, value2 in value.items():
                    ft.update_key(curr_group, key2, value2, is_overwrite=is_overwrite)
            else:
                ft.update_key(slf['general'], key, value, is_overwrite=is_overwrite)

    def add_open_ephys_data(self, folder, prefix, digital_channels=()):
        """
        add open ephys raw data to self, in acquisition group
        :param folder: str, the folder contains open ephys raw data
        :param prefix: str, prefix of open ephys files
        :param digital_channels: list of str, digital channel
        :return:
        """
        output = oew.pack_folder_for_nwb(folder=folder, prefix=prefix, digital_channels=digital_channels)

        for key, value in output.items():

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

                for key2, value2 in value.items():

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

    def add_phy_template_clusters(self, folder, module_name, ind_start=None, ind_end=None):
        """
        extract phy-template clustering results to nwb format. Only extract spike times, no template for now.
        Usually the continuous channels of multiple files are concatenated for kilosort. ind_start and ind_end are
        Used to extract the data of this particular file.

        :param folder: folder containing phy template results.
                       expects cluster_groups.csv, spike_clusters.npy and spike_times.npy in the folder.
        :param module_name: str, name of clustering module group
        :param ind_start: int, the start index of continuous channel of the current file in the concatenated file.
        :param ind_end: int, the end index of continuous channel of the current file in the concatenated file.
        :return:
        """

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

        clusters_path = os.path.join(folder, 'spike_clusters.npy')
        spike_times_path = os.path.join(folder, 'spike_times.npy')
        phy_template_output = kw.get_clusters(kw.read_csv(os.path.join(folder, 'cluster_groups.csv')))

        spike_ind = kw.get_spike_times_indices(phy_template_output, spike_clusters_path=clusters_path,
                                               spike_times_path=spike_times_path)

        mod = self.create_module(name=module_name)
        mod.set_description('phy-template manual clustering after kilosort')
        unit_times = mod.create_interface('UnitTimes')
        for unit in spike_ind.keys():
            curr_ts = np.array(spike_ind[unit])
            curr_ts = curr_ts[np.logical_and(curr_ts >= ind_start, curr_ts < ind_end)] - ind_start
            curr_ts = curr_ts / fs
            unit_times.add_unit(unit_name=unit, unit_times=curr_ts,
                                source='electrophysiology extracellular recording',
                                description="Data spike-sorted by: " + self.file_pointer['general/experimenter'].value +
                                            ' using phy-template. Spike time unit: seconds.')

        unit_times.finalize()
        mod.finalize()

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
            print ('\nWarning: {}'.format(log_path))
            print('Unequal number of displayFrames ({}) and timeStamps ({}).'.format(len(display_frames),
                                                                                     len(time_stamps)))

        if stim_name == 'SparseNoise':
            self._add_sparse_noise_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'FlashingCircle':
            self._add_flashing_circle_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'UniformContrast':
            self._add_uniform_contrast_stimulation(log_dict, display_order=display_order)
        elif stim_name == 'DriftingGratingCircle':
            self._add_drifting_grating_circle_stimulation(log_dict, display_order=display_order)
            pass
        else:
            raise ValueError('stimulation name {} unrecognizable!'.format(stim_name))

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
        stim.set_value('center_location_deg', log_dict['stimulation']['center'])
        stim.set_value('center_location_format', '[azimuth, altitude]')
        stim.set_value('color', log_dict['stimulation']['color'])
        stim.set_value('background_color', log_dict['stimulation']['background'])
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
        stim.set_source('corticalmapping.VisualStim.DriftingGratingCircle for stimulus; '
                        'corticalmapping.VisualStim.DisplaySequence for display')
        stim.set_value('background_color', log_dict['stimulation']['background'])
        stim.finalize()

    def _check_display_order(self, display_order=None):
        """
        check display order make sure each presentation has a unique position, and move from increment order.
        also check the given display_order is of the next number
        """
        stimuli = self.file_pointer['stimulus/presentation'].keys()

        print('\nExisting visual stimuli:')
        print('\n'.join(stimuli))

        stimuli = [int(s[0:s.find('_')]) for s in stimuli]
        stimuli.sort()
        if stimuli != range(len(stimuli)):
            raise ValueError('display order is not incremental.')

        if display_order is not None:

            if display_order != len(stimuli):
                raise ValueError('input display order not the next display.')

    def add_visual_stimulations(self, log_paths):

        exist_stimuli = self.file_pointer['stimulus/presentation'].keys()

        for i, log_path in enumerate(log_paths):
            self.add_visual_stimulation(log_path, i + len(exist_stimuli))

    def analyze_photodiode(self):
        # todo: finish this method
        pass




    def add_image_series(self, name, image_matrix, image_path, time_stamps):
        # not for now
        pass

    def add_segmentation_result(self):
        # todo: finish this method
        pass

    def add_roi_traces(self):
        # todo: finish this method
        pass

    def add_strf(self):
        # todo: finish this method
        pass

    def add_motion_correction(self):
        # not for now
        pass

    def add_sync_data(self):
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

    print('for debug ...')