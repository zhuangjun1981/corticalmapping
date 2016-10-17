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

    def __init__(self, filename, **kwargs):

        if os.path.isfile(filename):
            keyboard_input = ''
            while keyboard_input != 'y' and keyboard_input != 'n':
                keyboard_input = raw_input('\nthe path "' + filename + '" already exists. Modify it? (y/n) \n')
                if keyboard_input == 'y':
                    super(RecordedFile, self).__init__(filename=filename, modify=True, **kwargs)
                elif keyboard_input == 'n':
                    raise IOError('file already exists.')
        else:
            print('\ncreating a new nwb file: ' + filename)
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

        if ind_start <= ind_end:
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
        mod.add_source('phy-template manual clustering after kilosort')
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



    def add_visual_stimulation(self):
        pass

    def add_segmentation_result(self):
        pass

    def add_roi_traces(self):
        pass

    def add_strf(self):
        pass

    def add_sync_data(self):
        # not for now
        pass

    def add_kilosort_clusters(self, folder):
        """
        expects spike_templates.npy, and spike_times.npy in the folder. use only for the direct outputs of kilosort,
        that haven't been modified with phy-template.
        :param folder:
        :return:
        """
        # clusters = np.load(open(os.path.join(folder, 'spike_clusters.npy')))
        # clusters_data = np.load(open(os.path.join(folder, 'spike_templates.npy')))
        # spikes_data = np.load(open(os.path.join(folder, 'spike_times.npy')))
        # templates = np.load(open(os.path.join(folder, 'templates.npy')))

        # not for now
        pass




if __name__ == '__main__':

    # =========================================================================================================
    # tmp_path = r"E:\data\python_temp_folder\test.nwb"
    # open_ephys_folder = r"E:\data\2016-07-12-160712-M240652-SparseNoise\002_sparse_noise_2016-07-12_09-08-21_Jun"
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
    tmp_path = r"E:\data\python_temp_folder\test.nwb"
    data_path = r"E:\data\2016-07-25-160722-M256896\processed_1"
    rf = RecordedFile(tmp_path)
    rf.add_phy_template_clusters(folder=data_path, module_name='LGN')
    rf.close()
    # =========================================================================================================