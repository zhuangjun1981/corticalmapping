import os
import numpy as np
import corticalmapping.ephys.OpenEphysWrapper as oew
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

    def add_open_ephys_data(self, folder, prefix, digital_channels=('cam_read', 'cam_trigger', 'visual_frame')):

        output = oew.pack_folder_for_nwb(folder=folder, prefix=prefix, digital_channels=digital_channels)

        for key, value in output.items():
            if 'CH' in key:  # analog channel for electrode recording
                ch_ind = int(key[key.find('CH') + 2:])
                ch_name = 'ch_' + ft.int2str(ch_ind, 4)
                ch_trace = value['trace']
                ch_series = self.create_timeseries('ElectricalSeries', ch_name, 'acquisition')
                ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
                                   resolution=1.)
                ch_series.set_time_by_rate(time_zero=value['header']['start_time'],
                                           rate=float(value['header']['sampleRate']))
                ch_series.set_value('electrode_idx', ch_ind)
                ch_series.set_value('num_samples', len(ch_trace))
                ch_series.set_description('extracellular continuous voltage recording from tetrode')
                ch_series.set_source('open ephys')
                ch_series.finalize()

            elif key != 'events':  # other continuous channels
                ch_name = key[len(prefix) + 1:]
                ch_trace = value['trace']
                ch_series = self.create_timeseries('AbstractFeatureSeries', ch_name, 'acquisition')
                ch_series.set_data(ch_trace, unit='bit', conversion=float(value['header']['bitVolts']),
                                   resolution=1.)
                ch_series.set_time_by_rate(time_zero=value['header']['start_time'],
                                           rate=float(value['header']['sampleRate']))
                ch_series.set_value('features', ch_name)
                ch_series.set_value('feature_units', 'bit')
                ch_series.set_value('num_samples', len(ch_trace))
                ch_series.set_value('help', 'continuously recorded analog channels with same sampling times as '
                                            'of electrode recordings')
                ch_series.set_description('continuous voltage recording from IO board')
                ch_series.set_source('open ephys')
                ch_series.finalize()

            else:  # digital events
                # todo: add digital events
                pass





    def add_sync_data(self):
        # not for now
        pass

    def add_kilosort_cluster_results(self):
        pass

    def add_visual_stimulation(self):
        pass


if __name__ == '__main__':

    tmp_path = r"E:\data\python_temp_folder\test.nwb"
    open_ephys_folder = r"E:\data\2016-07-12-160712-M240652-SparseNoise\002_sparse_noise_2016-07-12_09-08-21_Jun"
    rf = RecordedFile(tmp_path, identifier='', description='')
    # rf.add_general()
    rf.add_open_ephys_data(open_ephys_folder, '100')
    rf.close()