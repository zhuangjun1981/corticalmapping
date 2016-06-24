import OpenEphys as oe
import glob
import os
import h5py
import numpy as np
import corticalmapping.core.FileTools as ft

CONTINUOUS_TIMESTAMP_DEYPE = np.dtype('<i8') # dtype timestamp field in each record (block) of .continuous file
CONTINUOUS_SAMPLE_PER_RECORD_DTYPE = np.dtype('<u2') # dtype of samples per record field in each record (block) of .continuous file
CONTINUOUS_RECORDING_NUMBER_DTYPE = np.dtype('<u2') # dtype of recording number field in each record (block) of .continuous file
CONTINUOUS_SAMPLE_DTYPE = np.dtype('>i2') # dtype of each sample in each record (block) of .continuous file
CONTINUOUS_MARKER_BYTES = 10 # number of bytes of marker field in each record (block) of .continuous file

# def load_continuous(file_path, dtype=float, verbose=True):
#     """
#     modified from OpenEphys.loadContinuous function
#
#     :param file_path:
#     :return:
#     """
#
#     assert dtype in (float, np.int16), \
#         'Invalid data type specified for loadContinous, valid types are float and np.int16'
#
#     print "Loading continuous data from " + file_path
#
#     ch = {}
#     recordNumber = np.intp(-1)
#
#     samples = np.zeros(oe.MAX_NUMBER_OF_CONTINUOUS_SAMPLES, dtype)
#     timestamps = np.zeros(oe.MAX_NUMBER_OF_RECORDS)
#     recordingNumbers = np.zeros(oe.MAX_NUMBER_OF_RECORDS)
#     indices = np.arange(0, oe.MAX_NUMBER_OF_RECORDS * oe.SAMPLES_PER_RECORD, oe.SAMPLES_PER_RECORD, np.dtype(np.int64))
#
#     # read in the data
#     f = open(file_path, 'rb')
#
#     header = oe.readHeader(f)
#
#     fileLength = os.fstat(f.fileno()).st_size
#     if verbose: print 'total length of the file: ', fileLength, 'bytes.'
#
#     while f.tell() < fileLength:
#
#         recordNumber += 1
#
#         if verbose: print 'reading record (block): ' , recordNumber
#
#         timestamps[recordNumber] = np.fromfile(f, np.dtype('<i8'), 1)  # little-endian 64-bit signed integer
#         N = np.fromfile(f, np.dtype('<u2'), 1)  # little-endian 16-bit unsigned integer
#
#         if N != oe.SAMPLES_PER_RECORD:
#             raise Exception('Found corrupted record in block ' + str(recordNumber))
#
#         recordingNumbers[recordNumber] = (np.fromfile(f, np.dtype('>u2'), 1))  # big-endian 16-bit unsigned integer
#
#         if dtype == float:  # Convert data to float array and convert bits to voltage.
#
#             # big-endian 16-bit signed integer, multiplied by bitVolts
#             data = np.fromfile(f, np.dtype('>i2'), N) * float(header['bitVolts'])
#
#         else:  # Keep data in signed 16 bit integer format.
#             data = np.fromfile(f, np.dtype('>i2'), N)  # big-endian 16-bit signed integer
#
#         if data.shape[0] != N:
#             print 'samples in block ' + str(recordNumber) + ': ' + str(data.shape[0]) + ', not equal to expected sample' \
#                                                                                         ' size: ' + str(N[0]) + '!!!'
#             print 'there is ' + str(fileLength-f.tell()) + ' byte(s) left in the file.'
#
#         else:
#             samples[indices[recordNumber]:indices[recordNumber + 1]] = data
#
#         marker = f.read(10)  # dump
#
#     ch['header'] = header
#     ch['timestamps'] = timestamps[0:recordNumber]
#     ch['data'] = samples[0:indices[recordNumber]]  # OR use downsample(samples,1), to save space
#     ch['recordingNumber'] = recordingNumbers[0:recordNumber]
#     f.close()
#     return ch

# def pack_folder(folder, channels, prefix, dtype=np.float32):
#     """
#     pack .continuous files in the folder into a numpy array
#
#         [
#          [channel0_datapoint0, channel0_datapoint1, channel0_datapoint2, ..., channel0_datapointN],
#          [channel1_datapoint0, channel1_datapoint1, channel1_datapoint2, ..., channel1_datapointN],
#          [channel2_datapoint0, channel2_datapoint1, channel2_datapoint2, ..., channel2_datapointN],
#          ...
#          [channelM_datapoint0, channelM_datapoint1, channelM_datapoint2, ..., channelM_datapointN]
#          ]
#
#     :param folder:
#     :param channels: list of channels, specify channel order
#     :param prefix:
#     :param dtype
#     :return: 2-d array of numbers with defined data type, shape = (number of channels, number of data points)
#     """
#
#     all_files = os.listdir(folder)
#     file_list = []
#     for channel in channels:
#         file_num = 0
#         for file in all_files:
#             if prefix + '_CH'+str(channel) in file and '.continuous' in file:
#                 file_list.append(os.path.join(folder, file))
#                 file_num += 1
#         if file_num == 0:
#             raise LookupError('Did not find .continuous file for channel:' + str(channel) + '!')
#         elif file_num > 1:
#             raise LookupError('Fond more than one .continuous files for channel:' + str(channel) + '!')
#
#     print '\nLoad .continuous files from source folder: ', folder
#     print '.continuous files tobe loaded:'
#     print '\n'.join(file_list)
#
#     traces = []
#     sample_nums = []
#     fs = None
#
#     for file in file_list:
#         curr_header, curr_trace = load_continuous2(file, dtype=dtype)
#
#         if fs is None:
#             fs = curr_header['sampleRate']
#         else:
#             if fs != curr_header['sampleRate']:
#                 raise ValueError('sampling rate of current file does not match sampling rate of other files in this '
#                                  'folder!')
#
#         traces.append(curr_trace)
#         sample_nums.append(curr_trace.shape[0])
#
#     min_sample_num = min(sample_nums)
#     traces = np.array([trace[0:min_sample_num] for trace in traces]).astype(dtype)
#
#     return traces, min_sample_num, float(fs)

# def pack_folders(folder_list, output_folder, output_filename, channels, prefix, dtype):
#     """
#
#     :param folder_list:
#     :param output_folder:
#     :param output_filename:
#     :param channels:
#     :param prefix:
#     :return:
#     """
#
#     output_path_dat = os.path.join(output_folder, output_filename + '.dat')
#     output_path_h5 = os.path.join(output_folder, output_filename + '.hdf5')
#
#     if os.path.isfile(output_path_dat) or os.path.isfile(output_path_h5):
#         raise IOError('Output path already exists!')
#
#     h5_file = h5py.File(output_path_h5)
#     channels_h5 = h5_file.create_dataset('channel_order', data=channels)
#     channels_h5.attrs['device'] = 'tetrode'
#
#     # data_set = h5_file.create_dataset('data', (10, ), maxshape=(None, ), chunks=(30000 * len(channels), ), dtype=dtype)
#
#     curr_data_start_ind = 0
#     curr_folder_start_ind = 0
#     data_all = []
#     sampling_rate = None
#
#     for i, folder in enumerate(folder_list):
#
#         curr_data_array, curr_sample_num, fs = pack_folder(folder, channels, prefix, dtype)
#         print curr_data_array.shape
#
#         if sampling_rate is None:
#             sampling_rate = fs
#         else:
#             if fs != sampling_rate:
#                 err = 'The sampling rate (' + str(fs) + 'Hz) of folder: (' + folder + ') does not match the sampling' +\
#                     ' rate (' + str(sampling_rate) + ') of other folders.'
#                 raise ValueError(err)
#
#         data_all.append(curr_data_array.flatten(order='F'))
#
#         curr_group = h5_file.create_group('folder'+ft.int2str(i,4))
#         for k, channel in enumerate(channels):
#             curr_group.create_dataset('channel_' + ft.int2str(int(channel), 5), data=curr_data_array[k, :])
#         curr_group.attrs['path'] = folder
#         curr_group.attrs['start_index'] = curr_folder_start_ind
#         curr_group.attrs['end_index'] = curr_folder_start_ind + curr_sample_num
#         curr_folder_start_ind += curr_sample_num
#
#     h5_file.create_dataset('fs_hz', data=float(sampling_rate))
#
#     h5_file.close()
#
#     data_all = np.concatenate(data_all)
#
#     data_all.tofile(output_path_dat)


def load_continuous2(file_path, dtype=np.float32):
    """
    Jun's wrapper to load .continuous data from OpenEphys data files

    :param file_path:
    :param dtype: np.float32 or np.int16
    :return:
    """

    assert dtype in (np.float32, np.int16), \
        'Invalid data type specified for loadContinous, valid types are np.float32 and np.int16'

    print "\nLoading continuous data from " + file_path

    bytes_per_block = CONTINUOUS_TIMESTAMP_DEYPE.itemsize + CONTINUOUS_SAMPLE_PER_RECORD_DTYPE.itemsize + \
                      CONTINUOUS_RECORDING_NUMBER_DTYPE.itemsize + CONTINUOUS_MARKER_BYTES + \
                      CONTINUOUS_SAMPLE_DTYPE.itemsize * oe.SAMPLES_PER_RECORD

    # read in the data
    f = open(file_path, 'rb')

    file_length = os.fstat(f.fileno()).st_size
    print 'total length of the file: ', file_length, 'bytes.'

    print 'bytes per record block: ', bytes_per_block

    block_num = (file_length - oe.NUM_HEADER_BYTES) // bytes_per_block
    print 'total number of valid blocks: ', block_num

    header = oe.readHeader(f)

    samples = np.empty(oe.SAMPLES_PER_RECORD * block_num, dtype)

    for i in range(block_num):
        _ = np.fromfile(f, CONTINUOUS_TIMESTAMP_DEYPE, 1)
        N = np.fromfile(f, CONTINUOUS_SAMPLE_PER_RECORD_DTYPE, 1)[0]

        if N != oe.SAMPLES_PER_RECORD:
            raise Exception('samples per record specified in block ' + str(i) + ' (' + str(N) + \
                            ') does not equal to expected value (' + str(oe.SAMPLES_PER_RECORD)) + ')!'

        _ = (np.fromfile(f, CONTINUOUS_RECORDING_NUMBER_DTYPE, 1))

        if dtype == np.float32:
            curr_samples = (np.fromfile(f, CONTINUOUS_SAMPLE_DTYPE, N) * float(header['bitVolts'])).astype(np.float32)
        elif dtype == np.int16:
            curr_samples = np.fromfile(f, CONTINUOUS_SAMPLE_DTYPE, N).astype(np.int16)
        else:
            raise ValueError('Error in reading data of block:' + str(i))

        samples[i*oe.SAMPLES_PER_RECORD : (i+1)*oe.SAMPLES_PER_RECORD] = curr_samples

        _ = f.read(10)

    return header, samples


def pack_folder2(folder, prefix):
    """
    pack .continuous files in the folder into a dictionary.
    electrode channel will extracted as int16
    other analog channels will be extracted as float32, volts

    :param folder:
    :param prefix:
    :return: dictionary
    """

    all_files = os.listdir(folder)
    all_files = [f for f in all_files if f[0:len(prefix)+1] == prefix+'_' and f[-11:] == '.continuous']
    fs = None
    output = {}
    sample_num = []

    for file in all_files:
        curr_path = os.path.join(folder, file)
        print '\nLoad ' + file + ' from source folder: ', folder

        if file[0:len(prefix) + 3] == prefix + '_CH':
            curr_header, curr_trace = load_continuous2(curr_path, dtype=np.int16)
        else:
            curr_header, curr_trace = load_continuous2(curr_path, dtype=np.float32)

        if fs is None:
            fs = curr_header['sampleRate']
        else:
            if fs != curr_header['sampleRate']:
                raise ValueError('sampling rate of current file does not match sampling rate of other files in this '
                                 'folder!')

        curr_name = file[:-11]
        output.update({curr_name: curr_trace})
        sample_num.append(curr_trace.shape[0])

    min_sample_num = min(sample_num)
    for ch in output.iterkeys():
        output[ch] = output[ch][0:min_sample_num]

    for ch, trace in output.iteritems():
        print ch, ':', trace.shape

    return output, min_sample_num, float(fs)


def pack_folders2(folder_list, output_folder, output_filename, channels, prefix):
    """

    :param folder_list:
    :param output_folder:
    :param output_filename:
    :param channels:
    :param prefix:
    :return:
    """

    output_path_dat = os.path.join(output_folder, output_filename + '.dat')
    output_path_h5 = os.path.join(output_folder, output_filename + '.hdf5')

    if os.path.isfile(output_path_dat) or os.path.isfile(output_path_h5):
        raise IOError('Output path already exists!')

    h5_file = h5py.File(output_path_h5)
    channels_h5 = h5_file.create_dataset('channel_order', data=channels)
    channels_h5.attrs['device'] = 'tetrode'

    curr_folder_start_ind = 0
    data_all = []
    sampling_rate = None

    for i, folder in enumerate(folder_list):

        curr_group = h5_file.create_group('folder' + ft.int2str(i, 4))
        curr_group.attrs['path'] = folder

        curr_trace_dict, curr_sample_num, fs = pack_folder2(folder, prefix)
        all_channels = curr_trace_dict.keys()
        print all_channels

        if sampling_rate is None:
            sampling_rate = fs
        else:
            if fs != sampling_rate:
                err = 'The sampling rate (' + str(fs) + 'Hz) of folder: (' + folder + ') does not match the sampling' +\
                    ' rate (' + str(sampling_rate) + ') of other folders.'
                raise ValueError(err)

        curr_data_array = []
        # add electrode data
        for channel in channels:
            curr_prefix = prefix + '_CH' + str(channel)

            curr_key = [k for k in all_channels if k[:len(curr_prefix)] == curr_prefix]
            if len(curr_key) == 0:
                raise LookupError('no file is found in ' + folder +' for channel ' + str(channel) + '!')
            elif len(curr_key) > 1:
                raise LookupError('more than one files are found in ' + folder +' for channel ' + str(channel) + '!')
            curr_key = curr_key[0]
            curr_dset = curr_group.create_dataset('channel_' + ft.int2str(int(channel), 4),
                                                  data=curr_trace_dict[curr_key])
            curr_dset.attrs['unit'] = 'arbitrary_unit'
            curr_data_array.append(curr_trace_dict[curr_key])
        curr_data_array = np.array(curr_data_array, dtype=np.int16)
        data_all.append(curr_data_array.flatten(order='F'))

        for ch, trace in curr_trace_dict.iteritems():
            if '_CH' not in ch:
                curr_dset = curr_group.create_dataset(ch[len(prefix) + 1:], data=trace)
                curr_dset.attrs['unit'] = 'volt'


        curr_group.attrs['start_index'] = curr_folder_start_ind
        curr_group.attrs['end_index'] = curr_folder_start_ind + curr_sample_num
        curr_folder_start_ind += curr_sample_num

    h5_file.create_dataset('fs_hz', data=float(sampling_rate))

    h5_file.close()

    data_all = np.concatenate(data_all)

    data_all.tofile(output_path_dat)

# todo: add function to load .events file into pack folder function


if __name__ == '__main__':
    # pack_folder(r"G:\160610-M240652\100_spars_noise_open_ephys_03_2016-06-10_16-54-42_Jun", [9, 10, 11, 12], '100')

    # pack_folders([r"G:\160610-M240652\100_spars_noise_open_ephys_03_2016-06-10_16-54-42_Jun",
    #               r"G:\160610-M240652\101_spars_noise_open_ephys_04_2016-06-10_17-22-52_Jun",
    #               r"G:\160610-M240652\102_spars_noise_open_ephys_05_2016-06-10_17-22-52_Jun",
    #               r"G:\160610-M240652\103_spars_noise_open_ephys_06_2016-06-10_18-00-17_Jun",
    #               r"G:\160610-M240652\104_spars_noise_open_ephys_07_2016-06-10_18-06-23_Jun"],
    #               r"G:\160610-M240652\processed",
    #               '160610-M240652', [9, 10, 11, 12], '100', dtype=np.int16)

    # _ = pack_folder2(r"G:\160610-M240652\100_spars_noise_open_ephys_03_2016-06-10_16-54-42_Jun", '100')

    pack_folders2([r"G:\160610-M240652\100_spars_noise_open_ephys_03_2016-06-10_16-54-42_Jun",
                   r"G:\160610-M240652\101_spars_noise_open_ephys_04_2016-06-10_17-22-52_Jun",
                   r"G:\160610-M240652\102_spars_noise_open_ephys_05_2016-06-10_17-22-52_Jun",
                   r"G:\160610-M240652\103_spars_noise_open_ephys_06_2016-06-10_18-00-17_Jun",
                   r"G:\160610-M240652\104_spars_noise_open_ephys_07_2016-06-10_18-06-23_Jun"],
                   r"G:\160610-M240652\processed",
                   '160610-M240652', [9, 10, 11, 12], '100')
