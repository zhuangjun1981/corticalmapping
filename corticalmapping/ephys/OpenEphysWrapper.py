import OpenEphys as oe
import os
import h5py
import numpy as np
import corticalmapping.core.FileTools as ft
import warnings

CONTINUOUS_TIMESTAMP_DTYPE = np.dtype('<i8') # dtype timestamp field in each record (block) of .continuous file
CONTINUOUS_SAMPLE_PER_RECORD_DTYPE = np.dtype('<u2') # dtype of samples per record field in each record (block) of .continuous file
CONTINUOUS_RECORDING_NUMBER_DTYPE = np.dtype('<u2') # dtype of recording number field in each record (block) of .continuous file
CONTINUOUS_SAMPLE_DTYPE = np.dtype('>i2') # dtype of each sample in each record (block) of .continuous file
CONTINUOUS_MARKER_BYTES = 10 # number of bytes of marker field in each record (block) of .continuous file


def find_next_valid_block(input_array, bytes_per_block, start_index):
    """
    this is for finding starting byte index of first valid block after a certain position of a continuous file. Valid
    block is defined as the last 10 bytes equals oe.RECORD_MARKER if read as unsigned integer 8-bit (little endian). This
    is useful when two open ephys recordings are accidentally recorded in a same file. To extract the data of the
    second recording. It is necessary to find the first valid block after the first recording.

    :param input_array:
    :param bytes_per_block: positive integer, number of bytes per block
    :param start_index: non-negative int
    :return: first_block_start: non-negative int, the start index of the first block after start_index
    """

    if len(input_array.shape) != 1:
        raise ValueError('input_array should be 1-d array.')

    first_valid_block_start = None

    for i in range(start_index + bytes_per_block, len(input_array)):

        if np.array_equal(input_array[i-10: i], oe.RECORD_MARKER):
            first_valid_block_start = i - bytes_per_block
            break
    else:
        print('no valid block found after index:', start_index)

    return first_valid_block_start


def get_digital_line_for_plot(h5_group):
    """
    use plt.step to plot, 'where' parameter should be set to be 'post'

    :param h5_group:
    :param fs:
    :return:
    """

    rise_ts = h5_group['rise'].value
    fall_ts = h5_group['fall'].value

    rises = np.ones(rise_ts.shape, dtype=rise_ts.dtype)
    falls = np.zeros(fall_ts.shape, dtype=fall_ts.dtype)

    data = np.array([np.concatenate((rise_ts, fall_ts)), np.concatenate((rises, falls))])
    data = data.transpose()
    data = data[data[:,0].argsort()]

    if data[0, 1] == 1:
        plot_data = np.vstack(([0., 0.], data))
    else:
        plot_data = np.vstack(([0., 1.], data))

    return plot_data


def load_continuous_old(file_path, dtype=np.float32):
    """
    Jun's wrapper to load .continuous data from OpenEphys data files

    :param file_path:
    :param dtype: np.float32 or np.int16
    :return: header: dictionary, standard open ephys header for continuous file
             samples: 1D np.array
    """

    assert dtype in (np.float32, np.int16), \
        'Invalid data type specified for loadContinous, valid types are np.float32 and np.int16'

    print("\nLoading continuous data from " + file_path)

    bytes_per_block = CONTINUOUS_TIMESTAMP_DTYPE.itemsize + CONTINUOUS_SAMPLE_PER_RECORD_DTYPE.itemsize + \
                      CONTINUOUS_RECORDING_NUMBER_DTYPE.itemsize + CONTINUOUS_MARKER_BYTES + \
                      CONTINUOUS_SAMPLE_DTYPE.itemsize * oe.SAMPLES_PER_RECORD

    # read in the data
    f = open(file_path, 'rb')

    file_length = os.fstat(f.fileno()).st_size
    print('total length of the file: ', file_length, 'bytes.')

    print('bytes per record block: ', bytes_per_block)

    block_num = (file_length - oe.NUM_HEADER_BYTES) // bytes_per_block
    print('total number of valid blocks: ', block_num)

    header = oe.readHeader(f)

    samples = np.empty(oe.SAMPLES_PER_RECORD * block_num, dtype)

    is_break = False

    for i in range(block_num):
        if i == 0:
            # to get the timestamp of the very first record (block)
            # for alignment of the digital event
            start_ind = np.fromfile(f, CONTINUOUS_TIMESTAMP_DTYPE, 1)
            start_time = float(start_ind) / float(header['sampleRate'])
        else:
            _ = np.fromfile(f, CONTINUOUS_TIMESTAMP_DTYPE, 1)
        N = np.fromfile(f, CONTINUOUS_SAMPLE_PER_RECORD_DTYPE, 1)[0]

        if N != oe.SAMPLES_PER_RECORD:
            print(('samples per record specified in block ' + str(i) + ' (' + str(N) +
                  ') does not equal to expected value (' + str(oe.SAMPLES_PER_RECORD) + ')!'))
            samples = samples[0 : i * oe.SAMPLES_PER_RECORD]
            is_break = True
            break
            # raise Exception('samples per record specified in block ' + str(i) + ' (' + str(N) + \
            #                 ') does not equal to expected value (' + str(oe.SAMPLES_PER_RECORD) + ')!')

        _ = (np.fromfile(f, CONTINUOUS_RECORDING_NUMBER_DTYPE, 1))

        if dtype == np.float32:
            curr_samples = (np.fromfile(f, CONTINUOUS_SAMPLE_DTYPE, N) * float(header['bitVolts'])).astype(np.float32)
        elif dtype == np.int16:
            curr_samples = np.fromfile(f, CONTINUOUS_SAMPLE_DTYPE, N).astype(np.int16)
        else:
            raise ValueError('Error in reading data of block:' + str(i))

        samples[i*oe.SAMPLES_PER_RECORD : (i+1)*oe.SAMPLES_PER_RECORD] = curr_samples

        record_mark = np.fromfile(f, dtype=np.dtype('<u1'), count=10)
        if not np.array_equal(record_mark, oe.RECORD_MARKER):
            print(('record marker specified in block ' + str(i) + ' (' + str(record_mark) +
                  ') does not equal to expected array (' + str(oe.RECORD_MARKER) + ') !'))
            is_break = True
            break

    if is_break:
        samples = samples[0: oe.SAMPLES_PER_RECORD * (i -1)]

    header.update({'start_time': start_time})

    return header, samples


def load_continuous(file_path, dtype=np.float32, start_ind=0):
    """
    Jun's wrapper to load .continuous data from OpenEphys data files, this will try to load the whole file into memory,
     using np.fromfile. It can also start from any position in the file (defined by the start_ind)

    :param file_path:
    :param dtype: np.float32 or np.int16
    :param start_ind: non-negative int, default 0. start index to extract data.
    :return: header: dictionary, standard open ephys header for continuous file
             samples: 1D np.array
    """

    assert dtype in (np.float32, np.int16), \
        'Invalid data type specified for loadContinous, valid types are np.float32 and np.int16'

    print("\nLoading continuous data from " + file_path)

    bytes_per_block = CONTINUOUS_TIMESTAMP_DTYPE.itemsize + CONTINUOUS_SAMPLE_PER_RECORD_DTYPE.itemsize + \
                      CONTINUOUS_RECORDING_NUMBER_DTYPE.itemsize + CONTINUOUS_MARKER_BYTES + \
                      CONTINUOUS_SAMPLE_DTYPE.itemsize * oe.SAMPLES_PER_RECORD

    input_array = np.fromfile(file_path, dtype='<u1')
    file_length = input_array.shape[0]
    print('total length of the file: ', file_length, 'bytes.')

    valid_block_start = find_next_valid_block(input_array, bytes_per_block=bytes_per_block, start_index=start_ind)
    print('the beginning index of the first valid block after index: ' + str(start_ind) + ' is ' + \
          str(valid_block_start))

    print('bytes per record block: ', bytes_per_block)

    # read in the data
    f = open(file_path, 'rb')

    header = oe.readHeader(f)

    block_num = (file_length - valid_block_start) // bytes_per_block
    print('number of potential valid blocks after index', start_ind, ':', block_num)

    samples = np.empty(oe.SAMPLES_PER_RECORD * block_num, dtype=dtype)

    _ = np.fromfile(f, np.dtype('<u1'), valid_block_start - oe.NUM_HEADER_BYTES)

    for i in range(block_num):
        if i == 0:
            # to get the timestamp of the very first record (block)
            # for alignment of the digital event
            start_ind = np.fromfile(f, CONTINUOUS_TIMESTAMP_DTYPE, 1)
            start_time = float(start_ind) / float(header['sampleRate'])
        else:
            _ = np.fromfile(f, CONTINUOUS_TIMESTAMP_DTYPE, 1)
        N = np.fromfile(f, CONTINUOUS_SAMPLE_PER_RECORD_DTYPE, 1)[0]

        if N != oe.SAMPLES_PER_RECORD:
            print(('samples per record specified in block ' + str(i) + ' (' + str(N) +
                  ') does not equal to expected value (' + str(oe.SAMPLES_PER_RECORD) + ')!'))
            samples = samples[0 : i * oe.SAMPLES_PER_RECORD]
            print('samples per record specified in block ' + str(i) + ' (' + str(N) + \
                  ') does not equal to expected value (' + str(oe.SAMPLES_PER_RECORD) + ')!')
            break

        _ = (np.fromfile(f, CONTINUOUS_RECORDING_NUMBER_DTYPE, 1))

        if dtype == np.float32:
            curr_samples = (np.fromfile(f, CONTINUOUS_SAMPLE_DTYPE, N) * float(header['bitVolts'])).astype(np.float32)
        elif dtype == np.int16:
            curr_samples = np.fromfile(f, CONTINUOUS_SAMPLE_DTYPE, N).astype(np.int16)
        else:
            raise ValueError('Error in reading data of block:' + str(i))

        samples[i*oe.SAMPLES_PER_RECORD : (i+1)*oe.SAMPLES_PER_RECORD] = curr_samples

        record_mark = np.fromfile(f, dtype=np.dtype('<u1'), count=10)
        if not np.array_equal(record_mark, oe.RECORD_MARKER):
            print(('record marker specified in block ' + str(i) + ' (' + str(record_mark) +
                  ') does not equal to expected array (oe.RECORD_MARKER)!'))
            break

    header.update({'start_time': start_time})
    print('continuous channel start time (for aligning digital events): ', start_time)

    return header, samples


def load_events(file_path, channels=None):
    """
    return time stamps in seconds of each digital channel

    :param file_path:
    :param channels: name of channels
    :return: dictionary, {channel: {'rise':[timestamps of rising events in seconds],
                                    'fall':[timestamps of falling events in seconds}}
    """

    print('\n')

    if file_path[-7:] != '.events':
        raise LookupError('The input file: ' + file_path + ' is not a .events file!')

    with open(file_path) as f:
        header = oe.readHeader(f)
    fs = float(header['sampleRate'])

    events = oe.loadEvents(file_path)

    detected_channel_number = int(max(events['channel']) + 1)
    real_channels = ['ch_' + ft.int2str(c, 3) for c in range(detected_channel_number)]

    if channels is not None:
        if detected_channel_number != len(channels):
            warning_msg = '\nThe number of digital channels detected: ' + str(detected_channel_number) + \
                          ' does not match input channel number: ' + str(len(channels))
            warnings.warn(warning_msg)

        if len(channels) <= detected_channel_number:
            real_channels[0:len(channels)] = channels
        else:
            real_channels = channels[0:detected_channel_number]

    output = {}

    for i, ch in enumerate(real_channels):
        output.update({ch : {'rise' : [],
                             'fall' : []}
                       })

        rise = events['timestamps'][np.logical_and(events['channel'] == i, events['eventId'] == 1)]
        output[ch]['rise'] = np.array(rise.astype(np.float32) / fs).astype(np.float32)

        fall = events['timestamps'][np.logical_and(events['channel'] == i, events['eventId'] == 0)]
        output[ch]['fall'] = np.array(fall.astype(np.float32) / fs).astype(np.float32)

    print('events loaded.\n')

    return output


def pack_folder(folder, prefix, digital_channels=('cam_read', 'cam_trigger', 'visual_frame')):
    """
    pack .continuous and .events files in the folder into a dictionary.
    electrode channel will extracted as int16
    other analog channels will be extracted as float32, volts

    the universal start time will be subtracted from all channels. so the time stamps of continuous channels should
    all start from 0.0 second

    :param folder:
    :param prefix:
    :digital_channels:
    :return: dictionary
    """

    all_files = os.listdir(folder)
    continuous_files = [f for f in all_files if f[0:len(prefix)+1] == prefix+'_' and f[-11:] == '.continuous']
    events_files = [f for f in all_files if f[-7:] == '.events' and 'all_channels' in f ]
    fs = None
    start_time = None
    output = {}
    sample_num = []

    if len(events_files) != 1:
        raise LookupError('there should be one and only one .events file in folder: ' + folder)

    for file in continuous_files:
        curr_path = os.path.join(folder, file)
        print('\nLoad ' + file + ' from source folder: ', folder)

        if file[0:len(prefix) + 3] == prefix + '_CH':
            curr_header, curr_trace = load_continuous(curr_path, dtype=np.int16)
        else:
            curr_header, curr_trace = load_continuous(curr_path, dtype=np.float32)

        # check fs for each continuous channel
        if fs is None:
            fs = curr_header['sampleRate']
        else:
            if fs != curr_header['sampleRate']:
                raise ValueError('sampling rate of current file does not match sampling rate of other files in this '
                                 'folder!')

        # check start time for each continuous channel
        if start_time is None:
            start_time = curr_header['start_time']
        else:
            if start_time != curr_header['start_time']:
                raise ValueError('start time of current file does not match start time of other files in this '
                                 'folder!')

        curr_name = file[:-11]
        output.update({curr_name: curr_trace})
        sample_num.append(curr_trace.shape[0])

    min_sample_num = min(sample_num)
    for ch in output.keys():
        output[ch] = output[ch][0:min_sample_num]
    # for ch, trace in output.iteritems():
    #     print ch, ':', trace.shape

    events = load_events(os.path.join(folder, events_files[0]), channels=digital_channels)
    for ch, event in events.items():
        event['rise'] = event['rise'] - start_time
        event['fall'] = event['fall'] - start_time
    output.update({'events': events})

    return output, min_sample_num, float(fs)


def pack_folder_for_nwb(folder, prefix, digital_channels=None):
    """
    pack .continuous and .events files in the folder into a dictionary.
    continuous channel will extracted as int16
    bitVolts of each channel will be returned

    the universal start time will be subtracted from all channels. so the time stamps of continuous channels should
    all start from 0.0 second

    :param folder:
    :param prefix:
    :digital_channels:
    :return:
    """

    all_files = os.listdir(folder)
    continuous_files = [f for f in all_files if f[0:len(prefix)+1] == prefix+'_' and f[-11:] == '.continuous']
    events_files = [f for f in all_files if f[-7:] == '.events' and 'all_channels' in f ]
    fs = None
    start_time = None
    output = {}
    sample_num = []

    if len(events_files) != 1:
        raise LookupError('there should be one and only one .events file in folder: ' + folder)

    for file in continuous_files:
        curr_path = os.path.join(folder, file)
        print('\nLoad ' + file + ' from source folder: ', folder)

        curr_header, curr_trace = load_continuous(curr_path, dtype=np.int16)
        # curr_header, curr_trace = load_continuous_hack(curr_path, dtype=np.int16)

        # check fs for each continuous channel
        if fs is None:
            fs = curr_header['sampleRate']
        else:
            if fs != curr_header['sampleRate']:
                raise ValueError('sampling rate of current file does not match sampling rate of other files in this '
                                 'folder!')

        # check start time for each continuous channel
        if start_time is None:
            start_time = curr_header['start_time']
        else:
            if start_time != curr_header['start_time']:
                raise ValueError('start time of current file does not match start time of other files in this '
                                 'folder!')

        curr_name = file[:-11]
        output.update({curr_name: {'header': curr_header, 'trace': curr_trace}})
        sample_num.append(curr_trace.shape[0])

    min_sample_num = min(sample_num)
    for ch in output.keys():
        output[ch]['trace'] = output[ch]['trace'][0:min_sample_num]

    events = load_events(os.path.join(folder, events_files[0]), channels=digital_channels)
    try:
        sample_rate = float(curr_header['sampleRate'])
    except Exception:
        sample_rate = 30000.
    end_time = min_sample_num / sample_rate
    for ch, event in events.items():
        er = event['rise']
        er = er - start_time
        er = er[(er > 0) & (er <= end_time)]
        event['rise'] = er

        ef = event['fall']
        ef = ef - start_time
        ef = ef[(ef > 0) & (ef <= end_time)]
        event['fall'] = ef
    output.update({'events': events})

    return output


def pack_folders(folder_list, output_folder, output_filename, continous_channels, prefix, digital_channels):
    """

    :param folder_list:
    :param output_folder:
    :param output_filename:
    :param continous_channels:
    :param digital_channels:
    :param prefix:
    :return:
    """

    output_path_dat = os.path.join(output_folder, output_filename + '.dat')
    output_path_h5 = os.path.join(output_folder, output_filename + '.hdf5')

    if os.path.isfile(output_path_dat) or os.path.isfile(output_path_h5):
        raise IOError('Output path already exists!')

    h5_file = h5py.File(output_path_h5)
    h5_file.attrs['device'] = 'tetrode'
    _ = h5_file.create_dataset('channels', data=continous_channels)

    curr_folder_start_ind = 0
    data_all = []
    sampling_rate = None

    for i, folder in enumerate(folder_list):

        curr_group = h5_file.create_group('folder' + ft.int2str(i, 4))
        curr_group.attrs['path'] = folder
        curr_con_group = curr_group.create_group('continuous')
        curr_dig_group = curr_group.create_group('digital')
        curr_ts_group = curr_group.create_group('timestamps')

        curr_trace_dict, curr_sample_num, fs = pack_folder(folder, prefix, digital_channels=digital_channels)
        all_channels = list(curr_trace_dict.keys())
        print('\nall channels in folder ', folder, ':')
        print(all_channels)
        print()

        if sampling_rate is None:
            sampling_rate = fs
        else:
            if fs != sampling_rate:
                err = 'The sampling rate (' + str(fs) + 'Hz) of folder: (' + folder + ') does not match the sampling' +\
                    ' rate (' + str(sampling_rate) + ') of other folders.'
                raise ValueError(err)

        curr_data_array = []

        # add electrode channels
        for channel in continous_channels:
            curr_prefix = prefix + '_CH' + str(channel)

            curr_key = [k for k in all_channels if k[:len(curr_prefix)] == curr_prefix]
            if len(curr_key) == 0:
                raise LookupError('no file is found in ' + folder +' for channel ' + str(channel) + '!')
            elif len(curr_key) > 1:
                raise LookupError('more than one files are found in ' + folder +' for channel ' + str(channel) + '!')
            curr_key = curr_key[0]
            curr_dset = curr_con_group.create_dataset('channel_' + ft.int2str(int(channel), 4),
                                                      data=curr_trace_dict[curr_key])
            curr_dset.attrs['unit'] = 'arbitrary_unit'
            curr_data_array.append(curr_trace_dict[curr_key])
        curr_data_array = np.array(curr_data_array, dtype=np.int16)
        data_all.append(curr_data_array.flatten(order='F'))

        # add continuous channels
        for ch, trace in curr_trace_dict.items():
            if '_CH' not in ch and ch != 'events':
                curr_dset = curr_con_group.create_dataset(ch[len(prefix) + 1:], data=trace)
                curr_dset.attrs['unit'] = 'volt'

        # add digital events
        events = curr_trace_dict['events']
        for dch, dch_dict in events.items():
            curr_dch_group = curr_dig_group.create_group(dch)
            curr_dch_group.create_dataset('rise', data=dch_dict['rise'])
            curr_dch_group.create_dataset('fall', data=dch_dict['fall'])

        curr_group.attrs['start_index'] = curr_folder_start_ind
        curr_group.attrs['end_index'] = curr_folder_start_ind + curr_sample_num
        curr_folder_start_ind += curr_sample_num

    h5_file.create_dataset('fs_hz', data=float(sampling_rate))

    h5_file.close()

    data_all = np.concatenate(data_all)

    data_all.tofile(output_path_dat)


if __name__ == '__main__':

    pack_folders([r"G:\160610-M240652\100_spars_noise_open_ephys_03_2016-06-10_16-54-42_Jun",
                  r"G:\160610-M240652\101_spars_noise_open_ephys_04_2016-06-10_17-22-52_Jun",
                  r"G:\160610-M240652\102_spars_noise_open_ephys_05_2016-06-10_17-22-52_Jun",
                  r"G:\160610-M240652\103_spars_noise_open_ephys_06_2016-06-10_18-00-17_Jun",
                  r"G:\160610-M240652\104_spars_noise_open_ephys_07_2016-06-10_18-06-23_Jun"],
                  r"G:\160610-M240652\processed_1",
                  '160610-M240652', [9, 10, 11, 12], '100', ('cam_read', 'cam_trigger', 'visual_frame'))

    # ff = h5py.File(r"G:\160610-M240652\processed\160610-M240652.hdf5")
    # vsync = ff['folder0004']['visual_frame']
    # get_digital_line_for_plot(vsync)

