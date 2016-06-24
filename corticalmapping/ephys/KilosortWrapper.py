import csv
import os
import sys
import numpy as np
import h5py
import corticalmapping.core.FileTools as ft
# # adding the upper level into system path so that we can import cort.FileTools
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import core.FileTools as ft

def read_csv(csv_path):
    """
    read kilosort "cluster_groups.csv" results file
    :param csv_path:
    :return: list of tuples (id, type)
    """

    output = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            output.append(tuple(row[0].split('\t')))

    return output


def get_clusters(csv_output):
    """
    get cluster for each single unit and one cluster for all multi-unit activity
    :param csv_output: output of read_csv function. list of tuples (cluster id, type)
    :return: dictionary {unit_name : cluster_id, 'mua' : list of mua cluster_ids}
    """

    output = {'unit_mua' : []}

    for cluster in csv_output:

        try:
            cluster_id = int(cluster[0])
        except ValueError:
            print cluster[0], 'can not be converted into integer.'
            continue
        cluster_type = cluster[1]

        if cluster_type == 'good':
            output.update({'unit_' + ft.int2str(cluster_id, 5) : cluster_id})
        if cluster_type == 'mua':
            output['unit_mua'].append(cluster_id)

    return output


def get_spike_times_indices(clusters, spike_cluster_path, spike_times_path):
    """
    get spike timestamps of defined clusters
    :param clusters: output of get_clusters function.
                     dictionary {unit_name : cluster_id, 'mua' : list of mua cluster_ids}
    :param spike_cluster_path:
    :param spike_times_path:
    :return:
    """

    spike_times = np.load(spike_times_path).flatten()
    spike_cluster = np.load(spike_cluster_path).flatten()

    if len(spike_cluster) != len(spike_times):
        raise ValueError('length of spike_cluster does not match length of spike_times!')

    spike_ind = {}

    for cluster, cluster_id in clusters.iteritems():
        if cluster == 'unit_mua':
            mua_spike_ind = []
            for id in cluster_id:
                mua_spike_ind += list(spike_times[spike_cluster == id])
            mua_spike_ind.sort()
            spike_ind.update({'unit_mua':mua_spike_ind})
        else:
            curr_cluster_ind = list(spike_times[spike_cluster == cluster_id])
            curr_cluster_ind.sort()
            spike_ind.update({cluster:curr_cluster_ind})

    return spike_ind


def get_spike_timestamps(spike_ind, h5_path):
    """
    get timestamps in seconds of each defined unit in spike_ind for each file
    :param spike_ind:
    :param h5_path:
    :return: update the h5_file to contain timestamps in seconds of each defined unit in spike_ind for each file
    """
    h5_file = h5py.File(h5_path, 'r+')
    folder_list = [f for f in h5_file.keys() if f[0:6] == 'folder']
    fs = h5_file['fs'].value

    print folder_list
    print fs

    for folder in folder_list:
        curr_group = h5_file[folder]
        curr_start_ind = curr_group.attrs['start_index']
        curr_end_ind = curr_group.attrs['end_index']
        print curr_start_ind
        print curr_end_ind

        for unit, spikes in spike_ind.iteritems():

            curr_spike_ind = [spk for spk in spikes if spk >= curr_start_ind and spk < curr_end_ind]
            curr_spike_ind = np.array(curr_spike_ind, dtype=np.float32) - curr_start_ind
            curr_spike_timestamps = curr_spike_ind / fs
            curr_dataset = curr_group.create_dataset(unit, data=curr_spike_timestamps)
            curr_dataset.attrs['unit'] = 'second'


if __name__ == "__main__":

    csv_path = r"G:\160610-M240652\processed\cluster_groups.csv"
    spike_cluster_path = r"G:\160610-M240652\processed\spike_clusters.npy"
    spike_times_path = r"G:\160610-M240652\processed\spike_times.npy"
    h5_path = r"G:\160610-M240652\processed\160610-M240652.hdf5"

    cluster_group = read_csv(csv_path)
    print 'cluster_group:'
    print cluster_group
    clusters = get_clusters(cluster_group)
    print '\nclusters:'
    print clusters
    spike_ind = get_spike_times_indices(clusters, spike_cluster_path, spike_times_path)
    print '\nspike_ind:'
    print spike_ind.keys()
    for key, value in spike_ind.iteritems():
        print key, ':', len(value)
        print key, ':', value[0:10]

    get_spike_timestamps(spike_ind, h5_path)
