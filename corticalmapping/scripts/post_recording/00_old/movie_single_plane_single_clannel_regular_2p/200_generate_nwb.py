import os
import corticalmapping.NwbTools as nt

date_recorded = '180404'
mouse_id = '360495'
sess_num = '110'

experimenter = 'Jun'
genotype = 'Vipr2-IRES2-Cre-neo'
sex = 'male'
age = '180'
indicator = 'GCaMP6s'
imaging_rate = 15.24
imaging_depth = '150 microns'
imaging_location = 'visual cortex'
imaging_device = 'Sutter 2p Scope'
imaging_excitation_lambda = '920 nanometers'

curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

notebook_path = os.path.join(curr_folder, 'notebook.txt')
with open(notebook_path, 'r') as ff:
    notes = ff.read()

general = nt.DEFAULT_GENERAL
general['experimenter'] = experimenter
general['subject']['subject_id'] = mouse_id
general['subject']['genotype'] = genotype
general['subject']['sex'] = sex
general['subject']['age'] = age
general['optophysiology'].update({'imaging_plane_1': {}})
general['optophysiology']['imaging_plane_1'].update({'indicator': indicator})
general['optophysiology']['imaging_plane_1'].update({'imaging_rate': imaging_rate})
general['optophysiology']['imaging_plane_1'].update({'imaging_depth': imaging_depth})
general['optophysiology']['imaging_plane_1'].update({'location': imaging_location})
general['optophysiology']['imaging_plane_1'].update({'device': imaging_device})
general['optophysiology']['imaging_plane_1'].update({'excitation_lambda': imaging_excitation_lambda})
general['notes'] = notes

file_name = date_recorded + '_M' + mouse_id + '_' + sess_num + '.nwb'

rf = nt.RecordedFile(os.path.join(curr_folder, file_name), identifier=file_name[:-4], description='')
rf.add_general(general=general)
rf.close()



