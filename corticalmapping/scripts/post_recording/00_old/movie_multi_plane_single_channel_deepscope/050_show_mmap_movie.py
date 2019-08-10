import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([r"E:\data\github_packages\CaImAn"])

import os
import numpy as np
import caiman as cm

data_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\Jun\raw_data_rabies_project" \
              r"\180212-M360495-deepscope\2p_movies\04\04_"
plane_n = 'plane0'

plane_folder = os.path.join(data_folder, plane_n, 'corrected')
os.chdir(plane_folder)

fn = [f for f in os.listdir(plane_folder) if f[-5:] == '.mmap']
if len(fn) > 1:
    print('\n'.join(fn))
    raise LookupError('more than one file found.')
elif len(fn) == 0:
    raise LookupError('no file found.')
else:
    fn = fn[0]

cm.load(fn).play(fr=30,magnification=1,gain=2.)

# fn_parts = fn.split('_')
# d1 = int(fn_parts[fn_parts.index('d1') + 1]) # column, x
# d2 = int(fn_parts[fn_parts.index('d2') + 1]) # row, y
# d3 = int(fn_parts[fn_parts.index('d3') + 1]) # channel
# d4 = int(fn_parts[fn_parts.index('frames') + 1]) # frame, T
# order = fn_parts[fn_parts.index('order') + 1]
#
# print('playing {} ...'.format(fn))
#
# mov = np.memmap(filename=fn, shape=(d1, d2, d4), order=order, dtype=np.float32, mode='r')
# mov = mov.transpose((2, 1, 0))
#
# cm.movie(mov).play(fr=30,magnification=1,gain=2.)

