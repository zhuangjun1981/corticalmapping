import os
import h5py
import numpy as np
import corticalmapping.ephys.KilosortWrapper as kw
import corticalmapping.ephys.OpenEphysWrapper as ow

try:
    from nwb.nwb import NWB
except:
    print 'no Allen Institute NWB API. get this from ' \
          'http://stimash.corp.alleninstitute.org/projects/INF/repos/ainwb/browse'

class OEphysFile(NWB):

    def add_clustering(self):
        pass