__author__ = 'junz'

import numpy as np
import corticalmapping.core.ImageAnalysis as ia


def test_getTrace():
    mov = np.arange(64).reshape((4,4,4))
    print mov

    mask1 = np.zeros((4,4)); mask1[2,2]=1; mask1[1,1]=1
    trace1 = ia.get_trace(mov, mask1, maskMode='binary')
    assert(trace1[2] == 39.5)

    mask2 = np.zeros((4,4),dtype=np.float); mask2[:]=np.nan; mask2[2,2]=1; mask2[1,1]=1
    trace2 = ia.get_trace(mov, mask2, maskMode='binaryNan')
    assert(trace2[2] == 39.5)

    mask3 = np.zeros((4,4),dtype=np.float); mask3[2,2]=1; mask3[1,1]=2
    trace3 = ia.get_trace(mov, mask3, maskMode='weighted')
    assert(trace3[2] == 58)

    mask4 = np.zeros((4,4),dtype=np.float); mask4[:]=np.nan; mask4[2,2]=1; mask4[1,1]=2
    trace4 = ia.get_trace(mov, mask4, maskMode='weightedNan')
    assert(trace4[2] == 58)
