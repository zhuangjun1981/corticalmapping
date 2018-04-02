import os
import numpy as np
import unittest
from corticalmapping.core.TimingAnalysis import TimeIntervals


curr_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_folder)

class TestTimeIntervals(unittest.TestCase):

    def setUp(self):
        pass

    def test_instantialtion(self):
        import numpy as np
        intervals = ([5, 10], (12.5, 24.6), np.array([50., 61.]))

        ti = TimeIntervals(intervals = intervals)

        print('\nintervals:')
        print(ti.get_intervals())

    def test_overlap(self):

        import numpy as np
        intervals1 = ([5, 10], (12.5, 24.6), np.array([50., 61.]))
        intervals2 = [[3., 7.], [13., 26.], [30., 40.], [52.2, 55.5]]

        ti1 = TimeIntervals(intervals=intervals1)
        ti2 = TimeIntervals(intervals=intervals2)

        ti3 = ti1.overlap(time_intervals=ti2)
        tg_arr = np.array([[5., 7.], [13., 24.6], [52.2, 55.5]], dtype=np.float64)
        # print(tg_arr)
        # print(ti3.get_intervals())
        assert (np.array_equal(ti3.get_intervals(), tg_arr))