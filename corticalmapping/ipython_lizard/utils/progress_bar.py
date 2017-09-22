# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:04:27 2015

@author: chrism
"""
import sys
import time
try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class ProgressBar:
    """
    Shamelessly stolen, with slight modification, from: https://gist.github.com/minrk/2211026
    """
    
    def __init__(self, iterations,message=""):



        self.iterations = iterations
        self.message = message
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        # if self.have_ipython:
        self.animate = self.animate_ipython
        self.count = 0
        self.animate(self.count)
        # else:
        #     self.animate = self.animate_noipython

    def update(self,message=None):
        self.count += 1
        self.animate(self.count)
        if message is not None:
            self.message = message

    def animate_ipython(self, iter):
        try:
            clear_output()
        except Exception:
            # terminal IPython has no clear_output
            pass
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = self.message+' [' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) / 2) - len(str(percent_done)) + (len(self.message)+2)/2
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
        

if  __name__ == "__main__":
    pb = ProgressBar(100,message="Test")
    
    for i in range(100):
        pb.update()