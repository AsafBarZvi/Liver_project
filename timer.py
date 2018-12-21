import time
import numpy as np

class Timer:
    def __init__(self,name):
        self._name    = name
        self._tstarts = []
        self._tends   = []

    def __enter__(self):
        if len(self._tstarts) < 3000:
            self._tstarts.append(time.time())
        return self

    def __exit__(self,exc_type, exc_value, traceback):
        if len(self._tends) < 3000:
            self._tends.append(time.time())


    def stats(self):
        if (len(self._tstarts) == 0) or (len(self._tends) == 0):
            return 0 , 0 , 0
        starts = np.array(self._tstarts)
        ends   = np.array(self._tends)
        return np.mean(ends - starts) , np.std(ends - starts) , np.max(ends - starts)


timer_dict = dict(batch_get =Timer('batch_get'),
                  train = Timer('train'),
                  summary = Timer('summary'))

def timerStats():
    for n , t in timer_dict.iteritems():
        print "TIMER {:s} stats: mean {:.6f} , std {:.6f} , max {:.6f}".format(n , *t.stats())
