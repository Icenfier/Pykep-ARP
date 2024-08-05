from pykep.phasing import knn
from pykep.core import AU, EARTH_VELOCITY

class fixed_knn(knn):
    def __init__(self, planet_list, t, metric='orbital', ref_r=AU, ref_v=EARTH_VELOCITY, T=180.0):
        """
        USAGE: knn = knn(planet_list, t, metric='orbital', ref_r=AU, ref_v=EARTH_VELOCITY, T=365.25):

        - planet_list   list of pykep planets (typically thousands)
        - t             epoch
        - metric        one of ['euclidean', 'orbital']
        - ref_r         reference radius   (used as a scaling factor for r if the metric is 'euclidean')
        - ref_v         reference velocity (used as a scaling factor for v if the metric is 'euclidean')
        - T             average transfer time (used in the definition of the 'orbital' metric)

        Example::

        from pykep import *
        pl_list = [planet.gtoc7(i) for i in range(16257)]
        knn = phasing.knn(pl_list, epoch(t0), metric='orbital', T=180)
        neighb, ids, dists = knn.find_neighbours(pl_list[ast_0], query_type='knn', k=10000)
        neighb, ids, _ = knn.find_neighbours(pl_list[ast_0], query_type='ball', r=5000)
        """
        import numpy as np
        import pykep as pk
        self._asteroids = np.array(planet_list, dtype=object)
        self._ref_r = ref_r
        self._ref_v = ref_v
        self._t = t
        self._metric = metric
        self._T = T
        self._kdtree = self._make_kdtree(self._t)