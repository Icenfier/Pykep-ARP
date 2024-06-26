import numpy as np
import pykep as pk
from pandas import read_csv as pd_read_csv
from scipy.constants import G
from pykep.core import epoch as def_epoch
from numpy import deg2rad

from space_util import (
    Earth)

START_EPOCH = def_epoch(95739,"mjd2000")

n = 5
size = n
seed = 47
ast_id = 1

from pandas import read_pickle
ast_orbits = read_pickle("ast_orbits.pkl.gz")
print(ast_orbits)
from numpy.random import default_rng
rng = default_rng(seed)
ids = rng.integers(ast_orbits.index.min(), ast_orbits.index.max() + 1, size = size)
ast_orbits = ast_orbits[ids].reset_index(drop=True)
orb = ast_orbits[0]
print(orb)
print(ast_orbits[0].mu_self / G)

#orbit = ast_orbits.loc[ast_id]
#r, v = orbit.eph(START_EPOCH)
#print(orbit.mass)
#planet = pk.planet.keplerian(START_EPOCH, r, v, pk.MU_SUN, G*orbit.mass)
    
