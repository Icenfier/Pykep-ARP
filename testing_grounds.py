import numpy as np
import pykep as pk
from pandas import read_csv as pd_read_csv
from scipy.constants import G
from pykep.core import epoch as def_epoch
from numpy import deg2rad

# The start time of earth and asteroids orbit.
EARTH_START_EPOCH = def_epoch(59396,"mjd2000")
# The mission is limited within 20 years.
# 00:00:00 1st January 2121 (MJD) (first launch)
START_EPOCH = def_epoch(95739,"mjd2000")

class OrbitBuilder:
    def eliptic(a, e, i, raan, w, M, mass, epoch):
        return pk.planet.keplerian(epoch, (a * pk.AU, # AU
                                           e,         # no units
                                           i,         # rad
                                           raan,      # rad
                                           w,         # rad
                                           M),        # rad
                                    pk.MU_SUN, G*mass)


Earth_init = OrbitBuilder.eliptic(
    # Table 1 Earth’s orbital elements in the J2000 heliocentric ecliptic reference frame
    a = 9.998012770769207e-1 # AU
    , e = 1.693309475505424e-2
    , i = deg2rad(3.049485258137714e-3) # deg
    , raan = deg2rad(1.662869706216879e2) # deg
    , w = deg2rad(2.978214889887391e2) # omega deg
    , M = deg2rad(1.757352290983351e2) # deg
    , mass = 5.9722e24 # kg
    , epoch = EARTH_START_EPOCH) # MJD

print(Earth_init.eph(EARTH_START_EPOCH))
print(Earth_init.eph(START_EPOCH))