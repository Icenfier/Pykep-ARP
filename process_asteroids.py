import numpy as np
import pykep as pk
from pandas import read_csv as pd_read_csv
from pykep.core import epoch as def_epoch
from scipy.constants import G

# The start time of earth and asteroids orbit.
EARTH_START_EPOCH = def_epoch(59396,"mjd2000")

def to_orbit(ast):
    orbit = pk.planet.keplerian(EARTH_START_EPOCH, (ast.a * pk.AU, # m
                                                    ast.e,         # no units
                                                    ast.i,         # rad
                                                    ast.raan,      # rad
                                                    ast.w,         # rad
                                                    ast.M),        # rad
                                pk.MU_SUN, G*ast.mass)
    print(orbit)
    return orbit


# Read asteroids
asteroids = pd_read_csv("Candidate_Asteroids.txt.xz", header=None, sep='\s+',
                        # ID, epoch(MJD), a(semi-eje mayor, AU), e (eccentricity), i (inclination, deg), RAAN(right ascension of ascending node, deg), w(perigeo, deg), M (mean anomaly, deg) and mass(kg)
                        names = ['ID','epoch', 'a', 'e', 'i', 'raan', 'w', 'M', 'mass'], index_col = 'ID')
_ast_epoch = asteroids.epoch.unique()
# All asteroids have the same EPOCH as Earth. The code later relies on this.
assert len(_ast_epoch) == 1
assert EARTH_START_EPOCH.mjd2000 == _ast_epoch[0]
asteroids.drop('epoch',  axis=1, inplace=True)
deg_columns = ['i','raan','w','M'] 
asteroids[deg_columns] = asteroids[deg_columns].transform(np.deg2rad)
ast_orbits = asteroids.apply(to_orbit, axis=1)
ast_orbits.to_pickle("pykep_ast_orbits.pkl.gz")