import numpy as np
import pykep as pk
from pandas import read_csv as pd_read_csv
#from astropy.time import Time
from pykep.core import epoch as def_epoch
##from poliastro.bodies import Body
##from poliastro.twobody import Orbit
from NEW_Orbits import Orbit
##from poliastro.frames import Planes
from astropy import units as u
from astropy.constants import G
##from poliastro.core.angles import M_to_E,E_to_nu
#from space_util import MU, MJD_to_Time
# Useful for numpy vectors
#################################################M_to_E_v = np.vectorize(M_to_E)

#def theta_from_E(E, e):
#    return 2. * np.arctan(np.sqrt((1. + e)/(1. - e)) * np.tan(E / 2.))
    #return E_to_nu(E=E, ecc=e)

# The start time of earth and asteroids orbit.
EARTH_START_EPOCH = def_epoch(59396,"mjd2000"),
#################################################EARTH_START_EPOCH = MJD_to_Time(59396)

def true_anomaly(M, e):
    # M: must be radians
    # http://www.braeunig.us/space/orbmech.htm
    # theta = M + 2 * e * np.sin(M_t) + 1.25 * (e**2) * np.sin(2*M)
    # https://en.wikipedia.org/wiki/True_anomaly#From_the_mean_anomaly
    nu = M + (2 * e - 0.25 * e**3) * np.sin(M) + 1.25 * (e**2) * np.sin(2*M) + (13./12.) * (e**3) * np.sin(3*M)
    # E = np.vectorize(M_to_E_v(M=M, ecc=e)
    # theta = theta_from_E(E, e)
    return nu

class OrbitBase:
    # Create my own Sun from SPICE kernal
    pk.util.load_spice_kernel('pck00011.tpc')
    Sun = pk.planet.spice('SUN', 'SUN', 'J2000', 'NONE', pk.MU_SUN, pk.MU_SUN)
    
    #Body(
    #    parent=None,
    #    k = MU << (u.km ** 3 / u.s ** 2),
    #    name="Sun")

    @classmethod
    def eliptic(a, e, i, raan, w, nu, epoch):
        return Orbit.from_classical(Sun, a << u.AU,
                                    e << u.one,
                                    i << u.rad,
                                    raan << u.rad,
                                    w << u.rad,
                                    M = nu << u.rad,
                                    epoch = epoch, # MJD
                                    # Same as HeliocentricEclipticJ2000
                                    # https://github.com/poliastro/poliastro/blob/main/src/poliastro/frames/util.py
                                    )
 


def to_orbit(ast):
    orbit = pk.planet.keplerian(EARTH_START_EPOCH, (ast.a << u.AU,
                                                   ast.e << u.one,
                                                   ast.i << u.rad,
                                                   ast.raan << u.rad,
                                                   ast.w << u.rad,
                                                   ast.M << u.rad),
                               pk.MU_SUN, G*(ast.mass << u.kg), 1, 1.03, 'Asteroid')
    return orbit

    #return OrbitBase.eliptic(ast.a,
    #                         ast.e,
    #                         ast.i,
    #                         ast.raan,
    #                         ast.w,
    #                         nu=true_anomaly(M=ast.M, e=ast.e),
    #                         epoch=EARTH_START_EPOCH)

# Read asteroids
asteroids = pd_read_csv("Candidate_Asteroids.txt.xz", header=None, sep='\s+',
                        # ID, epoch(MJD), a(semi-eje mayor, AU), e (eccentricity), i (inclination, deg), RAAN(right ascension of ascending node, deg), w(perigeo, deg), M (mean anomaly, deg) and mass(kg)
                        names = ['ID','epoch', 'a', 'e', 'i', 'raan', 'w', 'M', 'mass'], index_col = 'ID')
_ast_epoch = asteroids.epoch.unique()
# All asteroids have the same EPOCH as Earth. The code later relies on this.
assert len(_ast_epoch) == 1
EARTH_START_EPOCH = _ast_epoch[0]
asteroids.drop('epoch',  axis=1, inplace=True)
deg_columns = ['i','raan','w','M'] 
asteroids[deg_columns] = asteroids[deg_columns].transform(np.deg2rad)
# asteroids.to_pickle("asteroids.pkl.gz")
ast_orbits = asteroids.apply(to_orbit, axis=1)
ast_orbits.to_pickle("ast_orbits.pkl.gz")
