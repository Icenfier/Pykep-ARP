import numpy as np
from numpy import deg2rad
import pykep as pk
from scipy.constants import G
from pykep.core import epoch as def_epoch
from pykep.core import DAY2SEC



# The start time of earth and asteroids orbit.
EARTH_START_EPOCH = def_epoch(59396,"mjd2000")
# The mission is limited within 20 years.
# 00:00:00 1st January 2121 (MJD) (first launch)
START_EPOCH = def_epoch(95739,"mjd2000")
# 00:00:00 1st January 2141 (MJD) (last launch)
LAST_EPOCH = def_epoch(103044,"mjd2000")

MAX_REVS = 10

class Asteroids:
    def __init__(self, size, seed):
        from pandas import read_pickle
        ast_orbits = read_pickle("pykep_ast_orbits.pkl.gz")
        from numpy.random import default_rng
        rng = default_rng(seed)
        ids = rng.integers(ast_orbits.index.min(), ast_orbits.index.max() + 1, size = size)
        self.ast_orbits = ast_orbits[ids].reset_index(drop=True)

    def get_orbit(self, ast_id):
        orbit = self.ast_orbits.loc[ast_id]
        r, v = orbit.eph(START_EPOCH)
        #return orbit.eph(START_EPOCH)
        return pk.planet.keplerian(START_EPOCH, r, v, pk.MU_SUN, orbit.mu_self, 0, 0) # 0.1 values for 'planet radius', needed to function but have no practical use

# Table 2 Constants and unit conversions
# TODO replace with pykep consts, just import in?
#AU = 1.49597870691e8 # km pk.AU
#MU = 1.32712440018e11 # km^3/s^2 pk.MU_SUN
#SEC_PER_DAY = 86400 # s pk.DAY2SEC
#DAYS_PER_YEAR = 365.25 # days 1/pk.DAY2YEAR


class OrbitBuilder:
    
    def eliptic(a, e, i, raan, w, M, mass, epoch):  
        return pk.planet.keplerian(epoch, (a * pk.AU, # AU
                                           e,         # no units
                                           i,         # rad
                                           raan,      # rad
                                           w,         # rad
                                           M),        # rad
                                    pk.MU_SUN, G*mass, 0, 0, 'Earth')


Earth = OrbitBuilder.eliptic(
                # Table 1 Earth’s orbital elements in the J2000 heliocentric ecliptic reference frame
                a = 9.998012770769207e-1, # AU
                e = 1.693309475505424e-2,
                i = deg2rad(3.049485258137714e-3), # deg
                raan = deg2rad(1.662869706216879e2), # deg
                w = deg2rad(2.978214889887391e2), # deg
                M = deg2rad(1.757352290983351e2), # deg
                mass = 5.9722e24, # kg
                epoch = EARTH_START_EPOCH) # epoch



def propagate(body, epoch):
    r, v = body.eph(epoch)
    orbit = pk.planet.keplerian(epoch, r, v, pk.MU_SUN, body.mu_self, 0.1, 0.1) # 0.1 values for required 'planet radius' input, doesnt affect anything
    ##try with propagate lagrangian instead?
    return orbit

def switch_orbit(to_orbit, epoch):
    r, v = to_orbit.eph(epoch)
    body = pk.planet.keplerian(epoch, r, v, pk.MU_SUN, to_orbit.mu_self, 0.1, 0.1) # 0.1 values for required 'planet radius' input, doesnt affect anything
    return body

def two_shot_transfer(from_orbit, to_orbit, t0, t1):
    # ENSURE from_orbit IS AT REFERENCE ORBIT BEFORE CALLING
    assert t0 >= 0 and t1 > 0,f'It must be true that t0={t0} >= 0 and t1={t1} > 0'
    #print('t1', t1)
    start_epoch = from_orbit.ref_mjd2000
    from_epoch = def_epoch(start_epoch + t0)
    from_orbit = propagate(from_orbit, from_epoch)
    #print(f'from_orbit.epoch: {from_orbit.epoch}')
    to_epoch = def_epoch(start_epoch + t0 + t1)
    to_orbit = propagate(to_orbit, to_epoch)
    #assert epoch.value < LAST_EPOCH.value
    try:
        man = pk.lambert_problem(from_orbit.eph(from_epoch)[0], to_orbit.eph(to_epoch)[0], tof = t1*DAY2SEC, mu = pk.MU_SUN, cw = False, max_revs = MAX_REVS)
        #if len(man.get_v1()) > 1:
         #   for i in range (0, len(man.get_v1())):
                
            
        # man_poli = Maneuver.lambert(from_orbit, to_orbit)
    except Exception as e:
        e.args = (e.args if e.args else tuple())
        print(f'two_shot_transfer failed: {type(e)} {str(e.args)}: {from_orbit.eph(from_epoch)[0]} {from_epoch} {to_orbit.eph(to_epoch)[0]} {to_epoch} {t0} {t1}')
        # import dump
        # dump.to_pickle(prefix='two_shot_transfer_proposal', from_orbit=from_orbit, to_orbit=to_orbit, e=e)
        # Return a very bad solution
        # TODO either find way of making bad lambert obj, or do exception wherever function called
        #return (Maneuver((0 * u.s, [1e6, 1e6, 1e6] * u.km / u.s),
        #                 (LAST_EPOCH.value * SEC_PER_DAY * u.s, [1e6, 1e6, 1e6] * u.km / u.s)),
        #        to_orbit)
        return False
    return man, from_orbit, to_orbit

def calc_cost(man, from_orbit, to_orbit):
    tof = man.get_tof()/DAY2SEC
    to_orbit = propagate(to_orbit, def_epoch(from_orbit.ref_epoch.mjd2000 + tof))
    cost_dv1 = np.linalg.norm(np.subtract(man.get_v1()[0], from_orbit.eph(from_orbit.ref_epoch)[1]))
    cost_dv2 = np.linalg.norm(np.subtract(to_orbit.eph(to_orbit.ref_epoch)[1], man.get_v2()[0]))
    cost = cost_dv1 + cost_dv2 # in m
    return cost/1000 # in km
