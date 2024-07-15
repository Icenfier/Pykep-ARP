import numpy as np
import pykep as pk
from pykep import DAY2SEC
from pandas import read_csv as pd_read_csv
from scipy.constants import G
from pykep.core import epoch as def_epoch
from numpy import deg2rad
from pykep.planet import _base
from poliastro.maneuver import Maneuver
from arp import AsteroidRoutingProblem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

from space_util import (
    Earth,
    EARTH_START_EPOCH,
    propagate,
    OrbitBuilder,
    two_shot_transfer,
    switch_orbit)

START_EPOCH = def_epoch(95739,"mjd2000")
MAX_REVS = 0

n = 5
size = n
seed = 47
ast_id = 1
'''
arp_instance = AsteroidRoutingProblem(10, 42)

def two_shot_transfer(from_orbit, to_orbit, t0, t1):
    assert t0 >= 0 and t1 > 0,f'It must be true that t0={t0} >= 0 and t1={t1} > 0'
    start_epoch = from_orbit.ref_mjd2000
    from_orbit = propagate(from_orbit, def_epoch(start_epoch + t0))
    #print(f'from_orbit.epoch: {from_orbit.epoch}')
    epoch = def_epoch(start_epoch + t0 + t1)
    to_orbit = propagate(to_orbit, epoch)
    #assert epoch.value < LAST_EPOCH.value
    try:
        man = pk.lambert_problem(from_orbit.eph(start_epoch)[0], to_orbit.eph(epoch)[0], tof = t1, mu = pk.MU_SUN, cw = False, max_revs = MAX_REVS)
        #if len(man.get_v1()) > 1:
         #   for i in range (0, len(man.get_v1())):
                
            
        # man_poli = Maneuver.lambert(from_orbit, to_orbit)
    except Exception as e:
        e.args = (e.args if e.args else tuple())
        print(f'two_shot_transfer failed: {type(e)} {str(e.args)}: {from_orbit.eph(start_epoch)[0]} {start_epoch} {to_orbit.eph(epoch)[0]} {epoch} {t0} {t1}')
        return False
    return man, to_orbit # TODO why returning to_orbit??

class Asteroids:
    def __init__(self, size, seed):
        from pandas import read_pickle
        ast_orbits = read_pickle("ast_orbits.pkl.gz")
        from numpy.random import default_rng
        rng = default_rng(seed)
        ids = rng.integers(ast_orbits.index.min(), ast_orbits.index.max() + 1, size = size)
        self.ast_orbits = ast_orbits[ids].reset_index(drop=True)

    def get_orbit(self, ast_id):
        orbit = self.ast_orbits.loc[ast_id]
        r, v = orbit.eph(START_EPOCH)
        #return orbit.eph(START_EPOCH)
        return pk.planet.keplerian(START_EPOCH, r, v, pk.MU_SUN, orbit.mu_self, 0.1, 0.1) # 0.1 values for 'planet radius', needed to function but have no practical use


orbits = Asteroids(size)
'''

Ast1 = OrbitBuilder.eliptic(
                # Table 1 Earth’s orbital elements in the J2000 heliocentric ecliptic reference frame
                a = 3.23790035, # AU
                e = 0.067977382,
                i = deg2rad(13.97503429), # deg
                raan = deg2rad(178.1380039), # deg
                w = deg2rad(352.4606128), # deg
                M = deg2rad(68.14966237), # deg
                mass = 1.170434380816670E+14, # kg
                epoch = EARTH_START_EPOCH) # epoch

Ast2 = OrbitBuilder.eliptic(
                # Table 1 Earth’s orbital elements in the J2000 heliocentric ecliptic reference frame
                a = 3.172741466, # AU
                e = 0.175766807,
                i = deg2rad(9.593594651), # deg
                raan = deg2rad(217.0053648), # deg
                w = deg2rad(51.37307117), # deg
                M = deg2rad(354.5474664), # deg
                mass = 3.617151418145980E+13, # kg
                epoch = EARTH_START_EPOCH) # epoch

Ast3 = OrbitBuilder.eliptic(
                # Table 1 Earth’s orbital elements in the J2000 heliocentric ecliptic reference frame
                a = 2.729304913, # AU
                e = 0.376888476,
                i = deg2rad(8.45609843), # deg
                raan = deg2rad(33.73699037), # deg
                w = deg2rad(28.3959106), # deg
                M = deg2rad(98.63124884), # deg
                mass = 1.387207112626830E+13, # kg
                epoch = EARTH_START_EPOCH) # epoch


epoch = START_EPOCH
ship = propagate(Earth, epoch)
fig = plt.figure(figsize=((10,8)))
ax = fig.add_subplot(projection='3d')

end_epoch = def_epoch(START_EPOCH.mjd2000+500)
man1 = pk.lambert_problem(Earth.eph(START_EPOCH)[0], Ast1.eph(end_epoch)[0], tof = 100*DAY2SEC, mu = pk.MU_SUN, cw = False, max_revs = 2)
new_epoch = def_epoch(end_epoch.mjd2000+100)
ship = switch_orbit(Ast1, end_epoch)
plot_planet(ship, t0 = end_epoch, tf = new_epoch, axes = ax, legend = (True, 'ast'))
plot_lambert(l=man1, axes = ax)
print(man1.get_x())

start_epoch = new_epoch
end_epoch = def_epoch(start_epoch.mjd2000 + 500)
man2 = pk.lambert_problem(Ast1.eph(start_epoch)[0], Ast2.eph(end_epoch)[0], tof = 100*DAY2SEC, mu = pk.MU_SUN, cw = False, max_revs = 2)
new_epoch = def_epoch(end_epoch.mjd2000+100)
ship = switch_orbit(Ast2, end_epoch)
plot_planet(ship, t0 = end_epoch, tf = new_epoch, axes = ax)
plot_lambert(l=man2, axes = ax)
print(man2.get_x())

start_epoch = new_epoch
end_epoch = def_epoch(start_epoch.mjd2000 + 500)
man3 = pk.lambert_problem(Ast2.eph(start_epoch)[0], Ast3.eph(end_epoch)[0], tof = 100*DAY2SEC, mu = pk.MU_SUN, cw = False, max_revs = 2)
new_epoch = def_epoch(end_epoch.mjd2000+100)
ship = switch_orbit(Ast3, end_epoch)
plot_lambert(l=man3, axes = ax)
print(man3.get_x())

plot_kepler(Earth.eph(START_EPOCH)[0], Earth.eph(START_EPOCH)[1], tof = 366 * DAY2SEC, mu = pk.MU_SUN, axes = ax, color = 'black')

ax.view_init(90, 90) 
plt.draw()