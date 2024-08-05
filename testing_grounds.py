from arp import AsteroidRoutingProblem
from arp_vis import plot_solution
import numpy as np
import pykep as pk
from pykep import epoch as def_epoch
from scipy.constants import G
import pandas as pd
import os
from greedy_nn import GreedyNN
from pandas import read_csv, concat
from space_util import START_EPOCH, Earth
from numpy import deg2rad
from pykep.core import DAY2SEC

'''
make r fitness
########################

	def select(self, lamb_sol, v_body1, v_body2, stats=None, *args, **kwargs):
		"""
		Selects one of the Lambert's problem solutions
		(in case multiple revolution solutions were found).
		Selection criterion: solution with the smallest dV.
		"""
		if stats is not None:
			stats.nr_lambert += 1
		
		# get, per solution, the spacecraft's velocity at each body
		v1sc = lamb_sol.get_v1()
		v2sc = lamb_sol.get_v2()
		
		# determine each solution's dV
		solutions = []
		for v1, v2 in zip(v1sc, v2sc):
			dV1 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body1, v1)))
			dV2 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body2, v2)))
			
			if self.dep_ast_id == 0:
				# Earth departure
				# on the first leg, we get up to 5 km/s for free
				dV1 -= 5000.0
			else:
				# If we're not coming from Earth, we must take into account the
				# dV given by the self-flyby leg just performed at the departure
				# asteroid. That maneuver will deliver the projectile and leave
				# the spacecraft with a dV of 400 m/s relative to the asteroid,
				# in any direction we like.
				dV1 -= dV_fb_min
			
			solutions.append((dV1 + dV2, v1, v2))
		
		# pick the solution with smallest dV, and log the spacecraft's
		# velocities at each body
		self.dV, *self.v_sc = min(solutions)
        
        #############################
        
	def select(self, lamb_sol, v_body1, v_body2, *args, **kwargs):
		"""
		Selects one of the Lambert's problem solutions
		(in case multiple revolution solutions were found).
		Selection criterion: solution with the smallest dV.
		"""
		# get, per solution, the spacecraft's velocity at each body
		v1sc = lamb_sol.get_v1()
		v2sc = lamb_sol.get_v2()
		
		# determine each solution's dV
		solutions = []
		for v1, v2 in zip(v1sc, v2sc):
			dV1 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body1, v1)))
			dV2 = sqrt(sum((a - b) * (a - b) for (a, b) in zip(v_body2, v2)))
			solutions.append((dV1 + dV2, v1, v2))
		
		# pick the solution with smallest dV, and log the spacecraft's
		# velocities at each body
		self.dV, *self.v_sc = min(solutions)
        
        ########################################
        
def orbital_indicator(body, t, dT, neg_v=False, **kwargs):
	# reimplemented from:
	# https://github.com/esa/pykep/blob/master/PyKEP/phasing/_knn.py#L46
	(r1, r2, r3), (v1, v2, v3) = body.eph(t)
	r1 /= dT
	r2 /= dT
	r3 /= dT
	if neg_v:
		return (r1 - v1, r2 - v2, r3 - v3,
		        r1, r2, r3)
	else:
		return (r1 + v1, r2 + v2, r3 + v3,
		        r1, r2, r3)
    
def rate__orbital(dep_ast, dep_t, leg_dT, **kwargs):
	"""
	Orbital Phasing Indicator.
	See: http://arxiv.org/pdf/1511.00821.pdf#page=12
	
	Estimates the cost of a `leg_dT` days transfer from departure asteroid
	`dep_ast` at epoch `dep_t`, towards each of the available asteroids.
	"""
	dep_t = pk.epoch(dep_t, 'mjd')
	leg_dT *= DAY2SEC
	orbi = [orbital_indicator(ast, dep_t, leg_dT, **kwargs)
	        for ast in asteroids]
	ast_orbi = np.array(orbi[dep_ast])
	return np.linalg.norm(ast_orbi - orbi, axis=1)


def rate__orbital_2(dep_ast, dep_t, leg_dT, **kwargs):
	"""
	Refinement over the Orbital indicator.
	Attempts to correct bad decisions made by the linear model by checking
	which asteroids are close at arrival, and not only which are close at the
	beginning.
	"""
	r1 = rate__orbital(dep_ast, dep_t, leg_dT)
	r2 = rate__orbital(dep_ast, dep_t + leg_dT, leg_dT, neg_v=True)
#	return np.minimum(r1, r2)
	return np.mean([r1, r2], axis=0)
  '''  
######################################

from space_util import (   
    Asteroids)
from scipy.spatial import distance
from phasing import fixed_knn
from arp import(
    AsteroidRoutingProblem)

n=10
seed = 42
asteroids = Asteroids(n, seed=seed)
get_ast_orbit = lambda x: Earth if x == -1 else asteroids.get_orbit(x)

def get_nearest_neighbor_euclidean(self, from_id, unvisited_ids, current_time, metric='euclidean'):
    epoch = def_epoch(START_EPOCH.mjd2000 + current_time)
    from_r = np.array([self.get_ast_orbit(from_id).eph(epoch)[0]])
    ast_r = np.array([ self.get_ast_orbit(ast_id).eph(epoch)[0] for ast_id in unvisited_ids ])
    ast_dist = distance.cdist(from_r, ast_r, metric)
    return unvisited_ids[np.argmin(ast_dist)]

def nearest_phasing(self, from_id, unvisited_ids, current_time, metric='euclidean'):
    epoch = def_epoch(START_EPOCH.mjd2000 + current_time)
    from_orbit = self.get_ast_orbit(from_id)
    ast_orbits = np.array([ self.get_ast_orbit(ast_id) for ast_id in unvisited_ids ])
    neighbours = fixed_knn(ast_orbits, epoch, metric=metric, T=365.25)
    nearest, near_id, dist = neighbours.find_neighbours(from_orbit, 
                                                        query_type='knn', k=[1])
    return unvisited_ids[near_id[0]]

arp = AsteroidRoutingProblem(n,seed)
current_time = 0
metric = 'euclidean'
from_id = -1
unvisited_ids = np.arange(n)
sequence = [ from_id ]
sol = arp.EmptySolution()
while len(unvisited_ids) > 0:
    to_id = nearest_phasing(arp, from_id = from_id, unvisited_ids = unvisited_ids, current_time = current_time, metric=metric)
    print(to_id)
    sol, t0, t1 = arp.optimize_transfer(from_id, to_id, current_time, sol=sol, t0_bounds = (0., 730.), t1_bounds = (1., 730.), return_times=True)
    unvisited_ids = np.setdiff1d(unvisited_ids, to_id)
    print(unvisited_ids)
    #print(f'Departs from {from_id} at time {current_time + t0} after waiting {t0} days and arrives at {to_id} at time {current_time + t0 + t1} after travelling {t1} days, total cost = {f_total}')
    from_id = to_id
    sequence += [ to_id ]
    print(sequence)
    current_time += t0 + t1
