import numpy as np
import pykep as pk
from pykep.core import epoch as def_epoch
from scipy.constants import G
from phasing import fixed_knn
from space_util import (   
    Asteroids,             
    two_shot_transfer,    
    START_EPOCH,          
    Earth,                 
    propagate,
    calc_cost
)

from scipy.optimize import minimize, Bounds
from scipy.spatial import distance

def assert_bounds(x, bounds):
    bounds = np.asarray(bounds)
    assert (x >= bounds[:,0]).all(), f'{x} >= {bounds[:,0]}'
    assert (x <= bounds[:,1]).all(), f'{x} <= {bounds[:,1]}'

def get_default_opts(method, tol = 1e-6, adaptive = True, eps = 1.4901161193847656e-08,
                     rhobeg = 1.0, maxls = 20, maxcor = 10, jac = "2-point", maxiter = 1000):
    options = { 'Nelder-Mead' : dict(tol = tol, options = dict(fatol=0.0001, adaptive = adaptive)),
                'COBYLA' : dict(tol = tol, options=dict(rhobeg = rhobeg)),
                'L-BFGS-B' : dict(tol = tol, jac = jac, options = dict(eps = eps, maxls = maxls, maxcor = maxcor)),
                'SLSQP' : dict(tol = tol, jac = jac, options = dict(maxiter = maxiter, eps = eps)), }
    return options[method]

class VisitProblem:
    VISIT_BOUNDS = (0., 730.) # (0 days, 2 years)
    TRANSFER_BOUNDS = (1., 730.) # (1 day, 2 years)
    
    COST_TIME_TRADEOFF = 2 / 30 # 2 km/s ~ 30 days

    bounds = [VISIT_BOUNDS, TRANSFER_BOUNDS]
    x0 = np.array([0., 30.]) # FIXME: perhaps it should be [0., 30.] to match the optimize_* functions below.
    assert_bounds(x0, bounds) 
    print_best = False
    print_all = print_best and False
    
    def __init__(self, from_orbit, to_orbit):
        self.best_leg_time = np.empty(len(self.x0))
        self.best_f = np.inf
        self.best_man = None
        self.lower = np.array(self.bounds)[:,0]
        self.upper = np.array(self.bounds)[:,1]
        
        self.from_orbit = from_orbit
        self.to_orbit = to_orbit

    def __call__(self, leg_time):
        man, from_orbit, to_orbit = two_shot_transfer(self.from_orbit, self.to_orbit, 
                                                      t0=leg_time[0], t1=leg_time[1])
        cost = calc_cost(man, from_orbit, to_orbit)
        time = leg_time.sum()
        f = self.update_best(leg_time, cost, time, man)
        return f

    @classmethod
    def f(self, cost, time):
        return cost + self.COST_TIME_TRADEOFF * time 

    def update_best(self, leg_time, cost, time, man):
        f = self.f(cost, time)
        if f < self.best_f:
            self.best_leg_time[:] = leg_time[:]
            self.best_f = f
            self.best_man = man
            if self.print_best:
                print(f'New best:{f}:{cost}:{time}:{leg_time}')
        elif self.print_all:
            print(f'{f}:{cost}:{time}:{leg_time}')
        return f

def inner_minimize_multistart(fun, full_fun, multi, bounds, x0 = False, 
                              method = 'SLSQP', constraints = (), **kwargs):
    # fun is part of function to minimise, full_fun is full function to allow return of different values
    
    # splitting into smaler chunks to find minima
    options = get_default_opts(method, **kwargs)
    best_f = np.inf
    best_t0 = None
    best_t1 = None
    deltas = [ .0, .25, .5, .75, 1.0, .125, .375, .625, .875]
    for d in deltas[:multi]: 
        if not x0:
            x0 = (bounds[0][0] + d * (bounds[0][1] - bounds[0][0]), min(30, bounds[1][1]))
        res = minimize(fun, x0 = x0, bounds = bounds, method = method, 
                       constraints = constraints, **options)
        if res.fun < best_f:
            best_f, best_t0, best_t1 = res.fun, res.x[0], res.x[1] 
            _, best_man = full_fun(best_t0, best_t1)
    return (best_f, best_t0, best_t1), best_man


def optimize_problem(problem, method = 'SLSQP', **kwargs):
    options = get_default_opts(method, **kwargs)
    result = minimize(problem, x0 = problem.x0, bounds = problem.bounds,
                      method=method, **options)
    return result

class Spaceship:

    def __init__(self, asteroids):
        self.get_ast_orbit = asteroids.get_orbit
        self.ast_list = []
        self.maneuvers = []
        self.costs = []
        self.orbit = propagate(Earth, START_EPOCH)
        self.leg_times = np.array([])
        self.f = []

    def add_ast(self, ast_id, leg_time, f_val, maneuver, cost):
        self.ast_list.append(ast_id)
        self.orbit = self.get_ast_orbit(ast_id)
        self.leg_times = np.append(self.leg_times, leg_time)
        self.f = np.append(self.f, f_val)
        self.maneuvers.append(maneuver)
        self.costs.append(cost)
        
    def return_all(self):
        return self.ast_list, self.leg_times, self.f, self.maneuvers, self.costs

class AsteroidRoutingProblem():
    # Class attributes
    problem_name = "ARP"

    @classmethod
    def read_instance(cls, instance_name): 
        *_, n, seed = instance_name.split("_")
        return cls(int(n), int(seed))
    
    def __init__(self, n, seed):
        self.asteroids = Asteroids(n, seed=seed)
        self.get_ast_orbit = lambda x: Earth if x == -1 else self.asteroids.get_orbit(x)
        self.n = n
        self.seed = seed
        self.instance_name = str(n) + "_" + str(seed)
        
        self.best_sol = None
        self.worst_sol = None
        self.best_fitness = None
        self.worst_fitness = None
        self.reset()
        
    def reset(self):
        self.evaluations = []
        self.solutions = []
        
    def check_permutation(self, sequence):
        # Assumes numpy array
        return ((sequence >= 0) & (sequence < self.n)).all() and np.unique(sequence).shape[0] == sequence.shape[0]
    
    def fitness(self, x):
        f = self.fitness_nosave(x)
        self.solutions.append(x)
        self.evaluations.append(f)
        return f
  
    def distance_to_best(self, perm, distance):
        if self.best_sol is None:
            return np.nan
        return distance(perm, self.best_sol)
    
    class _Solution:
        def __init__(self, instance):
            self.instance = instance
            self.ship = Spaceship(instance.asteroids)
            self._sequence = []
            self.unvisited_ids = np.arange(instance.n)

        def step(self, to_id):
            assert to_id not in self._sequence 
            assert len(self._sequence) < self.instance.n
            if len(self._sequence) == 0:
                from_id = -1
            else:
                from_id = self._sequence[-1]
            current_time = self.get_time()
            self.instance.optimize_transfer(from_id, to_id, current_time, self)
            
            self._sequence.append(to_id)
            return self._sequence, self.ship.f
        
        def log_visit(self, visited_ast):
            assert visited_ast not in self._sequence 
            assert len(self._sequence) < self.instance.n
            prev = np.asarray(self.unvisited_ids)
            new = np.setdiff1d(prev, visited_ast)
            self.unvisited_ids = new
            self._sequence.append(visited_ast)
            return
            
        @property
        def sequence(self):
            return np.asarray(self._sequence, dtype=int)

        @property
        def f(self):
            return sum(self.ship.f)

        def get_cost(self):
            return sum(self.ship.costs)

        def get_time(self):
            return sum(self.ship.leg_times)


    def EmptySolution(self):
        return self._Solution(self)
    
    def PartialSolution(self, copy):
        sol = self._Solution(self)
        ast_ids, leg_times, fs, mans, costs = copy.ship.return_all()
        for i, ast_id in enumerate(ast_ids):
            sol.ship.add_ast(ast_id, (leg_times[2*i],leg_times[2*i+1]), 
                             fs[i], mans[i], costs[i])
            sol.log_visit(ast_id)
        return sol
        
    def CompleteSolution(self, sequence):
        self.check_permutation(sequence)
        sol = self._Solution(self)
        for ast_id in sequence:
            sol.step(ast_id)
        return sol

    def evaluate_transfer(self, from_id, to_id, current_time, t0, t1, 
                          only_cost = False, free_wait = False):
        """Here t0 is relative to current_time and 
        t1 is relative to current_time + t0"""
        from_orbit = self.get_ast_orbit(from_id)
        to_orbit = self.get_ast_orbit(to_id)
        start_epoch = def_epoch(START_EPOCH.mjd2000 + current_time)
        from_orbit = propagate(from_orbit, start_epoch) # ensures correct ref. time for transfer
        man, from_orbit, to_orbit = two_shot_transfer(from_orbit, to_orbit, 
                                                      t0 = t0, t1=t1) 
        cost = calc_cost(man, from_orbit, to_orbit)
        assert not (only_cost and free_wait)
        if only_cost:
            return cost, man
        if free_wait:
            t0 = 0
        f = VisitProblem.f(cost, t0+t1)
        return f, man

       
    def optimize_transfer(self, from_id, to_id, current_time, sol = False,
                          t0_bounds = VisitProblem.VISIT_BOUNDS, 
                          t1_bounds = VisitProblem.TRANSFER_BOUNDS, 
                          total_time_bounds = False, multi = 1, 
                          only_cost = False, free_wait = False, return_times = False):
        from_orbit = self.get_ast_orbit(from_id)
        from_orbit = propagate(from_orbit, def_epoch(START_EPOCH.mjd2000 + current_time))
        to_orbit = self.get_ast_orbit(to_id)
        if not sol:
            sol = self.EmptySolution()
        if free_wait:
            t0_bounds = (0., 3650) # 10 yrs max
        x0 = False
        cons = ()
        if total_time_bounds:
            t0_s, t0_f = t0_bounds
            t1_s, t1_f = t1_bounds
            assert total_time_bounds[1] >= total_time_bounds[0]
            # We cannot do less than t0_bounds[0], but we could do more (by arriving later if needed).
            t0_s = max(t0_s, total_time_bounds[0] - t1_f)
            t1_f = min(t1_f, total_time_bounds[1] - t0_s)
            t0_f = max(t0_s, total_time_bounds[1] - t1_s)
            t0_bounds = (t0_s, t0_f)
            t1_bounds = (t1_s, t1_f)
            x0 = (t0_s, max(30, t1_s))
            # x references array of [t0, t1]
            cons = ({'type': 'ineq', 'fun': lambda x: total_time_bounds[1] - (x[0] + x[1]) }, 
                    {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - total_time_bounds[0]})
        res, man = inner_minimize_multistart(
            lambda x: self.evaluate_transfer(from_id, to_id, current_time, x[0], x[1],
                                             only_cost = only_cost, free_wait = free_wait)[0],
                                             full_fun = lambda t0, t1: self.evaluate_transfer(from_id, to_id, current_time, t0, t1,
                                                                                          only_cost = only_cost, free_wait = free_wait), 
                                             multi = multi, bounds = (t0_bounds, t1_bounds), x0=x0, constraints = cons)
        from_orbit = propagate(from_orbit, def_epoch(START_EPOCH.mjd2000 + current_time + res[1])) # propagated to start of transfer
        cost = calc_cost(man, from_orbit, to_orbit)
        sol.ship.add_ast(to_id, (res[1], res[2]), res[0], man, cost)
        if return_times:
            return sol, res[1], res[2]
        return sol
  
        
    def get_nearest_neighbor_euclidean(self, from_id, unvisited_ids, current_time, 
                                       metric='euclidean'):
        epoch = def_epoch(START_EPOCH.mjd2000 + current_time)
        from_r = np.array([self.get_ast_orbit(from_id).eph(epoch)[0]])
        ast_r = np.array([ self.get_ast_orbit(ast_id).eph(epoch)[0] for ast_id in unvisited_ids ])
        ast_dist = distance.cdist(from_r, ast_r, metric)
        return unvisited_ids[np.argmin(ast_dist)]
    
    def get_nearest_neighbor_energy(self, from_id, unvisited_ids, current_time, 
                                    metric='orbital'):
        from_a = self.get_ast_orbit(from_id).orbital_elements[0]
        from_mass = self.get_ast_orbit(from_id).mu_self / G
        ast_a = np.array([self.get_ast_orbit(ast_id).orbital_elements[0] for ast_id in unvisited_ids])
        ast_mass = np.array([self.get_ast_orbit(ast_id).mu_self / G for ast_id in unvisited_ids])
        from_energy = -pk.MU_SUN * from_mass / (2*from_a)
        ast_energies = -pk.MU_SUN * ast_mass / (2*ast_a)# approximation that solar mass >> asteroid mass
        energy_diff = np.array([np.linalg.norm(np.subtract(from_energy, ast_energy)) for ast_energy in ast_energies])
        return unvisited_ids[np.argmin(energy_diff)]
    
    def nearest_phasing(self, from_id, unvisited_ids, current_time, 
                        metric='euclidean'):
        epoch = def_epoch(START_EPOCH.mjd2000 + current_time)
        from_orbit = self.get_ast_orbit(from_id)
        ast_orbits = np.array([ self.get_ast_orbit(ast_id) for ast_id in unvisited_ids ])
        neighbours = fixed_knn(ast_orbits, epoch, metric=metric, T=365.25)
        nearest, near_id, dist = neighbours.find_neighbours(from_orbit, 
                                                            query_type='knn', k=[1])
        return unvisited_ids[near_id]

    def build_nearest_neighbor(self, current_time, metric = 'euclidean', 
                               use_phasing=False, free_wait = False, only_cost = False):
        """
        Metrics:
            'euclidean' - nearest neighbour by physical distance
            'orbital' - nearest by orbital energy
        """
        if use_phasing:
            get_nearest_neighbor = self.nearest_phasing
        else:
            if metric == 'euclidean':
                get_nearest_neighbor = self.get_nearest_neighbor_euclidean
            elif metric == 'orbital':
                get_nearest_neighbor = self.get_nearest_neighbor_energy
            else:
                print('Invalid nearest neighbor metric. Check documentation for list of methods')
                return np.inf, [-1,-1], np.inf 
        
        from_id = -1 # From Earth
        sequence = [ from_id ]
        sol = self.EmptySolution()
        while len(sol.unvisited_ids) > 0:
            to_id = get_nearest_neighbor(from_id = from_id, unvisited_ids = sol.unvisited_ids,
                                         current_time = current_time, metric=metric)
            sol, t0, t1 = self.optimize_transfer(from_id, to_id, current_time, 
                                                 sol=sol, free_wait = free_wait, 
                                                 only_cost = only_cost, return_times=True)
            sol.unvisited_ids = np.setdiff1d(sol.unvisited_ids, to_id)
            #print(f'Departs from {from_id} at time {current_time + t0} after waiting {t0} days and arrives at {to_id} at time {current_time + t0 + t1} after travelling {t1} days, total cost = {f_total}')
            from_id = to_id
            sequence += [ to_id ]
            current_time += t0 + t1
            
        if not only_cost and not free_wait:
            np.testing.assert_allclose(sol.get_cost() + (sol.get_time() * VisitProblem.COST_TIME_TRADEOFF), sol.f, 1e-5)
        return sol, sequence

    def evaluate_sequence(self, sequence, current_time):
        """
        Calculates optimal transfers for a given sequence of asteroids
        """
        sol = self.EmptySolution()
        for i in range(1, len(sequence)):
            from_id = sequence[i-1]
            to_id = sequence[i]
            sol, t0, t1 = self.optimize_transfer(from_id, to_id, current_time, 
                                                 sol=sol, return_times = True)
        
            #print(f'Departs from {sequence[i-1]} at time {current_time + t0} after waiting {t0} days and arrives at {sequence[i]} at time {current_time + t0 + t1} after travelling {t1} days, total cost = {f_total}')        
            current_time += t0+t1
        return (sol.f, sol.ship.leg_times)

    def fitness_nosave(self, sequence):
        return self.CompleteSolution(sequence).f


    
