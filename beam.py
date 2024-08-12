# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.

import pykep as pk
import numpy as np
from space_util import START_EPOCH
from paco_traj import traj_stats
from arp import AsteroidRoutingProblem
from beam_paco__gtoc5.paco.paco import paco, beam_paco
from tqdm import tqdm, trange
from beam_paco__gtoc5.experiments import initialize_rng
import os
from phasing import fixed_knn
           

def rate__orbital(dep_ast, arr_asteroids, dep_t, leg_dT, **kwargs):
    """
    Orbital Phasing Indicator.
    See: http://arxiv.org/pdf/1511.00821.pdf#page=12
    
    Estimates the cost of a `leg_dT` days transfer from departure asteroid
    `dep_ast` at epoch `dep_t`, towards each of the available asteroids.
    """
    dep_t = pk.epoch(dep_t, 'mjd')
    leg_dT *= pk.DAY2SEC
    knn = fixed_knn(arr_asteroids, dep_t, T=leg_dT)
    neighb, neighb_ids, dists = knn.find_neighbours(dep_ast, k=([1] if len(arr_asteroids)==1 else len(arr_asteroids)))

    dists = [x for y,x in sorted(zip(neighb_ids, dists))]
    return dists #np.linalg.norm(ast_orbi - orbi, axis=1)


def rate__orbital_2(dep_ast, arr_asteroids, current_time, leg_dT, **kwargs):
    """
    Refinement over the Orbital indicator.
    Attempts to correct bad decisions made by the linear model by checking
    which asteroids are close at arrival, and not only which are close at the
    beginning.
    """
    dep_t = START_EPOCH.mjd2000 + current_time

    r1 = rate__orbital(dep_ast, arr_asteroids, dep_t, leg_dT)
    r2 = rate__orbital(dep_ast, arr_asteroids, dep_t + leg_dT, leg_dT, neg_v=True)

    return np.mean([r1, r2], axis=0)


def heuristic(rating, unvisited, n, gamma=50):#, tabu=()):
    """
    Converts the cost ratings for a group of trajectory legs (min is better)
    into a selection probability per leg (max is better).
    
    The greater the provided `gamma` exponent, the greedier the probability
    distribution will be, in favoring the best rated alternatives.
    
    Alternatives at the `tabu` indices will get a selection probabity of 0.0.
    """
    # Rank the cost ratings. Best solutions (low cost) get low rank values
    rank = np.argsort(np.argsort(rating))
    # scale ranks into (0, 1], with best alternatives going from
    # low ratings/costs to high values (near 1).
    heur = 1. - rank / len(rank)
    # scale probabilities, to emphasize best alternatives
    heur = heur**float(gamma)
    # assign 0 selection probability to tabu alternatives
    full_heur = np.zeros(n)
    full_heur[unvisited] = heur
    return full_heur

def add_asteroid(ph, mission, next_ast, use_cache=False, stats=None, **kwargs):
    """
    Extend `mission` by visiting a new asteroid.
    """
        
    assert isinstance(next_ast, (int, np.integer)) and 0 <= next_ast <= 7075, \
        "Next asteroid should be given as an integer."
    next_ast = int(next_ast)
    
    if stats is not None:
        # increment total number of [rendezvous] legs defined
        stats.nr_legs += 1
    mission = rendezvous_leg(ph, mission, next_ast, stats=stats, **kwargs)
    
    return True, mission
    


# ==================================== ## ==================================== #
# ------------------------------------ # Define rendezvous and self-flyby legs

def rendezvous_leg(ph, mission, next_ast, leg_dT=None, leg_dT_bounds=None,
                   obj_fun=None, stats=None, **kwargs):
    """
    Define the leg that extends `mission` by performing a rendezvous with
    asteroid `next_ast`.
    """
    if leg_dT is None and leg_dT_bounds is None:
        leg_dT_bounds = (0,730)#rvleg_dT_bounds

    if len(mission.sequence) > 0:
        dep_ast = mission.sequence[-1]
        dep_t = mission.get_time()
    else:
        dep_ast = -1
        dep_t = 0
    mission, t0, t1 = ph.optimize_transfer(dep_ast, next_ast, dep_t, sol = mission, return_times = True) 

    if stats is not None:
        stats.nr_legs_distinct += 1
        
    mission.log_visit(next_ast)
    
    return mission

# ==================================== ## ==================================== #
# ---------------------------------- # Define problem and experimental procedure

class beam_arp(AsteroidRoutingProblem):
    """
    Handler for trajectories using beam search and P-ACO.
    """
    
    # indication that the costs of asteroid transfers are *not* symmetric
    symmetric = False
    
    # indication that the path handler, via the `.tabu()` method, *disallows*
    # previously visited nodes/asteroids from being revisited.
    allows_revisits = False
    
    
    def __init__(self, n=None, seed=None, ratingf=None, ref_dT=125, gamma=50, 
              add_ast_args = None):
        self.n = n
        self.seed = seed
        # phasing indicator used to rate destination asteroids
        self.rate_destinations = rate__orbital_2 if ratingf == None else ratingf
        # reference transfer time used in the Orbital phasing indicator
        self.ref_dT = ref_dT
        # "greediness exponent" used in the `heuristic` function
        self.gamma = gamma
        # parameters configuring the addition of new legs to missions.
        self.add_ast_args = {} if add_ast_args is None else add_ast_args 
        super().__init__(self.n, self.seed)
        
    
    def initialize(self, aco):
        "ACO is starting a new run. Reset all run state variables."
        self.add_ast_args['stats'] = self.stats = traj_stats(aco, self)
        
    
    def heuristic(self, ant_path):
        "Heuristic used to estimate the quality of node transitions."
        if len(ant_path.sequence)>0:
            dep_ast = self.get_ast_orbit(ant_path.sequence[-1])
        else:
            dep_ast = self.get_ast_orbit(-1)
        current_time = ant_path.get_time()
        asteroids = np.array([self.get_ast_orbit(ast) for ast in ant_path.unvisited_ids])
        rating = self.rate_destinations(dep_ast, asteroids, current_time, leg_dT=self.ref_dT)
        return heuristic(rating, ant_path.unvisited_ids, self.n, self.gamma)
    
    
    def start(self):
        "Start a new path through the graph."
        self.stop_walk = False
        sol = self.EmptySolution()
        self.current_time = 0
        return sol
        
    
    def add_node(self, ant_path, node):
        "Extend an ant's path with a new visited node."
        success, ant_path = add_asteroid(self, ant_path, int(node), **self.add_ast_args)
        self.stop_walk = not success
        return ant_path
        
    
    def get_nodes(self, ant_path):
        "Get the list of nodes visited so far along the ant's path."
        return ant_path.sequence 
        
    
    def get_links(self, ant_path):
        "Get an iterator over node transitions performed along an ant's path."
        path_nodes = self.get_nodes(ant_path)
        return zip(path_nodes[:-1], path_nodes[1:])
        
    
    def stop(self, ant_path, force_stop=False):
        "Indicate whether an ant's path should be terminated."

        if len(ant_path.sequence) == self.n:
            self.stats.log_mission(ant_path)
            return True
        if self.stop_walk or force_stop:
            self.stats.log_mission(ant_path)
            return True
        return False
        
    
    def evaluate(self, ant_path):
        "Quality function used to evaluate an ant's path through the graph."
        
        return ant_path.f 
        
    
    def sort(self, evaluated_paths, r=None):
        """
        Given a list of `evaluated_paths` (a list of (quality, ant_path)
        tuples), return a ranked list, with the top `r` paths (or all, if
        unspecified), sorted by decreasing order of quality
        (decreasing f value).
        """
        if r == 1:
            return [min(evaluated_paths, key=lambda i: i[0])]
        return sorted(evaluated_paths, key=lambda i: i[0])[:r] 
   
   ################################################## 
    
class experiment(object):
    "Experiment controller"
    
    def __init__(self, path_handler, nr_runs=100, log_data_every=2, 
                 max_nr_legs=None, max_nr_gens=None, path='', extra_info=None,
                 **kwargs):
        
        self.nr_runs = nr_runs
        self.max_nr_legs = max_nr_legs
        self.max_nr_gens = max_nr_gens
        self.log_data_every = log_data_every
        
        self.set_parameters(**kwargs)
        
        # instantiate the search method
        self.aco = self.aco_class(
            nr_nodes=len(path_handler.asteroids.ast_orbits), path_handler=path_handler,
            random_state=None, **self.aco_args)
        
        
        self.set_filename(path, extra_info)
        
    
    def set_parameters(self, variant='P-ACO', **kwargs):
        "Parameter settings for the experimental setup's different variants."
        
        self.variant = variant
        
        self.aco_class = {
            (True) : paco,
            (False) : beam_paco
            }[(variant == 'P-ACO')]
        
        # default parameter settings
        self.aco_args = dict(pop_size=3, ants_per_gen=25, alpha=1., beta=5.,
                             prob_greedy=0.5, use_elitism=True)
        
        # parameter changes for the Beam Search variants
        # ('beam_width' and 'branch_factor' should be defined via `kwargs`)
        diff = {
            # Hybridization of Beam Search and P-ACO
            'Beam P-ACO' : dict(),
            
            # Beam Search variant where successor nodes are picked
            # non-deterministically from a distribution defined solely by the
            # heuristic function. Equivalent to 'Beam P-ACO', in that it's a
            # beam search performing random restarts, but here with no knowledge
            # transfer between restarts (alpha=0).
            'Stochastic Beam' : dict(alpha=0., beta=1.),
            
            # Standard (deterministic) Beam Search
            'Beam Search' : dict(alpha=0., beta=1., prob_greedy=1.0),
            
            }.get(self.variant, {})
        
        self.aco_args.update(diff)
        self.aco_args.update(kwargs)
        
        # 'beam_width' is accepted as an alias to 'ants_per_gen'.
        # to enforce consistency, ensure 'ants_per_gen' is always set.
        if 'beam_width' in self.aco_args:
            self.aco_args['ants_per_gen'] = self.aco_args['beam_width']
        
    
    
    def set_filename(self, path='', extra_info=None):
        """
        SAVING NOT IMPLEMENTED
        """
        #"Set path and name of the file into which to save the results."
        #self.path = path if path[-1] == '/' else (path + '/')
        # create directory to store results (in case it doesn't exist yet)
        #if not os.path.exists(path):
        #    os.makedirs(path)
        
        #bf = self.aco_args.get('branch_factor', None)
        #if self.pareto_elite:
        #    pareto_elite_str = ' (pareto %df)' % self.aco.nr_elite_fronts
        #self.filename = \
        #    'pop_size={pop_size:d}, ants_per_gen={ants_per_gen:d}, ' \
        #    'alpha={alpha:.1f}, beta={beta:.1f}, prob_greedy={prob_greedy:.2f}'\
        #    ', elitism={use_elitism}{_branch_factor}, ' \
        #    'variant={variant:s}{pareto_elite}{extra_info}.pkl'.format(
        #        _branch_factor=(', branch_factor=%d' % bf) if bf else '',
        #        variant=self.variant,
        #        pareto_elite='' if not self.pareto_elite else pareto_elite_str,
        #        extra_info=(', %s' % extra_info) if extra_info else '',
        #        **self.aco_args)
        
    
    def show_setup(self):
        "Display the experimental setup's configuration."
        print('\nvariant: ' + self.variant, end='\n\n')
        #print(self.path + '\n' + self.filename, end='\n\n')
        print(self.aco_args); print('')
        print(self.aco.path.__class__, self.aco_class, end='\n\n')
      
    
    def print_best(self):
        "Print information about the best sequence found to date in a run."
        print('')
        st = self.aco.path.stats
        print(st.best())
        print('')
        
    
    def stats_best(self):
        "Obtain statistics about the best sequence found to date in a run."
        (q, m) = self.aco.best
        
        return '[Score: %2d, Time: %6.3f%s]' % (
            m.f, m.get_time() * pk.DAY2YEAR,
            '; |e|=%d' % len(self.aco.elite))
            
    def return_best(self):
        f, best = self.aco.best 
        return best        
        
    
    def run(self, seed=None):
        "Perform a full, independent run."
        self.aco.random, seed = initialize_rng(seed)
        print('Seed: ' + str(seed))
        
        print()
        prog_bar = tqdm(total=self.max_nr_legs, leave=True, position=0)
        
        self.aco.initialize()
        stats = self.aco.path.stats
        
        while (self.max_nr_legs is None) or (stats.nr_legs < self.max_nr_legs):
            self.aco.build_generation()
#            self.print_best()
            self.aco.nr_gen += 1
            prog_bar.desc = self.stats_best() + ' '
            prog_bar.update(stats.nr_legs - prog_bar.n)
            if self.max_nr_gens == self.aco.nr_gen:
                break
        
        prog_bar.desc = ''; prog_bar.refresh();
        prog_bar.close()
        
    
    def start(self):
        "Conduct an experiment, by performing multiple independent runs."
        self.show_setup()
        
        #stats, trajs = [], []
        #fname = self.path + self.filename
        
#        for r in range(self.nr_runs):
        for r in trange(self.nr_runs, leave=True, desc='RUNS'):
            
            self.run(seed=r)
            self.print_best()
            
            # SAVIMG NOT IMPLEMENTED
            # save experimental data
            #stats.append(self.aco.path.stats.export())
            #trajs.append(self.aco.best)
            #if (r + 1) % self.log_data_every == 0:
            #    safe_dump(data = stats, fname = fname, append=True)
            #    safe_dump(data=trajs, fname=fname[:-3] + 'TRAJ.pkl', append=True)
            #    stats, trajs = [], []
        
        #if stats != []:
            #safe_dump(stats, fname, append=True)
            #safe_dump(trajs, fname[:-3] + 'TRAJ.pkl', append=True)
     
            
    
    