"""
Functionality for tracking statistics about the search for trajectories.
"""
# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from functools import lru_cache
from itertools import combinations
from collections import namedtuple, Counter
import pykep as pk
import numpy as np

from arp import AsteroidRoutingProblem

# ==================================== ## ==================================== #
# ------------------------------------ # Run statistics


quality_nt = namedtuple('quality', ['score', 'agg', 'mass', 'tof'])
when_nt = namedtuple('when', [
    'nr_gen',
    'nr_traj',
    'nr_legs',
    'nr_legs_distinct',
    'nr_lambert'])



class traj_stats(object):
    
    def __init__(self, aco=None, path_handler=None):
        # links to the search algorithm & path handler instances
        self.aco = aco
        self.path = path_handler
        
        self.nr_traj = 0
        self.nr_legs = 0
        self.nr_legs_distinct = 0
        self.nr_lambert = 0
        
        self.when = []
        self.seq = []
        self.quality = []
        
        self.best_f = np.inf
        self.best_traj = None
        
    
    def log_mission(self, m):
        "log statistics from a completed trajectory"
        self.nr_traj += 1
        
        _tof = m.get_time() * pk.DAY2YEAR
        
        t = (self.aco.nr_gen, self.nr_traj, self.nr_legs,
             self.nr_legs_distinct, self.nr_lambert)
        
        self.when.append(t)
        self.seq.append(m.sequence)
        self.quality.append((m.f, _tof))
        
        if m.f < self.best_f:
            self.best_traj = m
            self.best_f = m.f
           
        
    
    def export(self):
        return self.when, np.asarray(self.seq), self.quality
        
    
    def load(self, *args):
        self.nr_traj, self.nr_lambert = None, None
        self.nr_legs, self.nr_legs_feasible = [None]*3
        self.when, self.seq, self.quality = args
        return self
        
        
    
    def best(self, max_i=None, max_qi=None):
        "Quality of the run's best solution."
        quality = (self.best_traj.f, self.best_traj.get_time() * pk.DAY2YEAR)
        return quality
        
    
    def best_seq(self, max_i=None, max_qi=None):
        "Sequence of asteroids in the run's best found solution."
        return self.best_traj.sequence
        
    
    def seq_similarity(self, discard_first=2, sample_size=10000, max_i=None):
        """
        Measure of similarity among all asteroid sequences produced in a run.
        Ignores sequences' first two bodies (Earth + 1st asteroid), by default,
        as these necessarily repeat across generated trajectories.
        
        NOT UPDATED
        """
        return seq_similarity(self.seq[:max_i], discard_first, sample_size)
    

# ==================================== ## ==================================== #
# ------------------------------------ # Sequence similarity measures

def seq_similarity(seqs, discard_first=0, sample_size=None):
    """
    Measure of the similarity among asteroid sequences (regardless of order).
    Optionally discards an initial sub-sequence of asteroids that may
    necessarily repeat across sequences (`discard_first`).
    Considers all possible pairs of sequences in `seq`, unless a `sample_size`
    is provided, in which case `sample_size` random pairings of sequences will
    be used instead to estimate the similarity.
    
    Returns mean & st. dev. of a value in [0, 1].
    * 0: no asteroid ever occurs in more than one sequence;
    * 1: all given sequences contain exactly the same asteroids.
    
    Note: among two given sequences, the maximum number of asteroids that may
    repeat is bounded by the size of the smallest sequence.
    """
    common_ratio = []
    
    if sample_size is None:
        pairs = combinations(seqs, 2)
    else:
        pairs = (
            (seqs[a], seqs[b if b < a else b + 1])
            for a in np.random.randint(0, len(seqs), size=sample_size)
            for b in np.random.randint(0, len(seqs) - 1, size=1)
            )
    
    for a, b in pairs:
        a = set(a[discard_first:])
        b = set(b[discard_first:])
        
        # calculate the fraction of asteroids common to both sequences
        # (the maximum is bounded by size of the smallest sequence)
        min_len = min(len(a), len(b))
        if min_len == 0:
            c = 0.
        else:
            c = len(a & b) / min_len
        
        common_ratio.append(c)
    
    return np.mean(common_ratio), np.std(common_ratio)
    

# ==================================== ## ==================================== #
# ------------------------------------ # Heuristic functions

def heuristic(rating, gamma=50, tabu=()):
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
    heur[tuple(tabu),] = 0.0
    return heur
    


def heuristic__norml(rating, gamma=50, tabu=()):
    """
    Converts the cost ratings for a group of trajectory legs (min is better)
    into a selection probability per leg (max is better).
    
    The greater the provided `gamma` exponent, the greedier the probability
    distribution will be, in favoring the best rated alternatives.
    
    Alternatives at the `tabu` indices will get a selection probabity of 0.0.

    ||| Alternative to the `heuristic()` function, that normalizes ratings based
    ||| on their extreme values. The resulting heuristic values are therefore 
    ||| sensitive to rating outliers.
    """
    tabu = tuple(tabu)
    
    # build a mask over ratings to considered (the non-tabu ones)
    incl_mask = np.ones(len(rating), dtype=np.bool)
    incl_mask[tabu,] = False
    
    # scale ratings into [0, 1], with best alternatives going from
    # low ratings/costs to high values (near 1).
    m = np.min(rating[incl_mask])
    M = np.max(rating[incl_mask])
    heur = 1.0 - (rating - m) / (M - m)
    
    # assign 0 selection probability to tabu alternatives
    heur[tabu,] = 0.0
    
    # scale probabilities, to emphasize best alternatives
    heur = heur**float(gamma)
    
    return heur
    