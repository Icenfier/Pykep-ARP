# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.


from collections import deque

import numpy as np


# ==================================== ## ==================================== #

class paco(object):
    """
    Population-based Ant Colony Optimization (P-ACO).
    Introduced by Michael Guntsch & Martin Middendorf (2002-2004).
    
    References
    ==========
    [1] http://dx.doi.org/10.1007/3-540-46004-7_8
    [2] http://dx.doi.org/10.1007/3-540-45724-0_10
    [3] http://dx.doi.org/10.1007/3-540-36970-8_33
    [4] http://d-nb.info/1013929756
    [5] http://iridia.ulb.ac.be/IridiaTrSeries/link/IridiaTr2011-006.pdf
        http://iridia.ulb.ac.be/supp/IridiaSupp2011-010/
    """
    
    def __init__(self, nr_nodes, path_handler, pop_size=3, ants_per_gen=25,
                 pher_init=None, pher_max=1.0, alpha=1., beta=5.,
                 prob_greedy=0.9, use_elitism=True, random_state=None,
                 **kwargs):
        # handler for solutions built by this P-ACO instance
        self.path = path_handler
        
        # number of combinatorial elements being assembled into sequences
        self.nr_nodes = nr_nodes
        # number of "champion" ants logged in the pheromone matrix (k)
        self.pop_size = pop_size
        # number of ants spawned per generation (m)
        self.ants_per_gen = ants_per_gen
        
        # minimum/initial pheromone concentration on an edge (\tau_{init})
        # (implements the convention of having rows/columns of initial
        # pheromone values summing to 1.0)
        self.pher_min = pher_init
        if self.pher_min is None:
            non_zero_cols = nr_nodes - (0 if self.path.allows_revisits else 1)
            self.pher_min = 1.0 / non_zero_cols
        
        # maximum pheromone concentration on an edge (\tau_{max})
        self.pher_max = pher_max
        
        # amounth of pheromone one ant lays down on an edge of the graph
        self.pher_incr = (self.pher_max - self.pher_min) / self.pop_size
        # in symmetric problems ants lay the same total amount of pheromone, but
        # split along both directions (ij and ji). NOTE: total pheromone in a
        # link may then range in [pher_min, pher_min + pop_size * pher_incr],
        # and not in [pher_min, pher_max / 2] as indicated in [1] and [4].
        self.pher_incr /= (2.0 if self.path.symmetric else 1.0)
        
        # exponents indicating the relative importance of pheromone (alpha)
        # and heuristic (beta) contributions to nodes' selection probabilities
        assert alpha > 0.0 or beta > 0.0, \
            'At least one of `alpha`/`beta` must be defined.'
        self.alpha = alpha
        self.beta = beta
        
        # probabiliy of an ant greedily/deterministically choosing the next
        # node to visit (q_0)
        self.prob_greedy = prob_greedy
        
        # Indication of whether one slot in the population is reserved for the
        # best solution seen so far. Elitism implemented as specified in [2].
        self.use_elitism = bool(use_elitism)
        
        self.random = np.random if random_state is None else random_state
        
        self._ph = np.zeros(self.nr_nodes)
        
        self.initialize()
        
    
    def initialize(self):
        "Reset all run state variables, and prepare to start a new run."
        # full paths taken by the ants that have deposited pheromones
        pop_len = self.pop_size - (1 if self.use_elitism else 0)
        self.population = deque(maxlen=pop_len)
        # Edges out from each given node along which ants have previously
        # deposited pheromones. Example: self.popul_pheromone[i] = [j,k,j]
        # indicates 3 ants have previously visited node i, two of which
        # moved on to j, while a third moved on to k.
        self.popul_pheromone = [deque() for i in range(self.nr_nodes)]
        
        if self.use_elitism:
            self.elite = deque(maxlen=1)
            self.elite_pheromone = [deque() for i in range(self.nr_nodes)]
        
        self.nr_gen = 0
        self.generation = None
        self.best = None
        
        self.path.initialize(self)
        
    
    def pheromone(self, ant_path=None, current_node=None):
        """
        Obtain the pheromone contribution to the probability distribution by
        which a successor node for the current `ant_path` is to be chosen.
        
        Produces the pheromone matrix row containing all pheromones deposited by
        previous ants, in their transitions from the node presently occupied by
        the considered ant.
        Enforces tabus: nodes the path handler indicates should be excluded from
        consideration as successor from `ant_path` receive a probability of 0.0.
        
        May alternatively be called by specifying only the `current_node`.
        """
        if current_node is None:
            try:
                current_node = self.path.get_nodes(ant_path)[-1]
            except IndexError:
                current_node = -1
            tabu = self.path.get_nodes(ant_path)
        else:
            assert ant_path is None, 'Redundant arguments given.'
            tabu = [] if self.path.allows_revisits else [current_node]
        
#        ph = np.zeros(self.nr_nodes) + self.pher_min
        ph = self._ph
        ph.fill(self.pher_min)
        
        for s in self.popul_pheromone[current_node]:
            ph[s] += self.pher_incr
        
        if self.use_elitism:
            for s in self.elite_pheromone[current_node]:
                ph[s] += self.pher_incr
        
        # give a 0.0 pheromone value to nodes that should be excluded from
        # consideration in the choice of successor node
        ph[list(tabu)] = 0.0
        
        return ph
        
    
    def pheromone_matrix(self):
        """
        Generates the full pheromone matrix, by stacking the rows produced
        in calls to .pheromone().
        """
        rows = [
            self.pheromone(current_node=i).copy()
            for i in range(self.nr_nodes)
            ]
        return np.vstack(rows)
      
    
    def _get_links(self, ant_path):
        """
        Get an iterator over the node transitions in a unit of information
        stored in the population (by default: a single ant's path).
        """
        return self.path.get_links(ant_path)
        
    
    def lay_down_pheromone(self, ant_path, update_elite=False):
        "Deposit pheromone along the path walked by an ant."
        # pick the population that is to be updated (the main one, or the elite)
        if update_elite:
            population, pheromone = self.elite, self.elite_pheromone
        else:
            population, pheromone = self.population, self.popul_pheromone
        
        # population behaves as a FIFO-queue: oldest ant is removed
        # in case population size limit has been reached.
        # Implements the "Age" population update strategy from P-ACO's papers.
        if len(population) == population.maxlen:
            ant_out = population.popleft()
            for (i, j) in self._get_links(ant_out):
                n = pheromone[i].popleft()
#                assert n == j, 'removed unexpected pheromone'
                if self.path.symmetric:
                    n = pheromone[j].popleft()
#                    assert n == i, 'removed unexpected pheromone'
        
        # add new `ant_path`
        population.append(ant_path)
        for (i, j) in self._get_links(ant_path):
            pheromone[i].append(j)
            if self.path.symmetric:
                pheromone[j].append(i)
        
    
    def ant_walk(self):
        "Create an ant, and have it travel the graph."
        ant_path = self.path.start()
        
        while not self.path.stop(ant_path):
            p = None
            if self.alpha > 0.0:
                p = self.pheromone(ant_path)**self.alpha
            if self.beta > 0.0:
                b = self.path.heuristic(ant_path)**self.beta
                p = b if p is None else (p * b)
            
            if self.random.rand() < self.prob_greedy:
                # greedy selection
                next_node = np.argmax(p)
            else:
                # probabilistic selection
                p /= p.sum()
                next_node = self.random.choice(self.nr_nodes, p=p)
            
            self.path.add_node(ant_path, next_node)
        
        return ant_path
        
    
    def build_generation(self):
        'Have a "generation" of ants travel the graph.'
        self.generation = []
        
        for _ in range(self.ants_per_gen):
            path = self.ant_walk()
            cost = self.path.evaluate(path)
            self.generation.append((cost, path))
            
        
        self.process_generation()
      
    
    def process_generation(self):
        """
        Process the most recent generation of ant walks:
        * identify the generation's most successful ant;
        * have it lay down pheromones along the path it took;
        * keep track of the best ant path seen so far (self.best);
        * update the elitist solution (and its pheromones), if applicable.
        """
        champion = self.path.sort(self.generation, r=1)[0]
        if self.alpha > 0.0:
            self.lay_down_pheromone(champion[1], update_elite=False)
        
        if self.best is None:
            self.best = champion
        else:
            self.best = self.path.sort([self.best, champion], r=1)[0]
        
        # if self.best (best ant path seen so far) now holds the current
        # generation's champion, then update the elitist solution.
        # In the current generation, the the same ant path will then then lay
        # down pheromone both in the main population, and in the elite one.
        # This is in agreement with the specification in [2].
        if self.alpha > 0.0 and self.best is champion:
            self.lay_down_pheromone(champion[1], update_elite=True)

    
    def solve(self, nr_generations=10000, reinitialize=False):
        """
        Solve the combinatorial problem. Over a span of multiple generations,
        ants walk through the graph, depositing pheromones which then influence
        the paths taken in subsequent walks.
        """
        if reinitialize:
            self.initialize()
        for g in range(nr_generations):
            self.nr_gen += 1
            self.build_generation()
        return self.best
    
    

# ==================================== ## ==================================== #

class beam_paco(paco):
    """
    Beam P-ACO: hybridization of P-ACO with Beam Search.
    """
    
    def __init__(self, *args, beam_width=None, branch_factor=None, **kwargs):
        # `beam_width`, the number of solutions kept per path depth, is enforced
        # via the number of `ants_per_gen`. Should the argument be specified
        # with this alias, it's copied to `ants_per_gen`, possibly overwriting
        # a redundant/inconsistent specification in it.
        if beam_width is not None:
            kwargs['ants_per_gen'] = beam_width
        
        super(beam_paco, self).__init__(*args, **kwargs)
        
        # nr. of successor nodes an ant should branch into per step of its path
        # (defaults to 2 * pop_size, if unspecified, ensuring at least pop_size
        # successors are generated without using pheromone information)
        if branch_factor is None:
            branch_factor = 2 * self.pop_size
        self.branch_factor = branch_factor
        
    
    def ant_walk(self, ant_path=None):
        """
        Have an ant take a step in its path through the graph, towards multiple
        successor nodes.
        """
        if ant_path is None:
            ant_path = self.path.start()
        
        # build nodes' selection probability distribution
        p = None
        if self.alpha > 0.0:
            p = self.pheromone(ant_path)**self.alpha
        if self.beta > 0.0:
            b = self.path.heuristic(ant_path)**self.beta
            p = b if p is None else (p * b)
        
        # select the `next_nodes` to branch into
        nz = np.nonzero(p)[0]
        nz = np.intersect1d(nz, ant_path.unvisited_ids)
        if len(nz) <= self.branch_factor:
            # if there are fewer than `branch_factor` nodes that can be branched
            # into (for instance, if most nodes are tabu), then branch into all
            # available ones, and skip computations below
            next_nodes = nz
        elif self.random.rand() < self.prob_greedy:
            # greedy selection
            # (identify indices into the `branch_factor` highest values in `p`)
            next_nodes = np.argpartition(-p, self.branch_factor - 1)
            next_nodes = next_nodes[:self.branch_factor]
        else:
            # probabilistic selection
            p /= p.sum()
            next_nodes = self.random.choice(
                self.nr_nodes, size=self.branch_factor, replace=False, p=p)
        
        # branch the ant's path into all successor nodes in `next_nodes`
        complete, ongoing = [], []
        for n in next_nodes:
            ap = self.path.PartialSolution(ant_path)
            ap = self.path.add_node(ap, n)
            (complete if self.path.stop(ap) else ongoing).append(ap)
        
        return complete, ongoing
        
    
    def build_generation(self):
        """
        Have a "generation" of ants travel the graph.
        Performs a full Beam Search, a constrained breadth-first search on a
        tree of ant paths: each tree node is branched into `self.branch_factor`
        successor nodes, and per tree depth only the `self.ants_per_gen` best
        solutions (the beam's width) are kept and carried forward to the next
        level. An ant path is here the succession of edges from the tree's root
        down to a leaf node.
        The generation's best solution is defined as the best ranked among the
        longest produced paths (those that reached the greatest tree depth).
        ph is path handler object"""
#        ongoing = [None] * self.ants_per_gen
        # single root node; all paths start from the same initial conditions
        ongoing = [None]
        
        while ongoing != []:
            # extend all the still ongoing paths, and split outcomes between
            # completed paths, and those that should still be further extended.
            complete, incomplete = [], []
            
            for ant_path in ongoing:
                c, o = self.ant_walk(ant_path)
                complete.extend(c)
                incomplete.extend(o)
            
            # evaluate and sort the incomplete paths
            incomplete = [(self.path.evaluate(p), p) for p in incomplete]
            incomplete = self.path.sort(incomplete)
            
            # select the best `ants_per_gen` paths out from those that are still
            # incomplete, and discard the remaining ones
            ongoing = [p for (c, p) in incomplete[:self.ants_per_gen]]
            
            # signal to the path handler that paths being discarded should be
            # forcibly stopped (trigger eventual clean up steps)
            for (c, p) in incomplete[self.ants_per_gen:]:
                self.path.stop(p, force_stop=True)
        
#        # All paths have completed. Pick the `ants_per_gen` best among the
#        # longest paths, and discard the remaining ones.
#        complete = [(self.path.evaluate(p), p) for p in complete]
#        self.generation = self.path.sort(complete, r=self.ants_per_gen)
        
        # Define the generation's paths as being *all* the complete paths having
        # the same maximal length. Does not prune down to at most `ants_per_gen`
        # solutions. In the end, `.generation` may hold more, or even less than
        # that number of solutions (if few feasible solutions reached the
        # maximal observed length).
        self.generation = [(self.path.evaluate(p), p) for p in complete]
        
        self.process_generation()
      