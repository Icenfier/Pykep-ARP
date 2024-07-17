# ARP TODO list

## Prereq.
- [X] Reinstall anaconda and Spyder
- [X] GitHub desktop, copy necessary files over
- [X] Add to conda
  - [X] Poliastro
  - [X] Astroquery
  - [X] Pykep
- [X] Ensure correct python version

## Main objectives
- [X] find alternative to Orbit class (planet module in pykep)
- [X] find alternative to orbit propagation (propagate.lagrangian OR planet.eph)
- [ ] ensure all units correct
- [ ] ensure start epochs correct (MJD2000)
- [X] process_asteroids
- [X] space_util
  - [X] two_shot_transfer
- [ ] arp
  - [X] use pk.lambert_problem, pk.phasing.knn, pk.phasing.three_impulses_approx() OR pk.trajopt.pl2pl_N_impulses
  - [X] Spaceship.visit (from_orbit, VisitProblem)
  - [ ] remove unecessary nearest neighbour functions
  - [ ] nearest neighbour by orbital energy 
  - [ ] combine CommonProblem and VisitProlem?
  - [ ] check free_wait and only_cost
- [ ] arp_vis
  - [X] epoch propagation
  - [X] plotting
  - [ ] add legend and labels
- [X] Test with transfer_example
- [X] check all pk.planet has input for planet radius
- [X] update .propagate to my own propagate function
- [X] better way for total dv cost
- [X] figure out why results are bad (unit conversions)
- [ ] add 'readable' option
  - [ ] .evaluate_sequence()
  - [ ] .optimize_transfer()
  - [ ] .evaluate_sequence()
  - [ ] .optimize_transfer_total_time()
  - [ ] .evaluate_transfer()
- [ ] add comments to explain code
- [ ] remove unused imports
- [X] SQSLP variants (removing wait time from obj. fun) (see build_nearest_neighbor only_cost)



## Other objectives
- [ ] using pykep for multirev Lambert
- [ ] generalize orbit creation (process_asteroids.py to_orbit, space_util.py OrbitBuilder)
- [ ] 'Orbital phasing indicators as heuristic estimators'
  - [ ] greedy nearest neighbour heuristic for distances in above paper
- [ ] Beam algorithms
  - [ ] Beam-search
  - [ ] Stochastic Beam
  - [ ] Beam-P-ACO
- [ ] convert to Python package
 
## File: space_util
- [X] poliastro replacements
  - [X] spherical_to_cartesian (copied function from poliastro)
  - [X] M_to_E, E_to_nu (no longer needed, 'Planet' module requires M, not nu)
  - [X] Body, Orbit, Planes
  - [X] Maneuver
- [X] astropy time (replaced with pk.epoch)
- [X] check epochs
- [ ] constants and unit conversions (import pykep equivalents)
- [X] Asteroids get_orbit: propagate
- [X] don't need M_to_nu
- [ ] OrbitBuilder
  - [X] don't need Sun object
  - [ ] replace Orbit class
    - [ ] from_vectors
    - [X] eliptic
    - [X] circular (not needed, removed)
- [X] Earth object, recreate as Planet
- [ ] apply_impulse, launch_from_earth, transfer_from_earth. needed?
  - [ ] apply_impulse: .propagate, r&v, from_vectors(r, v + dv, tmp.epoch)
    - [ ] called by launch_from_Earth
  - [ ] launch_from_Earth: .propagate, units
	- [ ] calls apply_impulse
    - [ ] called by transfer_from_earth
  - [ ] transfer_from_earth: poliastro spherical_to_cartesian nd Maneuver, epochs/to_timedelta, .propagate, to_orbit
	- [ ] calls launch_from_Earth	
- [X] two_shot_transfer: from_orbit, to_orbit, .propagate, epochs, Maneuver
- [ ] check units!
- [ ] MAX_REVS either set in space_util, or add as function input

## File: arp
- [X] assert_bounds(x, bounds) (ensures all x0 within bounds)
        called by VisitProblem
- [X] get_default_opts (allows various methods, currently set to slsqp)
        called by inner_minimize_multistart, inner_minimize, optimize_problem

- [X] CommonProblem
  - [X] __init__(self)
  - [X] to_Bounds(self), do we need this?? not called
  - [X] f(self, cost, time)
          called by update_best
  - [X] update_best(self, x, cost, time, man)
          called by VisitProblem __call__
  
- [X] VisitProblem(CommonProblem)
        called by Spaceship.visit
  - [X] __init__(self, from_orbit, to_orbit)
          from_orbit, to_orbit
  - [X] __call__(self, x)
- [ ] inner_minimize_multistart(fun, multi, bounds, method = 'SLSQP', constraints = (), **kwargs)
  - [ ] sort multi bounds
        called by AsteroidRoutingProblem.optimize_transfer_orbit
- [X] inner_minimize(fun, x0, bounds, method = 'SLSQP', constraints = (), **kwargs)
        called by AsteroidRoutingProblem.optimize_transfer_orbit_total_time
- [X] optimize_problem(problem, method = 'SLSQP', **kwargs)
        called by Spaceship.optimize

- [ ] Spaceship
  - [X] __init__(self, asteroids)
  - [X] add_ast(self, ast_id, x, f, maneuvers)
          called by optimize
  - [X] optimize(self, ast_id, instance, **kwargs)
  - [X] launch(self, ast_id, **kwargs)
          called by AsteroidRoutingProblem._Solution.step
  - [X] visit(self, ast_id, **kwargs)
  - [ ] get_energy_nearest(self, asteroids)
    - [ ] only used by nearest_neighbor, alternative to build_nearest_neighbor
      - [ ] ask if we want just euclidean dist. or if we want the option for energy
  - [ ] get_euclidean_nearest(self, asteroids)
  
- [ ] AsteroidRoutingProblem(Problem)
  - [X] _Solution
    - [X] __init__(self, instance)
    - [X] step(self, k)
    - [X] x(self)
    - [X] f(self)
    - [X] get_cost(self)
    - [X] get_time(self)
  - [X] EmptySolution(self)
  - [X] CompleteSolution(self, x)
  - [ ] PartialSolution(self, x), used in nearest_neighbour calc
  - [ ] read_instance(cls, instance_name) find where this is called
  - [X] __init__(self, n, seed)
  - [ ] nearest_neighbor(self, x, distance)
    - [ ] replace build_nearest_neighbor?
    - [ ] update get_energy_nearest
  - [ ] get_euclidean_distance(self, from_id, to_id, time) needed??
  - [X] _evaluate_transfer_orbit(self, from_orbit, to_orbit, current_time, t0, t1, only_cost, free_wait)
  - [X] evaluate_transfer(self, from_id, to_id, current_time, t0, t1, only_cost = False, free_wait = False)
  - [X] optimize_transfer_orbit_total_time(self, from_orbit, to_orbit, current_time, total_time_bounds,
                                           only_cost = False, free_wait = False)
  - [X] optimize_transfer_total_time(self, from_id, to_id, current_time, total_time_bounds,
                                     only_cost = False, free_wait = False)
  - [X] optimize_transfer_orbit(self, from_orbit, to_orbit, current_time, t0_bounds, t1_bounds,
                                only_cost = False, free_wait = False, multi = 1)
  - [X] optimize_transfer(self, from_id, to_id, current_time, t0_bounds, t1_bounds,
                          only_cost = False, free_wait = False, multi = 1)
  - [X] get_nearest_neighbor_euclidean(self, from_id, unvisited_ids, current_time)
  - [X] build_nearest_neighbor(self, current_time)
  - [X] evaluate_sequence(self, sequence, current_time)
  - [ ] fitness_nosave(self, x)
  
- [ ] better way for total dv cost?
- [ ] combine some functions
  
  
## File: transfer_example
- [X] arp_instance = AsteroidRoutingProblem()
- [X] arp_instance.CompleteSolution()
  - [X] Problem.check_permutation
  - [X] _Solution
- [X] arp_instance.evaluate_sequence()
  - [X] optimize_transfer_orbit
    - [X] inner_minimize_multistart
- [X] arp_instance.optimize_transfer()
- [X] arp_instance.build_nearest_neighbor()
- [X] arp_instance.evaluate_sequence()
- [X] arp_instance.optimize_transfer_total_time()
- [X] arp_instance.evaluate_transfer()
