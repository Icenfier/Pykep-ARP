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
- [ ] ensure frame of reference is consistant (J2000 heliocentric ecliptic)
- [X] process_asteroids
- [ ] space_util, MU, MJD
  - [ ] arp.py and arp_vis import stuff from space_util
- [ ] arp
  - [ ] use pk.lambert_problem, pk.phasing.knn, pk.phasing.three_impulses_approx() OR pk.trajopt.pl2pl_N_impulses
  - [ ] update all '.get_ast_orbit', look at where functions are called
  - [ ] Spaceship.visit (from_orbit, VisitProblem)
  - [ ] Spaceship.get_energy_nearest (orbit propagation)
- [ ] arp_vis
  - [ ] propagation
  - [ ] plotting
- [ ] Test with transfer_example
  - [ ] check all files available
- [ ] redo entire arp, (minimising function in slsqp, can make less "general")
  - [ ] combine CommonProblem and VisitProlem?
  - [ ] check i update pickle filename
- [ ] check all pk.planet has input for planet radius
- [ ] update .propagate to my own propagate function
- [ ] better way for total dv cost?
- [ ] selecting which transfer orbit per transfer ( multirev)


## Other objectives
- [ ] Investigating using SPICE for asteroids
- [ ] using pykep for multirev Lambert
- [ ] generalize orbit creation (process_asteroids.py to_orbit, space_util.py OrbitBuilder)
- [ ] check imports, inc. importing specific functions
 
## File: space_util
- [ ] poliastro replacements
  - [X] spherical_to_cartesian (copied function from poliastro)
  - [X] M_to_E, E_to_nu (no longer needed, 'Planet' module requires M, not nu)
  - [ ] Body, Orbit, Planes
  - [ ] Maneuver
- [X] astropy time (replaced with pk.epoch)
- [ ] check epochs
- [ ] constants and unit conversions (import pykep equivalents)
- [ ] Asteroids get_orbit: propagate
  - [ ] may not even need to propagate, find where function is called
- [X] don't need M_to_nu
- [ ] OrbitBuilder
  - [X] don't need Sun object
  - [ ] replace Orbit class
    - [ ] from_vectors
    - [X] eliptic
    - [X] circular (not needed, removed)
- [X] Earth object, recreate as Planet
- [ ] apply_impulse: .propagate, r&v, from_vectors(r, v + dv, tmp.epoch)
  - [ ] called by launch_from_Earth
- [ ] launch_from_Earth: .propagate, units
	- [ ] calls apply_impulse
	- [ ] called by transfer_from_earth
- [ ] transfer_from_earth: poliastro spherical_to_cartesian nd Maneuver, epochs/to_timedelta, .propagate, to_orbit
	- [ ] calls launch_from_Earth	
- [ ] two_shot_transfer: from_orbit, to_orbit, .propagate, epochs, Maneuver
- [ ] check units!
- [ ] MAX_REVS either set in space_util, or add as function input

## File: arp
- [ ] space_util imports
  - [ ] Asteroids         
  - [ ] transfer_from_Earth
  - [ ] two_shot_transfer
  - [ ] START_EPOCH
  - [ ] Earth 
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
  
- [ ] VisitProblem(CommonProblem)
        called by Spaceship.visit
  - [ ] __init__(self, from_orbit, to_orbit)
          from_orbit, to_orbit
  - [ ] __call__(self, x)
- [ ] inner_minimize_multistart(fun, multi, bounds, method = 'SLSQP', constraints = (), **kwargs)
        called by AsteroidRoutingProblem.optimize_transfer_orbit
- [ ] inner_minimize(fun, x0, bounds, method = 'SLSQP', constraints = (), **kwargs)
        called by AsteroidRoutingProblem.optimize_transfer_orbit_total_time
- [ ] optimize_problem(problem, method = 'SLSQP', **kwargs)
        called by Spaceship.optimize

- [ ] Spaceship
  - [X] __init__(self, asteroids)
  - [ ] add_ast(self, ast_id, x, f, maneuvers), f+=?
          called by optimize
  - [X] optimize(self, ast_id, instance, **kwargs)
  - [X] launch(self, ast_id, **kwargs)
          called by AsteroidRoutingProblem._Solution.step
  - [ ] visit(self, ast_id, **kwargs)
  - [ ] get_energy_nearest(self, asteroids)
  - [ ] get_euclidean_nearest(self, asteroids)
  
- [ ] from problem import Problem
- [ ] AsteroidRoutingProblem(Problem)
  - [ ] _Solution
    - [ ] __init__(self, instance)
    - [ ] step(self, k)
    - [ ] x(self)
    - [ ] f(self)
    - [ ] get_cost(self)
    - [ ] get_time(self)
  - [ ] EmptySolution(self)
  - [ ] CompleteSolution(self, x)
  - [ ] PartialSolution(self, x)
  - [ ] read_instance(cls, instance_name)
  - [ ] __init__(self, n, seed)
  - [ ] nearest_neighbor(self, x, distance)
  - [ ] get_euclidean_distance(self, from_id, to_id, time)
  - [X] _evaluate_transfer_orbit(self, from_orbit, to_orbit, current_time, t0, t1, only_cost, free_wait)
          - [ ] sort only_cost and free_wait
  - [ ] evaluate_transfer(self, from_id, to_id, current_time, t0, t1, only_cost = False, free_wait = False)
  - [ ] optimize_transfer_orbit_total_time(self, from_orbit, to_orbit, current_time, total_time_bounds,
                                           only_cost = False, free_wait = False)
  - [ ] optimize_transfer_total_time(self, from_id, to_id, current_time, total_time_bounds,
                                     only_cost = False, free_wait = False)
  - [ ] optimize_transfer_orbit(self, from_orbit, to_orbit, current_time, t0_bounds, t1_bounds,
                                only_cost = False, free_wait = False, multi = 1)
  - [ ] optimize_transfer(self, from_id, to_id, current_time, t0_bounds, t1_bounds,
                          only_cost = False, free_wait = False, multi = 1)
  - [ ] get_nearest_neighbor_euclidean(self, from_id, unvisited_ids, current_time)
  - [ ] build_nearest_neighbor(self, current_time)
  - [ ] evaluate_sequence(self, sequence, current_time)
  - [ ] fitness_nosave(self, x)
  
- [ ] better way for total dv cost?
  
  
## File: transfer_example
- [ ] arp_instance = AsteroidRoutingProblem()
- [ ] arp_instance.CompleteSolution()
  - [ ] Problem.check_permutation
  - [ ] _Solution
- [ ] arp_instance.evaluate_sequence()
  - [ ] optimize_transfer_orbit
    - [ ] inner_minimize_multistart
- [ ] arp_instance.optimize_transfer()
- [ ] arp_instance.build_nearest_neighbor()
- [ ] arp_instance.evaluate_sequence()
- [ ] arp_instance.optimize_transfer_total_time()
- [ ] arp_instance.evaluate_transfer()
