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
  - [ ] check if original GTOC asteroid file contains asteroid names
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
- [ ] copyright and credit stuff
- [ ] update to reflect changes:
  - [ ] cego.py (nearest_neighbor)
    - [ ] install R and rpy2?
  - [ ] umm.py
  - [ ] greedy_nn

## Other objectives
- [ ] acessing multirev Lambert, choosing best multirev results
- [ ] generalize orbit creation (process_asteroids.py to_orbit, space_util.py OrbitBuilder)
- [ ] 'Orbital phasing indicators as heuristic estimators'
  - [ ] greedy nearest neighbour heuristic for distances in above paper
- [ ] Beam algorithms
  - [ ] Beam-search
  - [ ] Stochastic Beam
  - [ ] Beam-P-ACO
- [ ] add path details so I can better organise files?
- [ ] add global cache? would help avoid reevaluation, can use to access maneuvers without having to explicitly return them?
- [ ] convert to Python package

## Repository: beam_paco_gtoc5
https://github.com/lfsimoes/beam_paco__gtoc5
- [ ] figure out which files are needed/useful
  - [ ] gtoc5
    - [ ] to replace/use in constants.py (ignoring any GTOC5 specific):
      - [ ] MU_SUN = 1.32712440018e11 * 1000**3
      - [ ] AU = 1.49597870691e8 * 1000
      - [ ] G0 = 9.80665
      - [ ] Average Earth velocity, m/s
            EARTH_VELOCITY = sqrt(MU_SUN / AU)
      - [ ] DAY2SEC = 86400.0
      - [ ] SEC2DAY = 1.0 / DAY2SEC
      - [ ] YEAR2DAY = 365.25
      - [ ] DAY2YEAR = 1.0 / YEAR2DAY
      - [ ] Earth's Keplerian orbital parameters
            _earth_op = (
            	pk.epoch(54000, 'mjd'),     # t
            	0.999988049532578 * AU,     # a
            	1.67168116316e-2,           # e
            	radians(8.854353079654e-4), # i
            	radians(175.40647696473),   # W
            	radians(287.61577546182),   # w
            	radians(257.60683707535)    # M
            	)
            earth = pk.planet.keplerian(_earth_op[0], _earth_op[1:],
                                        MU_SUN, 398601.19 * 1000**3, 0, 0, 'Earth')
      - [ ] bounds (days) in which to search for a rendezvous leg's dT
            rvleg_dT_bounds = [60., 500.]
    useful:
    - [X] __init__.py (imports constants and gtoc5.py functions)
    - [ ] gtoc5.py (global cache, can look for potential improvements to arp code)
    - [ ] lambert.py (acessing multirev solutions, some optimization)
    - [ ] multiobjective.py (probably not actually useful, but look into pareto front stuff)
    - [ ] phasing.py (as expected, phasing stuff)
    not:
    - [X] ast_ephem.py (same purpose as process_asteroids)
    - [X] ast_ephem.txt (specific to GTOC5)
    - [X] bestiary.py (specific to GTOC5 sequences)
    - [X] mass_optimal_1st_leg.pkl, time_optimal_1st_leg.pkl (specific to GTOC5)
  - [ ] paco
    - [X] __init__.py (imports paco)
    - [ ] paco.py (use all, allows paco and beam_paco runs, pareto_elite not implemented?)
  - [ ] experiments__paco.py (sets up experimental parameters)
  - [ ] experiments.py
  - [ ] paco.traj.py
  - [ ] traj.video.ipynb
  - [ ] usage_demos.ipynb
  
- [ ] install/replace/update required libraries
  - [ ] pykep<2
  - [ ] numpy
  - [ ] scipy
  - [ ] tqdm
  - [ ] pandas
  - [ ] matplotlib
  - [ ] seaborn
  - [ ] watermark
 
## File: space_util
- [ ] check constants and unit conversions (pykep equivalents)
- [ ] OrbitBuilder, combine with process_asteroids 'to_orbit'? 
- [ ] check units!
- [ ] MAX_REVS either set in space_util, or add as function input

## File: arp
- [ ] inner_minimize_multistart(fun, multi, bounds, method = 'SLSQP', constraints = (), **kwargs)
  - [ ] sort multi bounds
        called by AsteroidRoutingProblem.optimize_transfer_orbit
- [ ] inner_minimize(fun, x0, bounds, method = 'SLSQP', constraints = (), **kwargs)
  -[ ] update so results can be plotted?
        called by AsteroidRoutingProblem.optimize_transfer_orbit_total_time
- [ ] AsteroidRoutingProblem(Problem)
  - [ ] read_instance(cls, instance_name) find where this is called
  - [ ] fitness_nosave(self, x)
  
- [ ] combine optimise functions
- [ ] maybe replace nearest neighbor with pk.phasing.knn
  
## File: transfer_example
- [ ] ensure all can be plotted:
  - [ ] arp_instance.CompleteSolution()
  - [ ] arp_instance.evaluate_sequence()
  - [ ] arp_instance.optimize_transfer()
  - [X] arp_instance.build_nearest_neighbor()
  - [ ] arp_instance.evaluate_sequence()
  - [ ] arp_instance.optimize_transfer_total_time()
  - [ ] arp_instance.evaluate_transfer()
