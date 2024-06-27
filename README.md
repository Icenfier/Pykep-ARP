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

## Other objectives
- [ ] Investigating using SPICE for asteroids
- [ ] using pykep for multirev Lambert
 
## Current file: space_util
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
    - [ ] circular
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