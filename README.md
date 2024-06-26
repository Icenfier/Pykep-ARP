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
- [X] process_asteroids
  - [X] check all files available
  - [X] to_orbit function
  - [X] true_anomaly function (angles)
  - [X] find alternative to Orbit class (planet module in pykep)
  - [X] ensure all units correct
  - [X] ensure start epoch correct (MJD2000)
  - [X] ensure frame of reference is consistant (J2000 heliocentric ecliptic)
- [ ] space_util, MU, MJD
  - [ ] replace 'propagate' from poliastro twobody, propagate.lagrangian OR planet.eph
- [ ] Test with transfer_example
  - [ ] check all files available

## Other objectives
- [ ] Find asteroid r & v using Pykep
- [ ] Plotting using Pykep
- [ ] Investigating using SPICE for asteroids
- [ ] using pykep for Lambert
 
