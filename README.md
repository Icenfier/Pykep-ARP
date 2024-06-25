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
- [ ] process_asteroids
  - [X] check all files available
  - [ ] to_orbit function
  - [X] true_anomaly function (angles)
  - [ ] find alternative to Orbit class (planet module in pykep)
      - [ ] sort planes subclass
      - [ ] sort creation subclass
      - [ ] sort state subclass
      - [ ] test all
  - [ ] ensure all units correct
  - [ ] ensure start epoch correct
  - [ ] ensure frame of reference is consistant (Ecliptic)
- [ ] Replacing Body class with objects that use SPICE kernels?
- [ ] space_util, MU, MJD
- [ ] Test with transfer_example
  - [ ] check all files available

## Other objectives
- [ ] Find asteroid r & v using Pykep
- [ ] Plotting using Pykep
- [ ] Investigating using SPICE for asteroids
- [ ] using pykep for Lambert
 
