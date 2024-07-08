# Daily Log

## Week 1, 17/06
### Mon, 4 hrs
- PM: Pre-reading, looking at libraries

### Tues, 6 hrs
- AM: Downloading and trying to run parts of code, noting poliastro and pykep functions
- PM: Meeting, setting up github account

### Wed, 7 hrs
- AM: Attepting to update anaconda, reinstalling multiple times
- PM: Copying code to github, more issues with astroquery (had to downloaded older version of python).
-     Eventually managed to copy over bits of code and ensured I had the correct libraries to run it.

### Thurs, 7 hrs
- AM: Investigating poliastro's Orbits class, planning my own version of it to relace 
- PM: Started combining bits of poliastro's Orbits, Planes, and States classes

### Fri, 2.5 hrs
- PM: Replacing angle conversions


## Week 2, 24/06
### Mon, 6 hrs
- AM: attempting to use planet module to replace orbit class, ran into issue with planet.keplerian
- PM: more attempts to fix .keplerian, keep getting error "python argument types did not match C++ signature"
-     changed a few epochs and angles

### Tues, 6 hrs
- AM: fixed .keplerian issue, the problem was with astropy units, switched to using scipy
- PM: finished updating process_asteroids, now fully independent of poliastro
-     looked into replacing orbit propagation

### Wed, 8 hrs
- AM: replacing Earth object in space_util. Ran into issues when running, managed to fix (was expecting cls input)
- PM: completely replaced main space_util code (not all callable funcitions though)
-     Asteroids class now works with Planet objects, including changing epoch times
-     started looking at arp.py

### Thurs, 7 hrs
- AM: figuring out which functions can be replaced by pykep functions, and which just need to be amended
-     noting where functions are called
- PM: working on space_util functions, meeting

### Fri, 5.5 hrs
- PM: looking through arp.py, sorting two_shot_tranfer lambert solutions


## Week 3, 01/07
### Mon, 6 hrs
-     more two_shot_transfer stuff, issue with replacing 'Maneuver' class with pykep .lambert_solution

### Tues, 5 hrs
- AM: fixing two_shot_transfer
- PM: meeting, uni sent me mandatory employee training

### Wed, 7 hrs
- AM: trying to figure out how to get multirev solutions, it doesnt seem to give me any. ignoring for now (since realised issue, solved on thurs)
- PM: updating all code called by transfer_example, transfer evaluation stuff.

### Thurs, 8 hrs
-     attempting plotting stuff. took embarassingly long to figure out that plot_lambert takes time of flight in seconds, not days

### Fri, 5 hrs
- AM: more plotting. got maneuvers to work, but not the waiting on asteroids.