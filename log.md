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



## Week 4, 08/07
### Mon, 7 hrs
- AM: checking transfer calculations, some minor changes. looking more at plotting
- PM: meeting, creating graph outside of program to ensure I understand how the graphing functions work

### Tues, 7 hrs
- AM: checking all epochs to fix graph arcs not lining up
- PM: calculating costs

### Wed, 6 hrs
-     trying to figure out why im getting bad results, mapping which functions lead where in arp.py

### Thurs, 8 hrs
- AM: fixing unit conversions, therefore fixing cost calculations and, by extension, results
- PM: testing all functions in transfer_example

### Fri, 5 hrs
- AM: sorting out and updating markdown files
- PM: a few minor changes in arp and space_util



## Week 5, 15/07
### Mon, 7 hrs
- AM: combining and deleting unnecesary functions
- PM: formatting plot

### Tues, 6 hrs
- AM: creating option for nearest neighbour by energy
- PM: adding to plot_solution to allow input of previously calculated solutions
-     changing a bunch of stuff in optimization functions so that maneuvers can be returned
-     plot_solution can now either plot pre-existing solutions, or calculate solutions itself given an asteroid sequence 

### Wed, 6.5 hrs
- AM: reading paper on beam search and p-aco https://arxiv.org/pdf/1704.00702
- PM: researching beam searches and ant colony optimisation

### Thurs, 8 hrs
- AM: realising that cego.py, and most likely umm.py, calls functions that I have altered/removed. added fixing this to todo list
- PM: downloading and looking at beam paco repository, working out which functions will be useful and which are too specific
-     https://github.com/lfsimoes/beam_paco__gtoc5

### Fri, 7 hrs
- AM: using PACO code to fix a few things in my own code
- PM: testing PACO code to see how different functions work



## Week 6, 22/07
### Mon, 8.5 hrs
-     reorganising arp.py, using ship object for plotting without the need for re-evaluation

### Tues, 6 hrs
- AM: finishing fixing arp.py
- PM: writing code to read in preexisting results before performing the simulation itself, allowing a comparison

### Wed, 6 hrs
- AM: compared pykep and poliastro results, euclidean matches pretty much perfectly, energy calculated differently
- PM: fixed delta v calculations

### Thurs, 7 hrs
- AM: trying to install and use rpy2 to test cego.py, wouldn't install properly
- PM: fixing cego so arp instances are called properly

### Fri, 5hrs
-     fixing umm.py and random_search.py so arp instances are called properly



## Week 7, 29/07
### Mon, 8 hrs
- AM: fixed incorrect epoch that was causing a difference in final cost and cost used for f calculations
- PM: wrote function to compare poliastro and pykep random_search results

### Tues, 8 hrs
-     added much better, customisable plotting, finalised legend

### Wed, 7 hrs
-     allowed for either 2d or 3d projection when plotting

### Thurs, 7 hrs
- AM: overwrote knn.py _init_ so that code actually runs
- PM: implemented knn phasing into greedynn 

### Fri, 5 hrs
-     started looking into implementing beam search



## Week 8, 05/08
### Mon-Fri, 7 hrs per day
-     implementing beam search 
-     implementing p-aco
-     writing documentaion within code

