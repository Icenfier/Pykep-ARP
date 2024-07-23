# 
# Copyright (c) 2017 Luis F. Simoes (github: @lfsimoes)
# 
# Licensed under the MIT License. See the LICENSE file for details.

from math import sqrt, radians
import PyKEP as pk

# ==================================== # Problem constants
# Source: GTOC5 problem statement
# http://dx.doi.org/10.2420/AF08.2014.9

# Sun's gravitational parameter µ_S, m^3/s^2
MU_SUN = pk.MU_SUN

# Astronomical Unit AU, m
AU = pk.AU

# Average Earth velocity, m/s
EARTH_VELOCITY = sqrt(MU_SUN / AU)

# ==================================== # GTOC5 asteroids (body objects)

# Earth's Keplerian orbital parameters
# Source: http://dx.doi.org/10.2420/AF08.2014.9 (Table 1)
_earth_op = (
	pk.epoch(54000, 'mjd'),     # t
	0.999988049532578 * AU,     # a
	1.67168116316e-2,           # e
#	radians(9.954353079654e-4), # i (value in the original problem statement)
	radians(8.854353079654e-4), # i
	radians(175.40647696473),   # W
	radians(287.61577546182),   # w
	radians(257.60683707535)    # M
	)
earth = pk.planet.keplerian(_earth_op[0], _earth_op[1:],
                            MU_SUN, 398601.19 * 1000**3, 0, 0, 'Earth')

# ==================================== #
