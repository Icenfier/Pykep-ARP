import numpy as np
import pykep as pk
from pykep import DAY2SEC
from pandas import read_csv as pd_read_csv
from scipy.constants import G
from pykep.core import epoch as def_epoch
from numpy import deg2rad
from pykep.planet import _base
from poliastro.maneuver import Maneuver
from arp import AsteroidRoutingProblem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler
from arp import VisitProblem
from scipy.spatial import distance

from space_util import (
    Earth,
    EARTH_START_EPOCH,
    propagate,
    OrbitBuilder,
    two_shot_transfer,
    switch_orbit)

START_EPOCH = def_epoch(95739,"mjd2000")
MAX_REVS = 0

n = 5
size = n
seed = 47
ast_id = 1

arp_instance = AsteroidRoutingProblem(10, 73)
