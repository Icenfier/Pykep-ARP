import numpy as np
import pykep as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler
from pykep import epoch as def_epoch
from space_util import propagate, Asteroids
from pykep.core import DAY2SEC

#from poliastro.plotting import StaticOrbitPlotter
#from poliastro.plotting.util import generate_label
#from poliastro.twobody.propagation import propagate
from astropy import units as u
from astropy.time import TimeDelta
#from arp import AsteroidRoutingProblem
from space_util import (
    START_EPOCH,
    Earth,
    perform_lambert,
    switch_orbit
)

# From https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def get_fig_size(width, fraction=1, subplots=(1, 1), ratio = (5**.5 - 1) / 2):
    """Get figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    ratio: height = ratio * width, optional.
           By default, the golden ratio. https://disq.us/p/2940ij3
           
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'lncs':
        width_pt = 347.12354
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def plot_solution(self, x, ax = None):
    x = np.asarray(x)
    sol = self.CompleteSolution(x)
    t = sol.ship.x
    #print(ast_orbits[0])
    #print(t)
    #frame = StaticOrbitPlotter(ax = ax, plane=Earth.plane)
    epoch = START_EPOCH
    ship = propagate(Earth, epoch)
    
    fig = plt.figure(figsize=((10,8)))
    ax = fig.add_subplot(projection='3d')
    plot_planet(ship, t0 = epoch, N = 60, units = 1.0, legend="Earth", axes = ax)
    print(t)
    for k, (ast, man) in enumerate(zip(x,sol.ship.maneuvers)):
        #print(t)
        #print(k)
        #print(ast)
        #print(man)
        #print('pre-manuver: ', epoch)
        #epoch = def_epoch(epoch.mjd2000 + t[2*k])
        #print('mid maneuver: ', epoch)
        to_orbit = self.get_ast_orbit(ast)
        ship = propagate(ship, epoch)
        plot_lambert(l = man, N = 100, sol = 0, color = f'C{k+1}', axes = ax) #legend=f'Impulse {k}', axes = ax) #label = generate_label(epoch, f'Impulse {k}'))
        ship, r, v, epoch = perform_lambert(5e-6, man, epoch) # 5e-6 is mu, assuming ship mass ~ 100000kg
        ship = switch_orbit(to_orbit, epoch)
        #epoch = def_epoch(epoch.mjd2000 + t[2*k + 1])
        #print('post-manuver: ', epoch)
        if 2*(k+1) >= len(t):
            tofs = 0
            end_epoch = epoch
        else:
            #tofs = np.linspace(0,  t[2*(k+1)], num=100)
            tofs = t[2*(k+1)]
            end_epoch = (def_epoch(epoch.mjd2000+t[2*(k+1)]))

        plot_planet(ship, t0 = epoch, tf = end_epoch, color = f'C{k+1}', axes=ax)#legend = True, axes = ax)#label=generate_label(epoch, f'Asteroid {ast}'))
        #plot_kepler(r0=r, v0=v, tof = tofs*DAY2SEC, mu = pk.MU_SUN, color = f'C{k+1}', axes = ax) 
        epoch = def_epoch(epoch.mjd2000+tofs)
        ship = propagate(ship, epoch)#TODO sort all tis out
        #print('post-asteroid: ', epoch)
        
        ax.view_init(90, -90) 
        plt.draw()

    return ax, frame, sol.f, sol.get_cost(), sol.get_time()

def plot_solution_to_pdf(instance, sol, pdf_file, title = None, figsize = "lncs"):
    fig, ax = plt.subplots(figsize=get_fig_size(figsize, fraction=1))
    ax, _, f, cost, time = plot_solution(instance, sol, ax = ax)
    if title is not None:
        fig.suptitle(title + f'$\Delta v$={cost:.1f} km/s, $T$={time:.1f} days, $f$={f:.1f}', x=.58, y=0.94)
    fig.savefig(pdf_file, bbox_inches="tight")

# instance = AsteroidRoutingProblem(10, 42)
# x,f = instance.nearest_neighbor([], "euclidean")
# print(x)
# print(f)

# plot_solution(instance, x)
