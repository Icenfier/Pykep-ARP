import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D
#from pykep.orbit_plots import plot_planet, plot_lambert
from better_plots import plot_planet, plot_lambert
from pykep import epoch as def_epoch
from space_util import propagate
from pykep.core import DAY2SEC
from space_util import (
    START_EPOCH,
    Earth,
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


def plot_solution(problem, sequence, sol=False, ax = None, projection='2d'):
    fig = plt.figure(figsize=((10,8)))
    if projection == '3d':
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    if sequence[0] < 0 :
        return ax, np.inf, np.inf, np.inf
    sequence = np.asarray(sequence)
    if not sol:
        print('Solution object not provided. Sequence will be evaluated before plotting.')
        sol = problem.CompleteSolution(sequence)
    f = sol.f
    t = sol.ship.leg_times
    maneuvers = sol.ship.maneuvers
    cost = sol.get_cost()
    tot_time = sol.get_time()
    
    epoch = def_epoch(START_EPOCH.mjd2000 + t[0])
    ship = propagate(Earth, epoch)
    
    if projection == '3d':
        ax.scatter(0,0,0, color='y', s = 70, axes = ax, label = 'Sun')
    else:
        ax.scatter(0,0, color='y', s = 70, axes = ax, label = 'Sun')
    plot_planet(ship, t0 = epoch, N = 200, units = 1000, axes = ax, color = 'black', alpha = 0.5, legend=('Earth',False), projection = projection)
    for k, (ast, man) in enumerate(zip(sequence,maneuvers)):
        to_orbit = problem.get_ast_orbit(ast)
        ship = propagate(ship, epoch)
        plot_lambert(l = man, N = 200, units = 1000, sol = 0, color = f'C{k+1}',legend='False', axes = ax, projection = projection, linestyle = '--') 
        ax.lines[-1].set_label(f'Transfer {k+1}')
        epoch = def_epoch(epoch.mjd2000+man.get_tof()/DAY2SEC)
        ship = switch_orbit(to_orbit, epoch)
        if 2*(k+1) >= len(t):
            tofs = 0
            end_epoch = epoch
        else:
            tofs = t[2*(k+1)]
            end_epoch = (def_epoch(epoch.mjd2000+t[2*(k+1)]))

        plot_planet(ship, t0 = epoch, tf = end_epoch, units = 1000,  color = f'C{k+1}', axes = ax, legend=(f'Asteroid {ast}',False), projection = projection)
        epoch = def_epoch(epoch.mjd2000+tofs)
        ship = propagate(ship, end_epoch)
        ship_r = ship.eph(end_epoch)[0]
        if projection == '3d':
            ax.scatter(ship_r[0]/1000, ship_r[1]/1000, ship_r[2]/1000, color = f'C{k+1}', s = 40, axes=ax) #/1000 to convert to km
        else:
            ax.scatter(ship_r[0]/1000, ship_r[1]/1000, color = f'C{k+1}', s = 40, axes=ax) #/1000 to convert to km
        plt.draw()
        
    if projection == '3d':
        xlim = max(abs(a) for a in ax.get_xlim3d())
        ylim = max(abs(a) for a in ax.get_ylim3d())
        maxlim = max(xlim, ylim) * 1.02
        ax.set_xlim3d([-maxlim, maxlim])
        ax.set_ylim3d([-maxlim, maxlim])
        ax.view_init(90, -90) 
    else:
        xlim = max(abs(a) for a in ax.get_xlim())
        ylim = max(abs(a) for a in ax.get_ylim())
        maxlim = max(xlim, ylim) * 1.02
        ax.set_xlim([-maxlim, maxlim])
        ax.set_ylim([-maxlim, maxlim])
        ax.set_aspect('equal')
    
    plt.legend(ncol = (problem.n//15)+1, bbox_to_anchor=(1,1))
    fig.suptitle(f'$\Delta v$={cost:.1f} km/s, $T$={tot_time:.1f} days, $f$={f:.1f}', x=.58, y=0.94)
    plt.show()

    return ax, f, cost, tot_time

def plot_solution_to_pdf(instance, sol, pdf_file, title = None, figsize = "lncs"):
    fig, ax = plt.subplots(figsize=get_fig_size(figsize, fraction=1))
    ax, _, f, cost, time = plot_solution(instance, sol, ax = ax)
    if title is not None:
        fig.suptitle(title + f'$\Delta v$={cost:.1f} km/s, $T$={time:.1f} days, $f$={f:.1f}', x=.58, y=0.94)
    fig.savefig(pdf_file, bbox_inches="tight")

