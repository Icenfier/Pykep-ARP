def plot_planet(plnt, t0=0, tf=None, N=60, units=1.0, color='k', alpha=1.0, s=40, legend=(False, False), axes=None, projection = '3d', **kwargs):
    """
    ax = plot_planet(plnt, t0=0, tf=None, N=60, units=1.0, color='k', alpha=1.0, s=40, legend=(False, False), axes=None):

    - axes:      3D axis object created using fig.add_subplot(projection='3d')
    - plnt:      pykep.planet object we want to plot
    - t0:        a pykep.epoch or float (mjd2000) indicating the first date we want to plot the planet position
    - tf:        a pykep.epoch or float (mjd2000) indicating the final date we want to plot the planet position.
                 if None this is computed automatically from the orbital period (prone to error for non periodic orbits)
    - units:     the length unit to be used in the plot
    - color:     color to use to plot the orbit (passed to matplotlib)
    - s:         planet size (passed to matplotlib)
    - legend     2-D tuple of bool or string: The first element activates the planet scatter plot, 
                 the second to the actual orbit. If a bool value is used, then an automated legend label is generated (if True), if a string is used, the string is the legend. Its also possible but deprecated to use a single bool value. In which case that value is used for both the tuple components.

    Plots the planet position and its orbit.

    Example::

	import pykep as pk
	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	pl = pk.planet.jpl_lp('earth')
	t_plot = pk.epoch(219)
	ax = pk.orbit_plots.plot_planet(pl, ax = ax, color='b')
    """
    from pykep import MU_SUN, SEC2DAY, epoch, AU, RAD2DEG
    from pykep.planet import keplerian
    from math import pi, sqrt
    import numpy as np
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D

    if axes is None:
        fig = plt.figure()
        if projection == '3d':
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
    else:
        ax = axes

    if type(t0) is not epoch:
        t0 = epoch(t0)

    # This is to make the tuple API compatible with the old API
    if type(legend) is bool:
        legend = (legend, legend)

    if tf is None:
        # orbit period at epoch
        T = plnt.compute_period(t0) * SEC2DAY
    else:
        if type(tf) is not epoch:
            tf = epoch(tf)
        T = (tf.mjd2000 - t0.mjd2000)
        if T < 0:
            raise ValueError("tf should be after t0 when plotting an orbit")

       # points where the orbit will be plotted
    when = np.linspace(0, T, N)

    # Ephemerides Calculation for the given planet
    x = np.array([0.0] * N)
    y = np.array([0.0] * N)
    z = np.array([0.0] * N)

    for i, day in enumerate(when):
        r, v = plnt.eph(epoch(t0.mjd2000 + day))
        x[i] = r[0] / units
        y[i] = r[1] / units
        z[i] = r[2] / units

    # Actual plot commands
    if (legend[0] is True):
        label1 = plnt.name + " " + t0.__repr__()[0:11]
    elif (legend[0] is False):
        label1 = None
    elif (legend[0] is None):
        label1 = None
    else:
        label1 = legend[0]

    if (legend[1] is True):
        label2 = plnt.name + " orbit"
    elif (legend[1] is False):
        label2 = None
    elif (legend[1] is None):
        label2 = None
    else:
        label2 = legend[1]

    if projection == '3d':
        ax.plot(x, y, z, label=label2, c=color, alpha=alpha)
        ax.scatter([x[0]], [y[0]], [z[0]], s=s, marker='o', alpha=alpha, c=[color], label=label1, **kwargs)
    else:
        ax.plot(x, y, label=label2, c=color, alpha=alpha)
        ax.scatter([x[0]], [y[0]], s=s, marker='o', alpha=alpha, c=[color], label=label1, **kwargs)

    if legend[0] or legend[1]:
        ax.legend()
    return ax


def plot_lambert(l, N=60, sol=0, units=1.0, color='b', legend=False, axes=None, alpha=1., projection = '3d', **kwargs):
    """
    ax = plot_lambert(l, N=60, sol=0, units='pykep.AU', legend='False', axes=None, alpha=1.)

    - axes:     3D axis object created using fig.add_subplot(projection='3d')
    - l:        pykep.lambert_problem object
    - N:        number of points to be plotted along one arc
    - sol:      solution to the Lambert's problem we want to plot (must be in 0..Nmax*2)
                where Nmax is the maximum number of revolutions for which there exist a solution.
    - units:    the length unit to be used in the plot
    - color:    matplotlib color to use to plot the line
    - legend:   when True it plots also the legend with info on the Lambert's solution chosen

    Plots a particular solution to a Lambert's problem

    Example::

      import pykep as pk
      import matplotlib.pyplot as plt

      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')

      t1 = pk.epoch(0)
      t2 = pk.epoch(640)
      dt = (t2.mjd2000 - t1.mjd2000) * pk.DAY2SEC

      pl = pk.planet.jpl_lp('earth')
      pk.orbit_plots.plot_planet(pl, t0=t1, axes=ax, color='k')
      rE,vE = pl.eph(t1)

      pl = pk.planet.jpl_lp('mars')
      pk.orbit_plots.plot_planet(pl, t0=t2, axes=ax, color='r')
      rM, vM = pl.eph(t2)

      l = lambert_problem(rE,rM,dt,pk.MU_SUN)
      pk.orbit_plots.plot_lambert(l, ax=ax, color='b')
      pk.orbit_plots.plot_lambert(l, sol=1, axes=ax, color='g')
      pk.orbit_plots.plot_lambert(l, sol=2, axes=ax, color='g', legend = True)

      plt.show()
    """
    from pykep import propagate_lagrangian, AU
    import numpy as np
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D

    if axes is None:
        fig = plt.figure()
        if projection == '3d':
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
    else:
        ax = axes

    if sol > l.get_Nmax() * 2:
        raise ValueError("sol must be in 0 .. NMax*2 \n * Nmax is the maximum number of revolutions for which there exist a solution to the Lambert's problem \n * You can compute Nmax calling the get_Nmax() method of the lambert_problem object")

    # We extract the relevant information from the Lambert's problem
    r = l.get_r1()
    v = l.get_v1()[sol]
    T = l.get_tof()
    mu = l.get_mu()

    # We define the integration time ...
    dt = T / (N - 1)

    # ... and allocate the cartesian components for r
    x = np.array([0.0] * N)
    y = np.array([0.0] * N)
    z = np.array([0.0] * N)

    # We calculate the spacecraft position at each dt
    for i in range(N):
        x[i] = r[0] / units
        y[i] = r[1] / units
        z[i] = r[2] / units
        r, v = propagate_lagrangian(r, v, dt, mu)

    # And we plot
    if legend:
        label = 'Lambert solution (' + str((sol + 1) // 2) + ' revs.)'
    else:
        label = None
    if projection == '3d':
        ax.plot(x, y, z, c=color, label=label, alpha=alpha, **kwargs)
    else:
        ax.plot(x, y, c=color, label=label, alpha=alpha, **kwargs)

    if legend:
        ax.legend()

    return ax