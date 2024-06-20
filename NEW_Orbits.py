# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:37:16 2024
Orbit creation file
Recreation of poliastro creation.py, scalar.py, enums.py, states.py
planes from enums, creationmixin from creation, orbits from scalar, classicalstate from states
May separate/combine depending on how busy/empty this file gets.


@author: ezra
"""
from warnings import warn
from functools import cached_property

import numpy as np
from astropy import units as u
import pykep as pk
from pykep.core import epoch as def_epoch

#planes
from enum import Enum

class Planes(Enum):
    EARTH_EQUATOR = "Earth mean Equator and Equinox of epoch (J2000.0)"
    EARTH_ECLIPTIC = "Earth mean Ecliptic and Equinox of epoch (J2000.0)"
    BODY_FIXED = "Rotating body mean Equator and node of date"
    

class ClassicalState:
    
    def __init__(self, attractor, elements, plane):
        """Constructor.

        Parameters
        ----------
        # TODO replace this
        attractor : Body
            Main attractor.
        elements : tuple
            Six-tuple of orbital elements for this state.
        plane : ~poliastro.frames.enums.Planes
            Reference plane for the elements.

        """
        self._attractor = attractor
        self._elements = elements
        self._plane = plane
        
    """State defined by its classical orbital elements.

    plane : reference plane
    attractor : main attractor
    
    
    Orbital elements:

    p : ~astropy.units.Quantity
            Semilatus rectum.
    a : ~astropy.units.Quantity
            Semimajor axis.
    ecc : ~astropy.units.Quantity
            Eccentricity.
    inc : ~astropy.units.Quantity
            Inclination.
    raan : ~astropy.units.Quantity
            Right ascension of the ascending node.
    argp : ~astropy.units.Quantity
            Argument of the perigee.
    nu : ~astropy.units.Quantity
            True anomaly.

    """

    @property
    def plane(self):
        """Fundamental plane of the frame."""
        return self._plane

    @property
    def attractor(self):
        """Main attractor."""
        return self._attractor

    @property
    def p(self):
        """Semilatus rectum."""
        return self._elements[0]

    @property
    def a(self):
        """Semimajor axis."""
        return self.p / (1 - self.ecc**2)

    @property
    def ecc(self):
        """Eccentricity."""
        return self._elements[1]

    @property
    def inc(self):
        """Inclination."""
        return self._elements[2]

    @property
    def raan(self):
        """Right ascension of the ascending node."""
        return self._elements[3]

    @property
    def argp(self):
        """Argument of the perigee."""
        return self._elements[4]

    @property
    def nu(self):
        """True anomaly."""
        return self._elements[5]

    def to_value(self):
        return (
            self.p.to_value(u.km),
            self.ecc.value,
            self.inc.to_value(u.rad),
            self.raan.to_value(u.rad),
            self.argp.to_value(u.rad),
            self.nu.to_value(u.rad),
        )

    def to_vectors(self):
        """Converts to position and velocity vector representation."""
        r, v = pk.par2ic(
            self.to_value(), pk.MU_SUN
        )

        return r, v

    def to_classical(self):
        """Converts to classical orbital elements representation."""
        return self

#    def to_equinoctial(self):
#        """Converts to modified equinoctial elements representation."""
#        p, f, g, h, k, L = coe2mee(*self.to_value())
#        return ModifiedEquinoctialState(
#            self.attractor,
#            (
#                p << u.km,
#                f << u.rad,
#                g << u.rad,
#                h << u.rad,
#                k << u.rad,
#                L << u.rad,
#            ),
#            self.plane,
#        )

class OrbitCreationMixin:
    """
    Mixin-class containing class-methods to create Orbit objects
    """
    
    def __init__(self, *_, **__):  # HACK stub to make mypy happy
        ...                        # dunno what this does, feels like I need an _init_
        
    @classmethod
    @u.quantity_input(
        a=u.m, ecc=u.one, inc=u.rad, raan=u.rad, argp=u.rad, nu=u.rad
    )
    def from_classical(
        cls,
        attractor,
        a,
        ecc,
        inc,
        raan,
        argp,
        nu,
        epoch = def_epoch(0,"mjd2000"),
        plane=Planes.EARTH_EQUATOR,
    ):
        """Return `Orbit` from classical orbital elements.

        Parameters
        ----------
        attractor : Body
            Main attractor.
        a : ~astropy.units.Quantity
            Semi-major axis.
        ecc : ~astropy.units.Quantity
            Eccentricity.
        inc : ~astropy.units.Quantity
            Inclination
        raan : ~astropy.units.Quantity
            Right ascension of the ascending node.
        argp : ~astropy.units.Quantity
            Argument of the pericenter.
        nu : ~astropy.units.Quantity
            True anomaly.
        epoch : ~pykep.core.epoch, optional
            Epoch, default to J2000.
        # TODO change this
        plane : ~poliastro.frames.Planes
            Fundamental plane of the frame.

        """
        for element in a, ecc, inc, raan, argp, nu, epoch:
            if not element.isscalar:
                raise ValueError(f"Elements must be scalar, got {element}")

        if ecc == 1.0 * u.one:
            raise ValueError(
                "Doesn't support parabolic orbits"
            )

        if not 0 * u.deg <= inc <= 180 * u.deg:
            raise ValueError("Inclination must be between 0 and 180 degrees")

        if ecc > 1 and a > 0:
            raise ValueError("Hyperbolic orbits have negative semimajor axis")

        if not -np.pi * u.rad <= nu < np.pi * u.rad:
            warn("Wrapping true anomaly to -π <= nu < π", stacklevel=2)
            nu = (
                (nu + np.pi * u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad
            ).to(nu.unit)
            

        ss = ClassicalState(
            attractor, (a * (1 - ecc**2), ecc, inc, raan, argp, nu), plane
        )
        return cls(ss, epoch)
    
    ##############################################################################

class Orbit(OrbitCreationMixin):
    """Position and velocity of a body with respect to an attractor
    at a given time (epoch).

    Regardless of how the Orbit is created, the implicit
    reference system is an inertial one. For the specific case
    of the Solar System, this can be assumed to be the
    International Celestial Reference System or ICRS.

    """
    
    def __init__(self, state, epoch):
        """Constructor.

        Parameters
        ----------
        state : BaseState
            Position and velocity or orbital elements.
        epoch : ~astropy.time.Time
            Epoch of the orbit.

        """
        self._state = state  # type: BaseState
        self._epoch = epoch  # type: time.Time
        
    @property
    def attractor(self):
        """Main attractor."""
        return self._state.attractor

    @property
    def epoch(self):
        """Epoch of the orbit."""
        return self._epoch

    @property
    def plane(self):
        """Fundamental plane of the frame."""
        return self._state.plane


###############################################################################
class OrbitCreationMixin:
    """
    Mixin-class containing class-methods to create Orbit objects
    """
    
    def __init__(self, *_, **__):  # HACK stub to make mypy happy
        ...                        # dunno what this does, feels like I need an _init_
        
    @classmethod
    @u.quantity_input(
        a=u.m, ecc=u.one, inc=u.rad, raan=u.rad, argp=u.rad, nu=u.rad
    )
    def from_classical(
        cls,
        attractor,
        a,
        ecc,
        inc,
        raan,
        argp,
        nu,
        epoch = def_epoch(0,"mjd2000"),
        plane=Planes.EARTH_EQUATOR,
    ):
        """Return `Orbit` from classical orbital elements.

        Parameters
        ----------
        attractor : Body
            Main attractor.
        a : ~astropy.units.Quantity
            Semi-major axis.
        ecc : ~astropy.units.Quantity
            Eccentricity.
        inc : ~astropy.units.Quantity
            Inclination
        raan : ~astropy.units.Quantity
            Right ascension of the ascending node.
        argp : ~astropy.units.Quantity
            Argument of the pericenter.
        nu : ~astropy.units.Quantity
            True anomaly.
        epoch : ~astropy.time.Time, optional
            Epoch, default to J2000.
        plane : ~poliastro.frames.Planes
            Fundamental plane of the frame.

        """
        for element in a, ecc, inc, raan, argp, nu, epoch:
            if not element.isscalar:
                raise ValueError(f"Elements must be scalar, got {element}")

        if ecc == 1.0 * u.one:
            raise ValueError(
                "For parabolic orbits use Orbit.parabolic instead"
            )

        if not 0 * u.deg <= inc <= 180 * u.deg:
            raise ValueError("Inclination must be between 0 and 180 degrees")

        if ecc > 1 and a > 0:
            raise ValueError("Hyperbolic orbits have negative semimajor axis")

        if not -np.pi * u.rad <= nu < np.pi * u.rad:
            warn("Wrapping true anomaly to -π <= nu < π", stacklevel=2)
            nu = (
                (nu + np.pi * u.rad) % (2 * np.pi * u.rad) - np.pi * u.rad
            ).to(nu.unit)

        ss = ClassicalState(
            attractor, (a * (1 - ecc**2), ecc, inc, raan, argp, nu), plane
        )
        return cls(ss, epoch)