# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 04:38:23 2019

@author: slauniai
"""

import numpy as np
import matplotlib.pyplot as plt

#: machine epsilon
EPS = np.finfo(float).eps

#: [J mol\ :sup:`-1`\ ], latent heat of vaporization at 20\ :math:`^{\circ}`\ C
LATENT_HEAT = 44100.0
#: [kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O
MOLAR_MASS_H2O = 18.015e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of CO\ :sub:`2`\
MOLAR_MASS_CO2 = 44.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of C
MOLAR_MASS_C = 12.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of air
MOLAR_MASS_AIR = 29.0e-3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of H\ :sub:`2`\ O
SPECIFIC_HEAT_H2O = 4.18e3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of organic matter
SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
#: [J mol\ :sup:`-1` K\ :sup:`-1`\ ], molar heat capacity of air at constant pressure
SPECIFIC_HEAT_AIR = 29.3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], mass-based heat capacity of air at constant pressure
SPECIFIC_HEAR_AIR_MASS = 1004.67  
#: [W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8
#: [-], von Karman constant
VON_KARMAN = 0.41
#: [K], zero degrees celsius in Kelvin
DEG_TO_KELVIN = 273.15
#: [K], zero degrees celsius in Kelvin
NORMAL_TEMPERATURE = 273.15
#: [mol m\ :sup:`-3`\ ], density of air at 20\ :math:`^{\circ}`\ C
AIR_DENSITY = 41.6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], kinematic viscosity of air at 20\ :math:`^{\circ}`\ C
AIR_VISCOSITY = 15.1e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], thermal diffusivity of air at 20\ :math:`^{\circ}`\ C
THERMAL_DIFFUSIVITY_AIR = 21.4e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of CO\ :sub:`2` at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of H\ :sub:`2`\ at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
#: [J mol\ :sup:`-1` K\ :sup:``-1], universal gas constant
GAS_CONSTANT = 8.314
#: [kg m\ :sup:`2` s\ :sup:`-1`\ ], standard gravity
GRAVITY = 9.81
#: [kg m\ :sup:`-3`\ ], water density
WATER_DENSITY = 1.0e3
#: [umol m\ :sup:`2` s\ :sup:`-1`\ ], conversion from watts to micromol
PAR_TO_UMOL = 4.56
#: [rad], conversion from deg to rad
DEG_TO_RAD = 3.14159 / 180.0
#: [umol m\ :sup:`-1`], O2 concentration in air
O2_IN_AIR = 2.10e5
# [-], von Karman constant
VON_KARMAN = 0.41
#: [J kg\ :sup:`-1`\ ], latent heat of freezing
LATENT_HEAT_FREEZING = 333700.0
#: [\ :math:`^{\circ}`\ C], freezing point of water
FREEZING_POINT_H2O = 0.0
#: [kg m\ :sup:`-3`\ ], densities
ICE_DENSITY = 917.0

#: [J m\ :sup:`-3`\ K \ :sup:`-1`\], thermal condutivities
K_WATER = 0.57
K_ICE = 2.2
K_AIR = 0.025
K_ORG = 0.25

#: volumetric heat capacieties  [J m\ :sup:`-3`\ K \ :sup:`-1`\]
CV_AIR = 1297.0  # air at 101kPa
CV_WATER = 4.18e6  # water
CV_ICE = 1.93e6  # ice
CV_ORGANIC = 2.50e6  # dry organic matter
#CV_MINERAL = 2.31e6  # soil minerals


def surface_atm_conductance(zref, ustar=None, U=None, dT=0.0, zom=0.01, b=1.1e-3):
    """
    Soil surface - atmosphere transfer conductance for scalars. Two parallel
    mechanisms: forced and free convection
    Args:
        zref - reference height (m)
        ustar - friction velocity (m/s) at log-regime. if ustar not given,
                it is computed from Uo, zref and zom        
        ustar- - wind speed (m/s) at zref
        height - reference height (m). Log-profile assumed below zref.
        zom - roughness height for momentum (m), ~0.1 x canopy height
        ustar - friction velocity (m/s) at log-regime. if ustar not given,
                it is computed from Uo, zref and zom
        b - parameter for free convection. b=1.1e-3 ... 3.3e-3 from smooth...rough surface
    Returns:
        conductances for CO2, H2O and heat (mol m-2 s-1), dict
    References:
        Schuepp and White, 1975:Transfer Processes in Vegetation by Electrochemical Analog,
        Boundary-Layer Meteorol. 8, 335-358.
        Schuepp (1977): Turbulent transfer at the ground: on verification of
        a simple predictive model. Boundary-Layer Meteorol., 171-186
        Kondo & Ishida, 1997: Sensible Heat Flux from the Earth’s Surface under
        Natural Convective Conditions. J. Atm. Sci.
    
    """
    
    Sc_v = AIR_VISCOSITY / MOLECULAR_DIFFUSIVITY_H2O  
    Sc_c = AIR_VISCOSITY / MOLECULAR_DIFFUSIVITY_CO2
    Pr = AIR_VISCOSITY / THERMAL_DIFFUSIVITY_AIR 
  
    d = 0.0 # displacement height (m), neglect 
    
    if ustar == None:
        ustar = U * VON_KARMAN / np.log((zref - d) / zom)

    delta = AIR_VISCOSITY / (VON_KARMAN * ustar)
    
    gb_h = (VON_KARMAN * ustar) / (Pr - np.log(delta / zref))
    gb_v = (VON_KARMAN * ustar) / (Sc_v - np.log(delta / zref))
    gb_c = (VON_KARMAN*ustar) / (Sc_c - np.log(delta / zref))
    
    # free convection as parallel pathway, based on Condo and Ishida, 1997.
    #b = 1.1e-3 #ms-1K-1 b=1.1e-3 for smooth, 3.3e-3 for rough surface
    dT = np.maximum(dT, 0.0)
    
    gf_h = b * dT**0.33  # ms-1

    # mol m-2 s-1    
    gb_h = (gb_h + gf_h) * AIR_DENSITY
    gb_v = (gb_v + Sc_v / Pr * gf_h) * AIR_DENSITY
    gb_c = (gb_c + Sc_c / Pr * gf_h) * AIR_DENSITY
    
#    plt.figure()
#    plt.plot(friction_velocity, gb_v, '-')
    return {'co2': gb_c, 'h2o': gb_v, 'heat': gb_h}


""" --- hydraulic properties --- """

def water_retention(pF, theta=None, psi=None):
    """
    Water retention curve vanGenuchten - Mualem
    Args:
        pF - parameter dict
        theta - vol. water content (m3m-3)
        psi - matrix potential (m), <=0
    Returns:
        theta or psi
    """

    Ts = np.array(pF['ThetaS'], ndmin=1)
    Tr = np.array(pF['ThetaR'], ndmin=1)
    alfa = np.array(pF['alpha'], ndmin=1)
    n = np.array(pF['n'], ndmin=1)
    m = 1.0 - np.divide(1.0, n)

    def theta_psi(x):
        # converts water content (m3m-3) to potential (m)
        x = np.minimum(x, Ts)
        x = np.maximum(x, Tr)  # checks limits
        s = (Ts - Tr) / ((x - Tr) + EPS)
        Psi = -1e-2 / alfa*(s**(1.0 / m) - 1.0)**(1.0 / n)  # m
        Psi[np.isnan(Psi)] = 0.0
        return Psi

    def psi_theta(x):
        # converts water potential (m) to water content (m3m-3)
        x = 100*np.minimum(x, 0)  # cm
        Th = Tr + (Ts - Tr) / (1 + abs(alfa*x)**n)**m
        return Th

    if theta:
        return theta_psi(theta)
    if psi:
        return psi_theta(psi)
    
def hydraulic_conductivity(pF, x, Ksat=1):
    r""" Unsaturated liquid-phase hydraulic conductivity following 
    vanGenuchten-Mualem -model.

    Args:
        pF (dict):
            'ThetaS' (float/array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'ThetaR' (float/array): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (float/array): air entry suction [cm\ :sup:`-1`]
            'n' (float/array): pore size distribution [-]
        h (float/array): pressure head [m]
        Ksat (float or array): saturated hydraulic conductivity [units]
    Returns:
        Kh (float or array): hydraulic conductivity (if Ksat ~=1 then in [units], else relative [-])

    Kersti Haahti, Luke 8/1/2018
    """

    x = np.array(x)

    # water retention parameters
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    def relcond(x):
        Seff = 1.0 / (1.0 + abs(alfa*x)**n)**m
        r = Seff**0.5 * (1.0 - (1.0 - Seff**(1/m))**m)**2.0
        return r

    Kh = Ksat * relcond(100.0 * np.minimum(x, 0.0))

    return Kh

def rh(psi, T):
    """
    relative humidity in equilibrium with water potential
    Args:
        psi - water potential [m]
        T - temperature [degC]
    Returns
        rh - relative humidity [-]
    """

    rh = np.exp(GRAVITY*MOLAR_MASS_H2O*psi / (GAS_CONSTANT * (T + DEG_TO_KELVIN)))

    return rh
        
""" --- thermal properties  --- """

def volumetric_heat_capacity(poros, wliq=0.0, wice=0.0):
    r""" Computes volumetric heat capacity of porous organic matter.

    Args:
        poros: porosity [m3 m-3]
        wliq: volumetric water content [m3 m-3]
        wice: volumetric ice content [m3 m-3]
    Returns:
        cv: volumetric heat capacity [J m-3 K-1]
    """
    wair = poros - wliq - wice
    cv = CV_ORGANIC * (1. - poros) + CV_WATER * wliq + CV_ICE * wice + CV_AIR * wair

    return cv

def thermal_conductivity(wliq, wice=0.0):
    """ thermal conductivity in organic matter"""
    # o'Donnell et al. 2009
    k = 0.032 + 0.5 * wliq
    return k

def latent_heat_vaporization(T):
    """ latent heat of vaporization of water
    Arg:
        T - temperature (degC)
    Returns:
        Lv - lat. heat. of vaporization (J kg-1)
    """
    return 1.0e6*(2.501 - 2.361e-3*T)  # J kg-1

def frozen_water(T, wtot, fp=0.25, To=0.0):
    r""" Approximates ice content from soil temperature and total water content.

    Args:
        T : soil temperature [degC]
        wtot: total volumetric water content [m3 m-3]
        fp: parameter of freezing curve [-]
            2...4 for clay and 0.5-1.5 for sandy soils
            < 0.5 for peat soils (Nagare et al. 2012 HESS)
        To: freezing temperature of soil water [degC]
    Returns:
        wliq: volumetric water content [m3 m-3]
        wice: volumetric ice content [m3 m-3]
        gamma: dwice/dT
    References:
        For peat soils, see experiment of Nagare et al. 2012:
        http://scholars.wlu.ca/cgi/viewcontent.cgi?article=1018&context=geog_faculty
    """

    wtot = np.array(wtot)
    T = np.array(T)
    fp = np.array(fp)

    wice = wtot*(1.0 - np.exp(-(To - T) / fp))
    # derivative dwliq/dT
    gamma = (wtot - wice) / fp

    ix = np.where(T > To)[0]
    wice[ix] = 0.0
    gamma[ix] = 0.0

    wliq = wtot - wice

    return wliq, wice, gamma