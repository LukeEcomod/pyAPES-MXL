# -*- coding: utf-8 -*-

"""
Module micromet 
Contains general methods related to Micrometeorological transport processes and physical processes in atmospheric surface layer.
Call all methods as e.g. "cMet.flog_Wind_profile(args)", where "cMet" is instance of cMicroMet -class. 
See documentation of interface of each method for inputs, outputs and datatypes.
Uses numpy for array manipulation & calculation.
    METHODS:

    AUTHOR:
        Samuli Launiainen, METLA 1/2011 - 4/2014 
    VERSION:
        15.04.2014: Codes converted from Matlab to Python. Not fully tested!! \n     
"""

import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps  # machine epsilon

""" define constants """
        
NT = 273.15  # 0 degC in Kelvin
NP = 101300.0  # Pa, sea level normal pressure
R = 8.314462175  # J mol-1 K-1, universal gas constant. One gets specific gas constant by R/M where M is molar mass
CP_AIR_MOLAR = 29.3  # J mol-1 K-1 molar heat capacity of air at constant pressure
CP_AIR_MASS = 1004.67  # J kg-1 K-1 heat capasity of the air at constant pressure
MAIR_DRY = 28.964e-3  # kg mol-1, molar mass of dry air
MH2O = 18.02e-3  # kg mol-1, molar mass of H2O
MCO2 = 44.01e-3  # kg mol-1, molar mass of CO2

SIGMA = 5.6697e-8  # Stefan-Boltzman constant W m-2 K-4
VON_KARMAN = 0.41  # von Kárman constant (-)
GRAV_CONST = 9.81  # ms-2 acceleration due gravity
DEG_TO_RAD = np.pi/180.0  # conversion deg -->rad
RAD_TO_DEG = 180.0/np.pi  # conversion rad -->deg

    
def diffusivities_in_air(T):
    """
    Computes scalar diffusivities in still air as function of temperature.
    INPUT:
        T - air temperature (degC), scalar or array
    OUTPUT:
        Dt - heat diffusivity (m2s-1), scalars or arrays
        Dv - water vapor diffusivity(m2s-1)
        Dc - CO2 diffusivity (m2s-1)
        Do3 - O3 diffusivity (m2s-1)
    SOURCE:
        Based on tabulated diffusivities in Campbell and Norman, 1998.
        Introduction to Environmental Biophysics, Springer.
        Linear least squares fitting to data; valid at least in typical range ambient T (SL 25.4.2014).
    """    
    # diffusivities in m2 s-1
    Dt = 1e-6*(18.8 + 0.128*T)
    Dv = 1e-6*(21.2 + 0.1436*T)
    Dc = 1e-6*(13.8 + 0.096*T)
    Do3 = 1e-6*(17.6 + 0.12*T)
    
    return Dt, Dv, Dc, Do3
    
def pressure_from_altitude(ASL):
    """
    Approximates station pressure from site altitude
    INPUT:
        ASL - elevation above sea level (m)
    OUTPUT:
        Pamb - station pressure (Pa), assuming sea level at NP=101300 Pa
    SOURCE:
        Campbell & Norman, 1998. Introduction to Environmental biophysics, Springer.
    """
    ASL = np.array(ASL)
    Pamb = NP*np.exp(-(ASL/8200.0))
    return Pamb


# -- Functions related to saturation vapor pressure and phase changes of H2O

   
def latent_heat_vaporization(T, units="molar"):
    """
    Temperature dependency of latent heat of vaporization
    INPUT:
        T - temperature (degC)
        units - output units, "mass" = J kg-1 , "molar"= J mol-1
    OUTPUT:
        Lv - latent heat of vaporization in desired units
    """

    if np.any(T > 200):
        T = T - NT  #T must be in degC

    Lv = 1.0e6*(2.501 - 2.361e-3*T)*MH2O  # J mol-1
    
    if units=="mass":
        Lv = Lv / MH2O  # J kg-1
    return Lv
        
            
def saturation_vapor_pressure(T):
    """
    Computes saturation vapor pressure with respect to free and flat water surface for given temperature T
    INPUT:
        T - temperature (degC), scalar or array
    OUTPUT: 
        esat - saturation vapor pressure (Pa)
        delta - slope of saturation vapor pressure curve (Pa degC-1)
    SOURCE:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics.
    """
    # constants
    a = 611.0  # Pa
    b = 17.502  # (-)
    c = 240.97  # degC

    esat = a*np.exp(b*T / (T+c))  # Pa
    delta = b*c*esat / ((c + T)**2)  # Pa degC-1
    return esat, delta


def psycrometric_constant(T, Pamb=101300):
    """
    Computes Psycrometric constant at temperature T
    INPUT:
        T - temperature (degC)
        Pamb - ambient pressure (Pa)
    OUTPUT:
        g - psychrometric constant (Pa K-1)
    USES:
        latent_heat_vaporization
    """
    Lv_mass = latent_heat_vaporization(T, units="mass")  # J kg-1
    g = Pamb*CP_AIR_MASS / (0.622*Lv_mass)  # Pa K-1
    return g


def vpd_from_rh(T, RH):
    """
    Computes vapor pressure deficit from temperature and relative humidity
    INPUT:
        T - temperature (degC), array or scalar
        RH - relative humidity (%), array or scalar
    OUTPUT:
        VPD - vapor pressure deficit (Pa), array
    USES:
        saturation_vapor_pressure
    """
    RH = np.array(RH)
    T = np.array(T)
    VPD, _ = (1.0-RH / 100.0)*saturation_vapor_pressure(T)  # Pa
    return VPD


def air_density(T, P=101300.0, h2o=0.0, units="mass"):
    """
    Computes air density at temperature T, pressure P and vapor pressure H2O
    INPUT:
        T - air temperature (degC), scalar or array
        P - ambient pressure (Pa), scalar or array, optional
        H2O - water vapor partial pressure (Pa), scalar or array, optional (default = dry air)
        units - units to return density: "mass" (default), "molar")
    OUTPUT:
        rhoa - density of dry (default) or moist air (kg m-3 or mol m-3), scalar or array
    Samuli Launiainen 28.4.2014
    """

    Tk = T + NT  # K

    # partial pressures of ideal gas are additive
    Pdry = P - h2o  # pressure of dry air

    if units == "mass":
        rhoa = (Pdry*MAIR_DRY + h2o*MH2O) / (R*Tk)  # kg m-3

    elif units == "molar":
        rho_d = Pdry / (R*Tk)  # dry air, mol m-3
        rho_v = h2o / (R*Tk)  # water vapor, mol m-3
        rhoa = rho_d + rho_v

    else:
        print("-----micromet.air_density: Error - check output units requested ---------")
        rhoa = np.nan
    return rhoa

# ---- evapotranspiration equations ----

def eq_evap(AE, T, P=101300.0, units='W'):
    """
    Calculates the equilibrium evaporation according to McNaughton & Spriggs, 1986. \n
    INPUT: 
        AE - Available energy (Wm-2)  
        T - air temperature (C)
        P - pressure (Pa)
        units - W (Wm-2), mm (mms-1=kg m-2 s-1), mol (mol m-2 s-1)
    OUTPUT: 
        equilibrium evaporation rate (Wm-2)
        constants
        NT=273.15; %0 degC in K
    """
    NT = 273.15
    Mw = 18e-3  # kg mol-1
    L = 1e3*(3147.5 - 2.37*(T + NT))  # latent heat of vaporization of water [J/kg]
    _, s = saturation_vapor_pressure(T)  # des / dT, Pa
  
    g = P*CP_AIR_MASS / (0.622*L)  # psychrom. const (Pa)

    x = np.divide((AE*s), (s+g))  # Wm-2 = Js-1m-2
    if units == 'mm':
        x = x / L  # kg m-2 s-1 = mm s-1
    elif units == 'mol':
        x = x / L / Mw  # mol m-2 s-1

    return x    


def penman_monteith(AE, D, T, Gs, Ga, P=101300.0, units='W'):
    """
    Computes latent heat flux LE (Wm-2) i.e evapotranspiration rate ET (mm/s)
    from Penman-Monteith equation
    INPUT:
       AE - available energy [Wm-2]
       VPD - vapor pressure deficit [Pa]
       T - ambient air temperature [degC]
       Gs - surface conductance [ms-1]
       Ga - aerodynamic conductance [ms-1]
       P - ambient pressure [Pa]
       units - W (Wm-2), mm (mms-1=kg m-2 s-1), mol (mol m-2 s-1)
    OUTPUT:
       x - evaporation rate in 'units'
    """
    # --- constants
    cp = 1004.67  # J kg-1 K-1
    rho = 1.25  # kg m-3
    Mw = 18e-3  # kg mol-1
    _, s = saturation_vapor_pressure(T)  # slope of sat. vapor pressure curve
    g = psycrometric_constant(T)

    L = 1e3 * (3147.5 - 2.37 * (T + 273.15))

    x = (s * AE + rho * cp * Ga * D) / (s + g * (1.0 + Ga / Gs))  # Wm-2

    if units is 'mm':
        x = x / L  # kgm-2s-1 = mms-1
    if units is 'mol':
        x = x / L / Mw  # mol m-2 s-1

    x = np.maximum(x, 0.0)
    return x


# ----- Conductances

def aerodynamic_conductance_from_ust(Ust,U, Stanton):
    """    
    computes canopy aerodynamic conductance (ms-1) from momentum flux measurements
    IN:
       Ustar - friction velocity (ms-1)
       U - mean wind speed at flux measurement heigth (ms-1)
       Stanton - Stanton number (kB-1); for quasi-laminar boundary layer resistance. 
               Typically kB=1...12, use 2 for vegetation ecosystems (Verma, 1989, Garratt and Hicks, 1973)
    OUT:
       ga - aerodynamic conductance [ms-1]
    """   
    kv=0.4 #von Karman constant
    
    ra=U / (Ust**2+eps) + Stanton / ( kv*(Ust+eps) ) #resistance sm-1
    Ga=1./ra # ms-1
    return Ga, ra

def leaf_boundary_layer_conductance(u, d, Ta, dT, P=101300.):
    """
    Computes 2-sided leaf boundary layer conductance assuming mixed forced and free
    convection form two parallel pathways for transport through leaf boundary layer.
    INPUT: u - mean velocity (m/s)
           d - characteristic dimension of the leaf (m)
           Ta - ambient temperature (degC)
           dT - leaf-air temperature difference (degC)
           P - pressure(Pa)
    OUTPUT: boundary-layer conductances (mol m-2 s-1)
        gb_h - heat (mol m-2 s-1)
        gb_c- CO2 (mol m-2 s-1)
        gb_v - H2O (mol m-2 s-1)
        r - ratio of free/forced convection
    Reference: Campbell, S.C., and J.M. Norman (1998),
    An introduction to Environmental Biophysics, Springer, 2nd edition, Ch. 7
    Gaby Katul & Samuli Launiainen
    Note: the factor of 1.4 is adopted for outdoor environment. See Campbell
    and Norman, 1998, p. 89, 101.
    """
    
    # print('U', u, 'd', d, 'Ta', Ta, 'P', P)
    factor1 = 1.4*2  # forced conv. both sides, 1.4 is correction for turbulent flow
    factor2 = 1.5  # free conv.; 0.5 comes from cooler surface up or warmer down
    
    Da_v = 2.4e-5  # Molecular diffusivity of "water vapor" in air at STP (20C and 11kPa) [m2/s]
    Da_c = 1.57e-5  # Molecular diffusivity of "CO2" in air at STP [m2/s]
    Da_T = 2.14e-5  # Molecular diffusivity of "heat" in air at STP [m2/s]
    va = 1.51e-5  # air viscosity at STP [m2/s]
    g = 9.81  # gravitational constant [m/s2]

    # -- Adjust diffusivity, viscosity, and air density to pressure/temp.
    t_adj = (101300.0 / P)*((Ta + 273.15) / 293.16)**1.75
    Da_v = Da_v*t_adj
    Da_c = Da_c*t_adj
    Da_T = Da_T*t_adj
    va = va*t_adj
    rho_air = 44.6*(P / 101300.0)*(273.15 / (Ta + 273.13))  # [mol/m3]

    # ----- Compute the leaf-level dimensionless groups
    Re = u*d / va  # Reynolds number
    Sc_v = va / Da_v  # Schmid numbers for water
    Sc_c = va / Da_c  # Schmid numbers for CO2
    Pr = va / Da_T  # Prandtl number
    Gr = g*(d**3)*abs(dT) / (Ta + 273.15) / (va**2)  # Grashoff number

    # ----- aerodynamic conductance for "forced convection"
    gb_T = (0.664*rho_air*Da_T*Re**0.5*(Pr)**0.33) / d  # [mol/m2/s]
    gb_c=(0.664*rho_air*Da_c*Re**0.5*(Sc_c)**0.33) / d  # [mol/m2/s]
    gb_v=(0.664*rho_air*Da_v*Re**0.5*(Sc_v)**0.33) / d  # [mol/m2/s]

    # ----- Compute the aerodynamic conductance for "free convection"
    gbf_T = (0.54*rho_air*Da_T*(Gr*Pr)**0.25) / d  # [mol/m2/s]
    gbf_c = 0.75*gbf_T  # [mol/m2/s]
    gbf_v = 1.09*gbf_T  # [mol/m2/s]

    # --- aerodynamic conductance: "forced convection"+"free convection"
    gb_h = factor1*gb_T + factor2*gbf_T
    gb_c = factor1*gb_c + factor2*gbf_c
    gb_v = factor1*gb_v + factor2*gbf_v
    # gb_o3=factor1*gb_o3+factor2*gbf_o3

    r = Gr / (Re**2)  # ratio of free/forced convection

    return gb_h, gb_c, gb_v, r  
