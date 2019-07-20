# -*- coding: utf-8 -*-
"""
Coupled liquid - vapor - heat flow in porous media resembling moss canopy

Samuli Launiainen, Gaby Katul, Antti-Jussi Kieloaho, Kersti Haahti (2019)
Created on Mon Apr  8 01:07:25 2019

@author: slauniai
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from tools.utilities import tridiag as thomas, spatial_average

from canopy.radiation import canopy_sw_ZhaoQualls as sw_model
from canopy.radiation import canopy_lw as lw_model

from moss.moss_flow import MossFlow


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


class MLM_Moss(object):
    """
    Describes heat and water conduction in moss tissues and exchange with canopy air
    """
    def __init__(self, z, para, initial_state):
        """
        initializes object
        """
        self.Nlayers = len(z)
        self.z = z # m, 0.0 at ground, > upwards
        self.dz = z[1] - z[0] # >0
        
        # moss properties
        self.height = para['height'] # canopy height (m)
        self.lad = para['lad']  # shoot-area density (m2m-3)
        self.Lz = self.lad*self.dz # layer leaf-area (m2m-2 ground)
        self.LAI = sum(self.lad*self.dz) # shoot-area index (m2m-2)
        self.length_scale = para['length_scale']
        
        self.porosity = para['porosity']
        self.bulk_density = para['bulk_density'] # kg m-3
        self.dry_mass = self.bulk_density/ self.height # kg m-2 surface
        
        # radiation model
        self.Radiation = MossRadiation(para['radiation'], self.Lz)
        
        # flow model
        self.Flow = MossFlow(z=self.z, lad=self.lad,
                             para=para['flow'], 
                             utop=initial_state['Utop'],
                             ubot=initial_state['Ubot']
                            )

        # hydraulic properties
        self.max_water_content = para['max_water_content']
        self.min_water_content = para['min_water_content']
        self.pF = para['pF']
        self.pF['theta_s'] = self.max_water_content * self.bulk_density / WATER_DENSITY
        self.pF['theta_r'] = self.min_water_content * self.bulk_density / WATER_DENSITY
        self.Ksat = self.pF['Ksat'] #ms-1
        self.freezing_curve = para['freezing_curve']
        
        # state variables
        self.T = initial_state['temperature'] # [degC]
        
        # vol. water contents [m3 m-3]
        self.h = initial_state['water_potential'] # m
        self.Wtot = water_retention(self.pF, psi=self.h)
        #self.Wtot = initial_state['volumetric_water_content']
        self.Wliq, self.Wice, _ = frozen_water(self.T, self.Wtot,
                                               fp=self.freezing_curve, To=0.0)

        # gravimetric water content kg kg-1
        self.water_content = WATER_DENSITY / self.bulk_density * self.Wtot
        
        self.rh = relative_humidity(self.h, self.T) # [-]
        
        #self.available_water, self.available_space = self._flux_constraints()

        #self.Kwater = hydraulic_conductivity(self.pF, self.Wliq, self.Ksat) #ms-1
        #self.Kheat = thermal_conductivity(self.Wliq, wice=self.Wice) #
        
        # radiation model object
        #self.Radiation = MossRadiation(para['radiation'], self.Lz)
        
    def _water_flux_constraints(self, Wliq):
        """ returns available water and available storage space in kg m-2 """
        available_water = (Wliq - self.pF['theta_r'] - EPS) * self.dz * WATER_DENSITY
        available_space = (self.pF['theta_s'] - Wliq - EPS) * self.dz * WATER_DENSITY
        
        return available_water, available_space
    
    def _update_state(self, state):
        """ updates moss state """
        
        self.T = state['temperature']
        self.Wliq = state['volumetric_water_content']
        self.Wice = state['volumetric_ice_content']
        self.h = water_retention(self.pF, theta=self.Wliq)
        self.Wtot = self.Wliq + self.Wice
        self.water_content = WATER_DENSITY / self.bulk_density * self.Wtot
        self.rh = relative_humidity(self.h, self.T)
        
    def _restore_scalar_profiles(self, forcing):
        """ initialize well-mixed profiles """
        #dummy = np.ones(self.Nlayers)
        T = np.ones(self.Flow.Nlayers) * forcing['air_temperature']
        H2O = np.ones(self.Flow.Nlayers) * forcing['h2o']
        CO2 = np.ones(self.Flow.Nlayers) * forcing['co2']
        
        return T, H2O, CO2

    def run_timestep(self, dt, forcing, boundary, sub_dt=60.0):
        """ solves moss canopy model for one timestep 
            dt: timestep [s]
            forcing (dataframe): meteorological and soil forcing data
                'precipitation': precipitation rate [m s-1]
                'air_temperature': air temperature [\ :math:`^{\circ}`\ C]
                'dir_par': direct phototosynthetically active radiation [W m-2]
                'dif_par': diffuse phototosynthetically active radiation [W m-2]
                'dir_nir': direct near infrared radiation [W m-2]
                'dif_nir': diffuse near infrare active radiation [W m-2]
                'lw_in': Downwelling long wave radiation [W m-2]
                'wind_speed': wind speed [m s-1]
                'friction_velocity': friction velocity [m s-1]
                'co2': ambient CO2 mixing ratio [ppm]
                'h2o': ambient H2O mixing ratio [mol mol-1]
                'air_pressure': pressure [Pa]
                'zenith_angle': solar zenith angle [rad]
                'soil_temperature': [\ :math:`^{\circ}`\ C] properties of first soil node
                'soil_water_potential': [m] properties of first soil node
                'soil_evaporation': [molm-2s-1] from the first soil node
                'soil_respiration': [umolm-2s-1] from the first soil node 
            'parameters':
                'date'
                'thermal_conductivity': [W m\ :sup:`-1`\  K\ :sup:`-1`\ ] properties of first soil node
                'hydraulic_conductivity': [m s\ :sup:`-1`\ ] properties of first soil node
                'depth': [m] properties of first soil node
        """
        
        # --- current flow statistics
        self.Flow.update_state(forcing['friction_velocity'])
       
        # --- compute SW radiation within canopy ---
        sw_forcing = {'zenith_angle': forcing['zenith_angle'],
                      'dir_par': forcing['dir_par'],
                      'dif_par': forcing['dif_par'],
                      'dir_nir': forcing['dir_nir'],
                      'dif_nir': forcing['dif_nir'],
                      'water_content': np.mean(self.water_content)
                      }
        SWabs, PARabs, PARz, PAR_alb, NIR_alb, f_sl = self.Radiation.sw_profiles(sw_forcing)
        del sw_forcing
        
        # --- iterative solution of H2O, T, and Ts within canopy air space ---

        max_errT = 0.01  # maximum error
        max_errh2o = 0.01
        max_iter = 50 #50  # maximum iterations
        err_T, err_h2o, err_Ts= 999., 999., 999.

        # initialize air-space state variables
        #T, H2O, CO2 = self._restore_scalar_profiles(forcing)
        
        # initial profiles from last timestep
        if self.Flow.profiles is None:
            T, H2O, CO2 = self._restore_scalar_profiles(forcing)
            self.Flow.profiles = {'T': T, 'H2O': H2O}

        T = self.Flow.profiles['T']
        H2O = self.Flow.profiles['H2O']
        
        Ts0 = self.T.copy()
        Wliq0 = self.Wliq.copy()
        Wice0 = self.Wice.copy()        
        Ts = Ts0.copy()

        iter_no = 0
        while (err_T > max_errT or err_h2o > max_errh2o) and iter_no <= max_iter:

            iter_no += 1
            
            Told = T.copy()
            H2Oold = H2O.copy()
            #Tsold = Ts.copy()
 
            # solve long-wave
            lw_forcing = {'lw_in': forcing['lw_in'],
                          'soil_temperature': Ts[0], #forcing['soil_temperature'],
                          'T': Ts
                          }

            LWabs, LWdn, LWup, gr = self.Radiation.lw_profiles(lw_forcing)
            del lw_forcing
            
            # solve moss energy and water balance      
            ebal_forcing = {'SWabs': SWabs, 
                            'LWabs': LWabs,
                            'T': T[0:self.Nlayers].copy(),
                            'H2O': H2O[0:self.Nlayers].copy(),
                            'U': self.Flow.U[0:self.Nlayers].copy(),
                            'Pamb': forcing['air_pressure'],
                            'soil_temperature': 10.0
                            }
            
#            heat_lbc = {'type': 'flux', 'value': 0.0}
#            #water_lbc = {'type': 'flux', 'value': 0.0}
#            water_lbc = {'type': 'head', 'value': -0.01}
            #print(lbc)
            heat_lbc = boundary['heat']
            water_lbc = boundary['water']
            
            initial_state = {'Ts': Ts0, 'Wliq': Wliq0, 'Wice': Wice0}
            
            source, fluxes, state = self.solve_heat_water(dt,
                                                          ebal_forcing, 
                                                          initial_state,
                                                          heat_lbc, water_lbc
                                                         )
            Ts = state['temperature'].copy()
            err_Ts = 0.0 #np.max(abs(Ts - Tsold))
            
            # solve air-space profiles
            lbc = {'heat': 0.0, # heat flux from ground
                   'H2O': 0.0 # water flux from ground
                  }
            source['H2O'] = source['latent_heat'] / LATENT_HEAT
            initial_profile = {'T': Told, 'H2O': H2Oold}
            
            H2O, T, err_h2o, err_T = self.Flow.scalar_profiles(initial_profile,
                                                               source.copy(),
                                                               forcing['air_pressure'],
                                                               lbc)
            
            #err_t = np.max(abs(H2O - H2Oold))
            #err_h2o = np.max(abs((H2O - H2Oold) / H2Oold))
            
            #print('err_T', err_T < max_errT, 'err_Ts', err_Ts < max_errT, 'err_h2o', err_h2o < max_errh2o, 'itern_no', iter_no)
        
        # plot statistics
        H2Omoss = state['H2Os']
        es = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
        es = es / forcing['air_pressure'] 
        
        plt.figure(100)
        
        plt.subplot(321); plt.plot(Ts, self.z, '-', T, self.Flow.z, '--'); plt.title('Tmoss and T')
        plt.subplot(322); plt.plot(state['volumetric_water_content'], self.z, '-'); plt.title('Wliq')
        plt.subplot(323); plt.plot(source['heat']*self.dz, self.z); plt.title('H')
        plt.subplot(324); plt.plot(source['latent_heat']*self.dz, self.z); plt.title('LE')
        plt.subplot(325); plt.plot(H2O, self.Flow.z, '-', H2Omoss, self.z, '--', es, self.Flow.z, ':'); plt.title('H2O')
        plt.subplot(326); plt.plot(state['volumetric_ice_content'], self.z, '-'); plt.title('Wice')
        #plt.subplot(326); plt.plot(self.Flow.U, self.Flow.z); plt.title('U')
        
        # update profiles; initial guess for next dt
        canopyflx = {'Rnet': np.sum(SWabs + LWabs), 'H': np.sum(source['heat']*self.dz),
                     'LE': np.sum(source['latent_heat']*self.dz),
                     'LWabs': LWabs, 'SWabs': SWabs,
                     'LEz': source['latent_heat']*self.dz,
                     'Hz': source['heat']*self.dz,
                     }
        self.Flow.profiles = {'H2O': H2O, 'T': T}
        return state, canopyflx, T, H2O
    
    def solve_heat_water(self, dt, forcing, initial_state, heat_lbc, water_lbc):
        """
        solution of surface energy balance and heat flow in moss
        NOTE: uses object properties and state variables of water-balance
        
        Args:
            self - object
            dt - timestep [s]
            forcing (dict):
                SWabs - layer absorbed SW radiation [Wm-2 (ground)]
                LWabs - layer absorbed LW radiation [Wm-2 (ground)]    
                U - flow velocity [ms-1]
                T - air-space temperature [degC]
                H2O - air-space humidity [mol mol-1]
                radiative_conductance [mol m-2 s-1]
            Ts - moss temperature profile [degC]
            Wliq - moss liquid water content [m3m-3]
            Wice - moss ice content [m3m-3]
        Returns:
            S - heat sink/source term [Wm-3]
            E - evaporation / condensation rate [s-1]
            H - sensible heat flux [Wm-2 (ground)]
            LE - latent heat flux [Wm-2 (ground)]
            F - radiative conductance term [Wm-2 (ground)]
        """
        SWabs = forcing['SWabs']
        LWabs = forcing['LWabs']
        U = forcing['U']
        Ta = forcing['T']
        H2Oa = forcing['H2O']
        Pamb = forcing['Pamb']
        
        Ts0 = initial_state['Ts'].copy()
        Wliq0 = initial_state['Wliq'].copy()
        Wice0 = initial_state['Wice'].copy()        

        del initial_state
        
        #lw_up = 0.98*STEFAN_BOLTZMANN * (forcing['soil_temperature'] + DEG_TO_KELVIN)**4.0
        
        # heat and water flow parameters & bc's
        parameters = {'pF': self.pF,
                      'Ksat': self.Ksat,
                      'freezing_curve': self.freezing_curve,
                      'porosity': self.porosity}
            
        heat_ubc = {'type': 'flux', 'value': 0.0}
        #heat_lbc = {'type': 'flux', 'value': 0.0} 
        #heat_lbc  =  {'type': 'temperature', 'value': forcing['soil_temperature']}
        
        #water_ubc = {'type': 'flux', 'value': 0.0}
        #water_lbc = {'type': 'flux', 'value': 0.0}
        
        # boundary-layer conductance [molm-2s-1]
        gh = 0.135 * np.sqrt(U / self.length_scale) * self.Lz
        gv = 0.147 * np.sqrt(U / self.length_scale) * self.Lz
        #gr = gr * self.Lz

        dt0 = dt
        dt = 10.0 #dt0 / 100
        Ts = Ts0.copy()
        Wliq = Wliq0.copy()
        Wice = Wice0.copy()
        
        t = 0.0
        
        while t < dt0:
#            # solve long-wave
#            lw_forcing = {'lw_in': forcing['lw_in'],
#                          'soil_temperature': Ts[0], #forcing['soil_temperature'],
#                          'T': Ts
#                          } 
#            LWabs, LWdn, LWup, gr = self.Radiation.lw_profiles(lw_forcing)
            #LWabs = np.zeros(np.shape(self.z))
            #gr = gr * self.Lz
            
            # surface humidity
            h = water_retention(self.pF, theta=Wliq) # m
            rh = relative_humidity(h, Ts) #[-]

            es = 611.0 * np.exp((17.502 * Ts) / (Ts + 240.97))  # Pa
            es = es / Pamb    
            H2Os = rh * es # mol/mol
            
            #print('h2os', H2Os, 'h2oa', H2Oa)
            # evaporation / condensation rate
            E_demand = gv * (H2Os - H2Oa) * MOLAR_MASS_H2O # kgm-2s-1

            available_water, available_space = self._water_flux_constraints(Wliq) # kgm-2 in layer
            
            E = np.maximum(np.minimum(E_demand, 0.9*available_water / dt),
                           -0.9*available_space / dt) #kgm-2s-1

            # latent heat of vaporization
            L = 1e3 * (3147.5 - 2.37 * (Ts + DEG_TO_KELVIN)) # J kg-1
    
            # fluxes [Wm-2 (ground) per layer]
            H = SPECIFIC_HEAT_AIR * gh * (Ts - Ta)
            # F = SPECIFIC_HEAT_AIR * gr * (Ts - Ta)
            LE = L * E
            
            # heat sink / source term [Wm-2 (ground)]   
            S = (SWabs + LWabs) - H - LE #- F
            S = S / self.dz #Wm-3
            E = E / WATER_DENSITY / self.dz # s-1
            
            # solve heat flow
            initial_state = {'volumetric_water_content': Wliq,
                             'volumetric_ice_content': Wice,
                             'temperature': Ts}
            
            
            heat_fluxes, state, _ = heat_flow(dt, self.z, initial_state, parameters, S, heat_lbc, heat_ubc, steps=1)
            
            # convergence, then solve water flow
            #water_fluxes, state, dto = water_flow(dt, self.z, state, parameters, -E, water_lbc)
            
            Ts = state['temperature']
            Wliq = state['volumetric_water_content']
            Wice = state['volumetric_ice_content']

            # then solve water flow
            water_fluxes, state, dto = water_flow(dt, self.z, state, parameters, -E, water_lbc)
  
            t += dt
            #print('t', t)

            Ts = state['temperature']
            Wliq = state['volumetric_water_content']
            Wice = state['volumetric_ice_content']
            #initial_state = states.copy()
            
#            plt.figure(1)
#            plt.subplot(321); plt.plot(Ts, self.z); plt.title('Ts')
#            plt.subplot(322); plt.plot(Wliq, self.z); plt.title('Wliq')
#            plt.subplot(323); plt.plot(H, self.z); plt.title('H')
#            plt.subplot(324); plt.plot(LE, self.z); plt.title('LE')
#            plt.subplot(325); plt.plot(S*self.dz, self.z); plt.title('Sheat')
#            plt.subplot(326); plt.plot(SWabs + LWabs, self.z); plt.title('Rnet')

            # return dicts
            source = {'H2O': E * WATER_DENSITY * MOLAR_MASS_H2O,
                      'heat': H / self.dz,
                      'latent_heat': LE / self.dz,
                      }

            fluxes = {'heat conduction': heat_fluxes['Fheat'],
                      'ebe': heat_fluxes['ebe'],
                      #'water_conduction': water_fluxes['Fwater'],
                      'Qbot': water_fluxes['Qbot'],
                      'mbe': water_fluxes['mbe']
                     }
            state['H2Os'] = H2Os
            
        return source, fluxes, state

        
#%% water and heat flow in moss tissues

def water_flow(t_final, z, initial_state, parameters, source, lbc, steps=10):
    
    r""" Solves liquid water flow in 1-D using implicit, backward finite difference
    solution of Richard's equation.

    Args:
        t_final (float): solution timestep [s]
        'z': node elevations
        initial_state (dict):
            'volumetric_water_content' [m3 m-3]
            'volumetric_ice_content' [m3 m-3]
            'temperature' [degC]
        parameters (dict):
            pF (dict): water retention parameters (van Genuchten)
                'theta_s' (float): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
                'theta_r' (float): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
                'alpha' (float): air entry suction [cm\ :sup:`-1`]
                'n' (float): pore size distribution [-]
                'l' (float): pore connectivity
                Ksat (float): saturated hydraulic conductivity [m s-1]
            freezing_curve (float): freezing curve parameter
        source (array): [s-1], <0 = sink
        lbc (dict): lower boundary condition
            'type': 'flux', 'head',
            'value': flux (ms-1, <0 out) or pressure head (m)
        steps (int): initial number of subtimesteps used to proceed to 't_final'
    Returns:
        fluxes (dict): [m s-1]
            'Eflx': layerwise evaporation/condensation rate [ms-1]
            'Qbot': flux through bottom boundary [ms-1, <0 downward]
            'mbe': mass-balance error [m]
            'dw': water content change [m]
        state (dict):
            'water_potential' [m]
            'volumetric_water_content' [m3 m-3]
            'volumetric_ice_content' [m3m-3] NOT CHANGED DURING COMPUTATIONS

    References:
        vanDam & Feddes (2000): Numerical simulation of infiltration, evaporation and shallow
        groundwater levels with the Richards equation, J.Hydrol 233, 72-85.

    """
    #note: assumes constant properties across layers. if these are arrays, remember to flip all
    pF = parameters['pF']
    Ksat = pF['Ksat']
    fp = parameters['freezing_curve']
    
    # discretation of Richards equation assumes node 0 = top and node N = bottom as in soil-code
    # in moss-object these are opposite so we flip all input vectors, solve water flow and
    # flip outputs again before return
    
    # initial state    
    T_ini = initial_state['temperature']
    W_tot = initial_state['volumetric_water_content'] + initial_state['volumetric_ice_content']

    # flip arrays
    z = z[::-1]
    S = -source[::-1] # s-1
    T_ini = T_ini[::-1]
    W_tot = W_tot[::-1]

    N = len(z) # zero at moss canopy top, z<= 0, constant steps
    dz = np.abs(z[0] - z[1])
    dz2 = dz**2
    
    # initial and computational time step [s]
    dto = t_final / steps
    dt = dto # adjusted during solution
    dt_min = 0.1 # minimum timestep
    
    # convergence criteria
    crit_W = 1.0e-4  # moisture [m3 m-3] 
    crit_h = 1.0e-5  # head [m]
           
    # cumulative fluxes for 0...t_final
    Evap = 0.0
    Q_bot = 0.0
    Q_top = 0.0
    Eflx = np.zeros(N)
    
    # Liquid and ice content, and dWliq/dTs
    W_ini, Wice_ini, gamma = frozen_water(T_ini, W_tot, fp=fp)
    h_ini = water_retention(pF, theta=W_ini) # h [m]
    del W_tot
    
    """ solve for t =[0; t_final] using implict iterative scheme """
    # variables updated during solution
    W = W_ini.copy()
    h = h_ini.copy()

    t = 0.0
    while t < t_final:
        # solution of previous timestep
        h_old = h.copy()
        W_old = W.copy()

        # state variables updated during iteration of time step dt
        h_iter = h.copy()
        W_iter = W.copy()
        #Wice_iter = Wice.copy()

        del h, W
        
        # hydraulic condictivity [m s-1 = kg m-2 s-1]
        KLh = hydraulic_conductivity(pF, W_old, Ksat=Ksat)
        KLh = spatial_average(KLh, method='arithmetic')
        
        # initiate iteration over dt
        err_h = 999.0
        err_W = 999.0
        #err_Wice = 999.0
        iterNo = 0
        
        while (err_h > crit_h or err_W > crit_W): # or err_Wice > crit_Wice or err_T > crit_T):

            iterNo += 1

            # previous iteration values
            h_iter0 = h_iter.copy()                                      
            W_iter0 = W_iter.copy()

            #E = np.zeros(N)
            
            # add here constraint that Ew <= (theta - theta_r) / dt
            f = np.where((W_iter - pF['theta_r'] - 10*EPS)/dt - S <= 0)[0]
            if len(f) >0:
                print(f)
            S[f] = 0.0
            del f
            
            q_top = 0.0
            
            """ lower boundary condition """
            if lbc['type'] == 'free_drain':
                q_bot = - min(KLh[-1], 1e-6)  # avoid extreme high drainage
            elif lbc['type'] == 'impermeable':
                q_bot = 0.0
            elif lbc['type'] == 'flux':
                # note: this restriction does not work for upward flux
                q_bot = np.sign(lbc['value']) * min(abs(lbc['value']), KLh[-1])
            elif lbc['type'] == 'head':
                h_bot = lbc['value']
                # approximate flux: do this now only after convergence
                #q_bot = -KLh[-1] * (h_iter[-1] - h_bot) / dz - KLh[-1]

            # differential water capacity [m-1]
            C = diff_wcapa(pF, h_iter)    
        
            """ set up tridiagonal matrix """
            a = np.zeros(N)  # sub diagonal
            b = np.zeros(N)  # diagonal
            g = np.zeros(N)  # super diag
            f = np.zeros(N)  # rhs
        
            # intermediate nodes i=1...N-1
            b[1:-1] = C[1:-1] + dt / dz2 * (KLh[1:N-1] + KLh[2:N])
            a[1:-1] = - dt / dz2 * KLh[1:N-1]
            g[1:-1] = - dt / dz2 * KLh[2:N]
            f[1:-1] = C[1:-1] * h_iter[1:-1] - (W_iter[1:-1] - W_old[1:-1]) + dt / dz \
                        * (KLh[1:N-1] - KLh[2:N]) - S[1:-1] * dt
        
            # top node i=0 is zero flux
            b[0] = C[0] + dt / dz2 * KLh[1]
            a[0] = 0.0
            g[0] = -dt / dz2 * KLh[1]
            f[0] = C[0] * h_iter[0] - (W_iter[0] - W_old[0]) + dt / dz \
                    * (-q_top - KLh[1]) - S[0] * dt
        
            # bottom node i=N is prescribed flux; head boundary is converted to flux-based
            if lbc['type'] == 'flux':
                b[-1] = C[-1] + dt / dz2 * KLh[N-1]
                a[-1] = -dt / dz2 * KLh[N-1]
                g[-1] = 0.0
                f[-1] = C[-1] * h_iter[-1] - (W_iter[-1] - W_old[-1]) + dt / dz \
                            * (KLh[N-1] + q_bot) - S[-1] * dt
            
            else:  # head boundary
                h_bot = lbc['value']
                b[-1] = C[-1] + dt / dz2 * (KLh[N-1] + KLh[N])
                a[-1] = - dt / dz2 * KLh[N-1]
                g[-1] = 0.0
                f[-1] = C[-1] * h_iter[-1] - (W_iter[-1] - W_old[-1]) + \
                        dt / dz * (KLh[N-1] - KLh[N]) + dt / dz2 *KLh[N] * h_bot - S[-1] * dt
        
            # solve new pressure head and corresponding moisture
            h_iter = thomas(a, b, g, f)
            W_iter = water_retention(pF, psi=h_iter)
                                     
            # check solution, if problem continue with smaller dt or break
            if any(abs(h_iter - h_iter0) > 1.0) or any(np.isnan(h_iter)):
                if dt > dt_min:
                    dt = max(dt / 3.0, dt_min)
                    print('(iteration %s) Solution blowing up, retry with dt = %.1f s' %(iterNo, dt))
                    #print('err_h: %.5f, errW: %.5f, errT: %.5f,, errWice: %.5f' %(err_h, err_W, err_T, err_Wice))
                    iterNo = 0
                    h_iter = h_old.copy()
                    W_iter = W_old.copy()
                    #Wice_iter = Wice_old.copy()
                    continue
                else:  # convergence not reached with dt=30s, break
                    print('(iteration %s)  No solution found (blow up), h and Wtot set to old values' %(iterNo))
                    h_iter = h_old.copy()
                    W_iter = W_old.copy()
                    time.sleep(5)
                    #Wice_iter = Wice_old.copy()
                    break

            # if problems reaching convergence devide time step and retry
            if iterNo == 20:
                if dt > dt_min:
                    dt = max(dt / 3.0, dt_min)
                    print('iteration %s) More than 20 iterations, retry with dt = %.1f s' %(iterNo, dt))
                    print('err_h: %.5f, errW: %.5f' %(err_h, err_W))
                    iterNo = 0
                    continue
                else:  # convergence not reached with dt=dt_min, break
                    print('(iteration %s) Solution not converging, err_h: %.5f, errW: %.5f:' 
                          %(iterNo, err_h, err_W))
                    break

            # errors for determining convergence
            err_h = max(abs(h_iter - h_iter0))
            err_W = max(abs(W_iter - W_iter0))

        # end of iteration loop
            
        # new state
        h = h_iter.copy()
        W = W_iter.copy()
        
        del h_iter, W_iter

        if lbc['type'] == 'head':
            q_bot = -KLh[-1] * (h[-1] - h_bot) / dz - KLh[-1]

        # cumulative fluxes over integration time
        Q_bot += q_bot * dt
        Q_top += q_top * dt
        Evap += sum(S*dz)*dt
        
        Eflx += S*dz*dt

        # solution time and new timestep
        t += dt

        dt_old = dt  # save temporarily
        
        # select new time step based on convergence
        if iterNo <= 3:
            dt = dt * 2
        elif iterNo >= 6:
            dt = dt / 2
        # limit to minimum
        dt = max(dt, dt_min)
        
#        # save dto for output to be used in next run
        if dt_old == t_final or t_final > t:
            dto = min(dt, t_final)
        # limit by time left to solve
        dt = min(dt, t_final-t)
        
#        # fig
#        plt.figure(100)
#        plt.subplot(221); plt.plot(Eflx/dt, z); plt.title('Eflx')
#        plt.subplot(222); plt.plot(h, z); plt.title('head')
#        plt.subplot(223); plt.plot(W, z); plt.title('theta')

    """ time loop ends """

    Wice = Wice_ini.copy()
    # mass balance error [m]
    mbe = (sum(W_ini + Wice_ini) - sum(W + Wice))*dz + Q_bot - Evap
    dw = (sum(W_ini + Wice_ini) - sum(W + Wice))*dz
    
    # a[::-1] flips output arrays
    
    fluxes = {'Eflx': Eflx[::-1] / t_final,  # evaporation/condensation from each layer(m s-1)
              'Qbot': Q_bot / t_final, # liquid water flow at bottom boundary (m s-1), <0 downward
              'mbe': mbe,              # mass balance error (m)
              'dw': dw,                # change in water content (m)
              }
    
    states = {'water_potential': h[::-1],
              'volumetric_water_content': W[::-1],
              'volumetric_ice_content': Wice[::-1],
              'temperature': T_ini[::-1]
             }
    
    return fluxes, states, dto

def heat_flow(t_final, z, initial_state, parameters, source, lbc, ubc, steps=10):
    """ solves heat flow in 1-D using implicit, backward finite difference solution
    Args:
        self: object
        t_final: timestep [s]
        initial_state (dict):
            'water_potential': initial water potential [m]
            'volumetric_water_content' [m3 m-3]
            'volumetric_ice_content' [m3 m-3]
            'temperature' [degC]
        source (array): heat sink/source [Wm-3]
        lbc (dict): {'flux': float} or {'temperature': float}. Note: flux TO layer positive
        ubc (dict): {'flux': float} or {'temperature': float}. Note: flux TO layer positive
    """
    
    #ubc = {'type': 'flux', 'value': 0.0}
 
    #note: assumes constant properties across layers. if these are arrays, remember to flip all
    
    fp = parameters['freezing_curve']
    poros = parameters['porosity']
    
    # discretation of Richards equation assumes node 0 = top and node N = bottom as in soil-code
    # in moss-object these are opposite so we flip all input vectors, solve water flow and
    # flip outputs again before return
    
    # initial state    
    T_ini = initial_state['temperature'].copy()
    Wtot = initial_state['volumetric_water_content'] + initial_state['volumetric_ice_content']

    # flip arrays
    z = z[::-1]
    S = -source[::-1] # s-1
    T_ini = T_ini[::-1]
    Wtot = Wtot[::-1]

    N = len(z) # zero at moss canopy top, z<= 0, constant steps
    dz = np.abs(z[0] - z[1])
    dz2 = dz**2
    
    # initial and computational time step [s]
    dto = t_final / steps
    dt = dto # adjusted during solution
    dt_min = 0.1 # minimum timestep       
    
    # initiate heat flux array [W/m3]
    Fheat = np.zeros(N + 1)

    # Liquid and ice content, and dWliq/dTs
    Wliq, Wice, gamma = frozen_water(T_ini, Wtot, fp=fp, To=FREEZING_POINT_H2O)

    # specifications for iterative solution
    Conv_crit1 = 1.0e-3  # degC
    Conv_crit2 = 1.0e-5  # ice content m3/m3

    # running time [s]
    t = 0.0
    # initial and computational time step [s]
    dto = t_final / steps
    dt = dto  # adjusted during solution

    T = T_ini.copy()
    """ solve water flow for 0...t_final """
    while t < t_final:
        # these will stay during iteration over time step "dt"
        T_old = T
        Wliq_old = Wliq
        Wice_old = Wice

        # bulk soil heat capacity [Jm-3K-1]
        CP_old = volumetric_heat_capacity(poros, Wliq_old, Wice_old)
        
        # apparent heat capacity due to freezing/thawing [Jm-3K-1]
        A = ICE_DENSITY*LATENT_HEAT_FREEZING*gamma

        # thermal conductivity - this remains constant during iteration
        Kt = thermal_conductivity(Wliq, wice=Wice)
        Kt = spatial_average(Kt, method='arithmetic')

        # changes during iteration
        T_iter = T.copy()
        Wice_iter = Wice.copy()

        err1 = 999.0
        err2 = 999.0
        iterNo = 0

        """ iterative solution of heat equation """
        while err1 > Conv_crit1 or err2 > Conv_crit2:

            iterNo += 1

            # bulk soil heat capacity [Jm-3K-1]33
            CP = volumetric_heat_capacity(poros, Wliq, Wice)
            # heat capacity due to freezing/thawing [Jm-3K-1]
            A = ICE_DENSITY*LATENT_HEAT_FREEZING*gamma

            """ set up tridiagonal matrix """
            a = np.zeros(N)
            b = np.zeros(N)
            g = np.zeros(N)
            f = np.zeros(N)

            # intermediate nodes
            b[1:-1] = CP[1:-1] + A[1:-1] + dt / dz2 * (Kt[1:N-1] + Kt[2:N])
            a[1:-1] = - dt / (dz2) * Kt[1:N-1]
            g[1:-1] = - dt / (dz2) * Kt[2:N]
            f[1:-1] = CP_old[1:-1] * T_old[1:-1] + A[1:-1] * T_iter[1:-1] \
                    + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[1:-1] - Wice_old[1:-1]) - S[1:-1] * dt

            # top node i=0
            if ubc['type'] == 'flux':  # flux bc
                F_top = -ubc['value']
                b[0] = CP[0] + A[0] + dt / dz2 * Kt[1]
                a[0] = 0.0
                g[0] = -dt / dz2 * Kt[1]
                f[0] = CP_old[0]*T_old[0] + A[0]*T_iter[0] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[0] - Wice_old[0])\
                        - dt / dz * F_top - dt*S[0]

            if ubc['type'] == 'temperature':  # temperature bc
                T_top = ubc['value']
                b[0] = CP[0] + A[0] + dt / dz2* (Kt[0] + Kt[1])
                a[0] = 0.0
                g[0] = -dt / dz2 * Kt[1]
                f[0] = CP_old[0]*T_old[0] + A[0]*T_iter[0] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[0] - Wice_old[0])\
                        + dt / dz2 *Kt[0]*T_top - dt*S[0]

            # bottom node i=N
            if lbc['type'] == 'flux':  # flux bc
                F_bot = -lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / dz2 * Kt[N-1]
                a[-1] = -dt / dz2 * Kt[N-1]
                g[-1] = 0.0
                f[-1] = CP_old[-1]*T_old[-1] + A[-1]*T_iter[-1] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[-1] - Wice_old[-1])\
                        - dt / dz * F_bot - dt*S[-1]
    
            if lbc['type'] == 'temperature':  # temperature bc
                T_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / dz2 * (Kt[N-1] + Kt[N])
                a[-1] = -dt / dz2 * Kt[N-1]
                g[-1] = 0.0
                f[-1] = CP_old[-1]*T_old[-1] + A[-1]*T_iter[-1] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[-1] - Wice_old[-1])\
                        + dt / dz2 * Kt[N]*T_bot - dt*S[-1]

            """ solve triagonal matrix system """
            # save old iteration values
            T_iterold = T_iter.copy()
            Wice_iterold = Wice_iter.copy()

            # solve new temperature and ice content
            T_iter = thomas(a, b, g, f)
            Wliq_iter, Wice_iter, gamma = frozen_water(T_iter, Wtot, fp=fp, To=FREEZING_POINT_H2O)


            # if problems reaching convergence devide time step and retry
            if iterNo == 20:
                if dt / 3.0 > 101.0:
                    dt = max(dt / 3.0, dt_min)
#                    logger.debug('%s (iteration %s) More than 20 iterations, retry with dt = %.1f s',
#                                 date, iterNo, dt)
                    print('iterno == 20')
                    iterNo = 0
                    T_iter = T_old.copy()
                    Wice_iter = Wice_old.copy()
                    continue
                else:  # convergence not reached with dt=300.0s, break
#                    logger.debug('%s (iteration %s) Solution not converging, err_T: %.5f, err_Wice: %.5f, dt = %.1f s',
#                                 date, iterNo, err1, err2, dt)
                    break
            # check solution, if problem continues break
            elif any(np.isnan(T_iter)):
                if dt / 3.0 > 101.0:
                    dt = max(dt / 3.0, dt_min)
#                    logger.debug('%s (iteration %s) Solution blowing up, retry with dt = %.1f s',
#                                 date, iterNo, dt)
                    print('(iteration %s) Solution blowing up, retry with dt = %.1f s', iterNo, dt)
                    iterNo = 0
                    T_iter = T_old.copy()
                    Wice_iter = Wice_old.copy()
                    continue
                else:  # convergence not reached with dt=300.0s, break
#                    logger.debug('%s (iteration %s) No solution found (blow up), T and Wice set to old values.',
#                                 date,
#                                 iterNo)
                    print('(iteration %s) No solution found (blow up), T and Wice set to old values.', iterNo, dt)
                    T_iter = T_old.copy()
                    Wice_iter = Wice_old.copy()
                    break

            # errors for determining convergence
            err1 = np.max(abs(T_iter - T_iterold))
            err2 = np.max(abs(Wice_iter - Wice_iterold))

        """ iteration loop ends """

        # new state at t
        T = T_iter.copy()
        
        # Liquid and ice content, and dWice/dTs
        Wliq, Wice, gamma = frozen_water(T, Wtot, fp=fp, To=FREEZING_POINT_H2O)
        
        # Heat flux [W m-2]
        Fheat[1:-1] += -Kt[1:-1]*(T[1:] - T[:-1]) / dz * dt / t_final
        if ubc['type'] == 'temperature':
            Fheat[0] += -Kt[0]*(T[0] - T_top) / dz * dt / t_final
        if ubc['type'] == 'flux':
            Fheat[0] += -F_top * dt / t_final
        if lbc['type'] == 'temperature':
            Fheat[-1] += -Kt[-1]*(T_bot - T[-1]) / dz * dt / t_final
        if lbc['type'] == 'flux':
            Fheat[-1] += -F_bot * dt / t_final

        """ solution time and new timestep """
        t += dt
        #print('t', t)
        dt_old = dt  # save temporarily
        # select new time step based on convergence
        if iterNo <= 3:
            dt = dt * 2.0
        elif iterNo >= 6:
            dt = dt / 2.0

        # limit to minimum
        dt = max(dt, dt_min)

        # save dto for output to be used in next run of heatflow1D()
        if dt_old == t_final or t_final > t:
            dto = min(dt, t_final)

        # limit by time left to solve
        dt = min(dt, t_final-t)

    """ time stepping loop ends """

    def heat_content(x):
        _, wice, _ = frozen_water(x, Wtot, fp=fp, To=FREEZING_POINT_H2O)
        Cv = volumetric_heat_capacity(poros, Wliq, Wice)
        return Cv * x - LATENT_HEAT_FREEZING * ICE_DENSITY * wice

#    def heat_balance(x):
#        return heat_content(T_ini) + t_final * (Fheat[:-1] - Fheat[1:]) / dz - heat_content(x)

     
    heat_be = sum(dz * (heat_content(T) - heat_content(T_ini))) / t_final - Fheat[0] - Fheat[-1] + sum(S*dz)
    
    fluxes = {'Fheat': Fheat[::-1],
            'ebe': heat_be}

    state = {'temperature': T[::-1],
             'volumetric_water_content': Wliq[::-1],
             'volumetric_ice_content': Wice[::-1]}

    return fluxes, state, dto

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

    Ts = np.array(pF['theta_s'], ndmin=1)
    Tr = np.array(pF['theta_r'], ndmin=1)
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

    if theta is not None:
        return theta_psi(theta)
    else:
        return psi_theta(psi)

def hydraulic_conductivity(pF, wliq, Ksat=1.0):
    r""" Unsaturated liquid-phase hydraulic conductivity following 
    vanGenuchten-Mualem -model.

    Args:
        pF (dict):
            'theta_s' (float/array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'theta_r' (float/array): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (float/array): air entry suction [cm\ :sup:`-1`]
            'n' (float/array): pore size distribution [-]
        wliq (float or array): liquid water content
        Ksat (float or array): saturated hydraulic conductivity [units]
    Returns:
        Kh (float or array): hydraulic conductivity (if Ksat ~=1 then in [units], else relative [-])

    """
    w = np.array(wliq)
    
    # water retention parameters
    l = pF['l']
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)
    
    # saturation ratio
    S = np.minimum(1.0, (w - pF['theta_r']) / (pF['theta_s'] - pF['theta_r']) + EPS)

    Kh = Ksat * S**l * (1 - (1 - S**(1/m))**m)**2

    return Kh

def diff_wcapa(pF, h):
    r""" Differential water capacity calculated numerically.
    Args:
        pF (dict):
            'theta_s' (array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'theta_r' (array): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (array): air entry suction [cm\ :sup:`-1`]
            'n' (array): pore size distribution [-]
        h (array): pressure head [m]
    Returns:
        dwcapa (array): differential water capacity dTheta/dhead [m\ :sup:`-1`]

    Kersti Haahti, Luke 8/1/2018
    """

    dh = 1e-5
    theta_plus = water_retention(pF, psi=h + dh)
    theta_minus = water_retention(pF, psi=h- dh)

    # differential water capacity
    dwcapa = (theta_plus - theta_minus) / (2 * dh)

    return dwcapa

def relative_humidity(psi, T):
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

def latent_heat_vaporization(T):
    """ latent heat of vaporization of water
    Arg:
        T - temperature (degC)
    Returns:
        Lv - lat. heat. of vaporization (J kg-1)
    """
    return 1.0e6*(2.501 - 2.361e-3*T)  # J kg-1

def saturation_vapor_density(T):
    """
    Args:
        T [degC]
    Returns
        rhos [kg m-3]
    """
    es = 611.0 * np.exp((17.502 * T) / (T + 240.97)) #Pa
    
    rhos = MOLAR_MASS_H2O / (GAS_CONSTANT * (T + DEG_TO_KELVIN)) * es #kg m-3
    return rhos

def air_density(T, P=101300.0):
    """
    Args:
        T - temperature [degC]
        P - ambient pressure [Pa]
    OUTPUT:
        rhoa - density of dry air [kg m-3]
    """
    T = T + DEG_TO_KELVIN

    rhoa = P * MOLAR_MASS_AIR / (GAS_CONSTANT * T) # kg m-3

    return rhoa

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
        gamma: dwice/dT [K-1]
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

def molecular_diffusivity_porous_media(T, wtot, porosity, scalar, P=101300.0):
    r""" Estimates molecular diffusivity in porous media.
    Args:
        T - temperature [degC]
        vol_water + ice: [m3m-3]
        porosity: [m3m-3]
        scalar: 'h2o', 'co2', 'heat' (add' o2', 'ch4')
        P: [Pa]

    Returns:
        D - molecular diffusivity [m2s-1]
    Ref: Millington and Quirk (1961)

    """
    T =T + DEG_TO_KELVIN
    
    # temperature sensitivity of diffusivity in air
    t_adj = (101300.0 / P) * (T / 293.16)**1.75
             
    if scalar.lower() == 'h2o':
        Do = MOLECULAR_DIFFUSIVITY_H2O * t_adj
    if scalar.lower() == 'co2':
        Do = MOLECULAR_DIFFUSIVITY_CO2 * t_adj
    if scalar.lower() == 'heat':
        Do = THERMAL_DIFFUSIVITY_AIR * t_adj 
    
    # [m3/m3], air filled porosity
    afp = np.maximum(0.0, porosity - wtot)

    # [mol m-3], air molar density
    #cair = P / (GAS_CONSTANT * T)
    
    # D/Do, diffusivity in porous media relative to that in free air,
    # Millington and Quirk (1961)
    f = np.power(afp, 10.0/3.0) / porosity**2
                        
    return f * Do

def surface_energy_balance(forcing, params, Ts, gsoil):
    
    alb = params['albedo']
    emi = params['emissivity']
    zref = params['zref']
    
    #SWabs = forcing['SWabs']
    SWup= alb * forcing['SWdn']
    SWabs = (1-alb) * forcing['SWdn']
    LWin = (1-emi) * forcing['LWdn']
    ust = forcing['ust']
    Ta = forcing['Ta'] + DEG_TO_KELVIN
    Ts = Ts + DEG_TO_KELVIN
    
    
    # coeffs
    a = SWabs + LWin
    b = emi*STEFAN_BOLTZMANN
    d = gsoil
    
    def surface_temperature(T, *para):
        a, b, c, d, Ta, Ts = para
        f = a - b * np.power(T, 4) - c*(T - Ta) - d *(T - Ts)
        
        return f

    err = 999.0
    dT = 0.0
    T = 0.5 * (Ta + Ts)
    while err > 0.01:
        gh, _, _ = moss_atm_conductance(zref, ust=ust, dT=dT) # molm-2s-1
        c = SPECIFIC_HEAT_AIR * gh
        para = (a, b, c, d, Ta, Ts)
        
        Told = np.copy(T)
        
        # --- find surface temperature
        T = fsolve(surface_temperature, T, para)
        
        err = abs(T - Told)
    
    # -- fluxes (Wm-2)
    LWup = b * np.power(T, 4)
    H = c * (T - Ta)
    G = d * (T - Ts)
    
    ebe = SWabs + LWin - LWup - H - G
    res = {'H': H, 'G': G, 'SWup': SWup, 'LWup': LWup,
           'Tsurf': T - DEG_TO_KELVIN, 'ebe': ebe}
    return res

def moss_atm_conductance(zref, ust=None, U=None, zom=None, dT=0.0, b=1.1e-3):
    """
    Soil surface - atmosphere transfer conductance for scalars. Two paralell
    mechanisms: forced and free convection
    Args:
        U - wind speed (m/s) at zref
        zref - reference height (m). Log-profile assumed below zref.
        zom - roughness height for momentum (m), ~0.1 x canopy height
        ustar - friction velocity (m/s) at log-regime. if ustar not given,
                it is computed from Uo, zref and zom
        b - parameter for free convection. b=1.1e-3 ... 3.3e-3 from smooth...rough surface
    Returns:
        gh, gv, gc - conductances for heat, water vapor and CO2
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

    d = 0.0 # displacement height
    
    if ust == None:
        ust = U * VON_KARMAN / np.log((zref - d) / zom)

    delta = AIR_VISCOSITY / (VON_KARMAN * ust)
    
    gb_h = (VON_KARMAN * ust) / (Pr - np.log(delta / zref))
    gb_v = (VON_KARMAN * ust) / (Sc_v - np.log(delta / zref))
    gb_c = (VON_KARMAN * ust) / (Sc_c - np.log(delta / zref))
    
    # free convection as parallel pathway, based on Condo and Ishida, 1997.
    #b = 1.1e-3 #ms-1K-1 b=1.1e-3 for smooth, 3.3e-3 for rough surface
    dT = np.maximum(dT, 0.0)
    
    gf_h = b * dT**0.33  # ms-1

    # mol m-2 s-1    
    gb_h = (gb_h + gf_h) * AIR_DENSITY
    gb_v = (gb_v + Sc_v / Pr * gf_h) * AIR_DENSITY
    gb_c = (gb_c + Sc_c / Pr * gf_h) * AIR_DENSITY
    
    return gb_h, gb_v, gb_c

def forward_diff(y, dx):
    """
    computes gradient dy/dx using forward difference
    assumes dx is constatn
    """
    N = len(y)
    dy = np.ones(N) * np.NaN
    dy[0:-1] = np.diff(y)
    dy[-1] = dy[-2]
    return dy / dx

def central_diff(y, dx):
    """
    computes gradient dy/dx with central difference method
    assumes dx is constant
    """
    N = len(y)
    dydx = np.ones(N) * np.NaN
    # -- use central difference for estimating derivatives
    dydx[1:-1] = (y[2:] - y[0:-2]) / (2 * dx)
    # -- use forward difference at lower boundary
    dydx[0] = (y[1] - y[0]) / dx
    # -- use backward difference at upper boundary
    dydx[-1] = (y[-1] - y[-2]) / dx

    return dydx

class MossRadiation(object):
    """ handles SW-radiation in multi-layer moss canopy"""
        
    def __init__(self, para, Lz):
        
        self.PAR_albedo = para['PAR_albedo']
        self.NIR_albedo = para['NIR_albedo']
        self.emissivity = para['emissivity']
        self.leaf_angle = para['leaf_angle'] # (1=spherical, 0=vertical, inf.=horizontal) (-)
        self.Clump = para['clump']
        self.soil_albedo = 0.0
        self.max_water_content = para['max_water_content'] # gH2O g-1 dry mass
        
        #note: self.Lz must include one additional layer above moss canopy, where Lz=0.0
        self.Lz = np.append(Lz, 0) # layerwise leaf-area per unit ground area (m2m-2)   
        
    def sw_profiles(self, forcing):
        """ --- SW profiles within canopy ---
        Args:
            forcing - dict
                bulk water_content [gg-1] 
                zenith_angle [rad]
                dir_par, dif_par - direct & diffuse par [Wm-2]
                dir_nir, dif_nir - direct & diffuse nir [Wm-2]
        Returns:
            SWabs - absorbed SW in a layer [Wm-2 (ground)]
            PARabs - absorbed PAR in a layer per unit leaf-area [umolm-2(leaf)s-1]
            PARn - downwelling PAR to a layer [umolm-2(ground)s-1]
            PAR_alb, NIR_alb - canopy albedos [-]
            f_sl - sunlit fraction
        """
        
        # here adjust albedo for water content; this is from Antti's code and represents bulk moss
        #f = 1.0 + 3.5 / (1.0 + np.power(10.0, 4.53 * forcing['water_content'] / self.max_water_content))
        f = 1.0
        PARb, PARd, PARu, _, _, q_sl, q_sh, q_soil, f_sl, PAR_alb = sw_model(self.Lz,
            self.Clump, self.leaf_angle, forcing['zenith_angle'], forcing['dir_par'], forcing['dif_par'],
            self.PAR_albedo * f, SoilAlbedo=self.soil_albedo, PlotFigs=False)
         
        PARabs = q_sl * f_sl + q_sh * (1 - f_sl) # Wm-2(leaf)
        #PAR_incident = PAR_TO_UMOL * (PAR_sl * f_sl + PAR_sh * (1 - f_sl)) # umolm-2s-1

        # NIR
        _, _, _, _, _, q_sl, q_sh, q_soil, f_sl, NIR_alb = sw_model(self.Lz,
            self.Clump, self.leaf_angle, forcing['zenith_angle'], forcing['dir_nir'], forcing['dif_nir'],
            self.NIR_albedo * f, SoilAlbedo=self.soil_albedo, PlotFigs=False)
        
        NIRabs = q_sl * f_sl + q_sh * (1 - f_sl)
        
        SWabs = (PARabs + NIRabs) * self.Lz # Wm-2 ground per layer
        
        # PAR fluxes returned in umolm-2s-1
        PARabs *= PAR_TO_UMOL
        PARin = (PARb + PARd + PARu) * PAR_TO_UMOL
        
        return SWabs[0:-1], PARabs[0:-1], PARin[1:], PAR_alb, NIR_alb, f_sl
    

    def lw_profiles(self, forcing):
        """ --- LW profiles in moss canopy ---
        Args:
            forcing - dict
                T [degC]
                lw_in [Wm-2] downward long-wave
        Returns:
            LWlayer [Wm-2 (leaf)] net absorbed long-wave per layer
            LWdn [Wm-2] downward LW at layer
            LWup [Wm-2] upward LW at layer
            gr [molm-2s-1] radiative conductance
        """
        # pad input Tmoss in case only moss internal profile is given
        if len(forcing['T']) < len(self.Lz):
            forcing['T'] = np.append(forcing['T'], forcing['T'][-1])

        forcing['lw_up'] = 0.98 * STEFAN_BOLTZMANN *(forcing['soil_temperature'] + DEG_TO_KELVIN)**4.0
        LWlayer, LWdn, LWup, gr = lw_model(self.Lz, self.Clump, self.leaf_angle, forcing['T'], 
                                          forcing['lw_in'], forcing['lw_up'], leaf_emi=self.emissivity, PlotFigs=False)
        
        LWlayer *= self.Lz # Wm-2 ground per layer
        
        return LWlayer[0:-1], LWdn[1:], LWup[0:-1], gr[0:-1]