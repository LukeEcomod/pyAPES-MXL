#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module: canopy
    :synopsis: APES-model component
.. moduleauthor:: Kersti Haahti

Describes H2O, CO2, energy transfer in multilayer canopy.
Based on MatLab implementation by Samuli Launiainen.

Created on Tue Oct 02 09:04:05 2018

Note:
    Migrated to python3:
        - absolute imports
        - for each loop: usage of enumerate() instead of list(range(len(foo)))
        - if dictionary IS modified in a for-loop dict.items()/.keys()/.values() wrapped in a list
        (this is not necessary if only modifying and not adding/deleting)
        - if dictionary IS NOT modified in a for-loop, wrapped anyways (for caution).

References:
Launiainen, S., Katul, G.G., Lauren, A. and Kolari, P., 2015. Coupling boreal
forest CO2, H2O and energy flows by a vertically structured forest canopy –
Soil model with separate bryophyte layer. Ecological modelling, 312, pp.385-405.

Note: 25.02.19/SL : Check L493 T_prev
                    Changed def run_daily call; added Rew as argument
"""

import logging
# from copy import deepcopy
import numpy as np
from canopy.constants import MOLAR_MASS_H2O, PAR_TO_UMOL, EPS, STEFAN_BOLTZMANN, DEG_TO_KELVIN,\
                             SPECIFIC_HEAT_AIR


from canopy.radiation import canopy_sw_Spitters as sw_model
from canopy.radiation import canopy_lw as lw_model

from .moss_flow import closure_model_U_moss, closure_1_model_scalar
from moss_heat_water import MLM_Moss
#from .heat_and_water_flows import frozen_water, water_retention
from canopy.interception import Interception


logger = logging.getLogger(__name__)


class MossCanopy(object):
    r"""Multi-layer moss canopy
    """

    def __init__(self, para, ini_cond):
        r""" Initializes multi-layer moss object and submodel objects using given parameters.

        Args:
            para - dict of parameters
            ini_cond - dict of initial conditions

        """
        # computation grid
        self.z = np.linspace(0, para['grid']['zmax'], para['grid']['Nlayers'])  # grid [m] above ground
        self.dz = self.z[1] - self.z[0]  # gridsize [m]
        self.Nlayers = len(self.z)
        self.ones = np.ones(len(self.z))  # dummy

        # create MLM_Moss instance:
        self.Moss = MLM_Moss(self.z, para, ini_cond)
        self.canopy_nodes = np.where(self.Moss.lad > 0)[0]

        # moss canopy properties
        #self.height = Moss.height # canopy height (m)
        self.lad = np.zeros(self.Nlayers)
        self.lad[self.canopy_nodes] = self.Moss.lad # shoot-area density (m2m-3)        
   
        # flow and scalar transport in moss canopy air space
        # compute non-dimensional flow velocity Un = U/ust and momentum diffusivity
        Utop = ini_cond['Utop'] # U/ust at zref
        Ubot = 0.0 # no-slip
        self.Sc = para['Schmidt_nr']
        self.taun, self.Un, self.Kmn, _ = closure_model_U_moss(self.z, self.lad, self.hc, Utop, Ubot)        
        
        #self.U = None
        #self.Ks = None
        self.length_scale = para['length_scale']
        
        self.Switch_WMA = False
                
    def run_timestep(self, dt, forcing, parameters):
        r""" Calculates one timestep and updates state of CanopyModel object.

        Args:
            dt: timestep [s]
            forcing (dataframe): meteorological and soil forcing data  !! NOT UP TO DATE
                'precipitation': precipitation rate [m s-1]
                'air_temperature': air temperature [\ :math:`^{\circ}`\ C]
                'dir_par': direct fotosynthetically active radiation [W m-2]
                'dif_par': diffuse fotosynthetically active radiation [W m-2]
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

        Returns:
            fluxes (dict)
            states (dict)
        """
        logger = logging.getLogger(__name__)

        U = self.Un * forcing['friction_velocity'] # ms-1
        #ustar = np.sqrt(abs(self.taun) * forcing['friction_velocity']**2) # ms-1
        Ks = self.Sc * self.Kmn * forcing['friction_velocity'] # scalar diffusivity [m2s-1]
        

        # --- compute SW radiation within canopy ---
        sw_forcing = {'zenith_angle': forcing['zenith_angle'],
                      'dir_par': forcing['dir_par'],
                      'dif_par': forcing['dif_par'],
                      'dir_nir': forcing['dir_nir'],
                      'dif_nir': forcing['dif_nir'],
                      'water_content': self.bulk_water_content
                      }
        SWabs, PARinc, PAR_albedo, NIR_albedo = self.Moss.Radiation.sw_profiles(sw_forcing)
        del sw_forcing
        
        # --- iterative solution of H2O, T, and Ts within canopy air space ---

        max_err = 0.01  # maximum relative error
        max_iter = 25  # maximum iterations
        gam = 0.5  # weight for new value in iterations
        err_T, err_h2o, err_Ts= 999., 999., 999., 999.
        Switch_WMA = self.Switch_WMA

        # initialize air-space state variables
        T, H2O, CO2 = self._restore(forcing)
        sources = {
            'h2o': None,  # [mol m-3 s-1]
            'co2': None,  # [umol m-3 s-1]
            'heat': None,  # [W m-3]
            'latent_heat': None,  # [W m-3]
            #'fr': None  # [W m-3]
        }
        Ts0 = self.Moss.T.copy()
        iter_no = 0
        while (err_T > max_err or err_h2o > max_err or err_Ts > max_err) and iter_no <= max_iter:

            iter_no += 1

#            if self.Switch_Ebal:
#                # ---  LW profiles within canopy ---
#                lw_forcing = {'lw_in': forcing['lw_in'],
#                              'soil_temperature': forcing['soil_temperature'],
#                              'T': Tmoss,
#                              }
#                LWlayer, LWdn, LWup, gr = self.Radiation.lw_profiles(lw_forcing)
                
                
#            # --- heat, h2o source terms
#            for key in sources.keys():
#                sources[key] = 0.0 * self.ones
            
            ebal_forcing = {'SWabs': SWabs, 'T': T, 'U': U, 'H2O': H2O,
                            'Pamb': forcing['air_pressure'],
                            'soil_temperature': 10.0,
                            'lw_in': forcing['lw_in']
                           }
                
            # solve moss layer energy balance and heat transfer
            S, fluxes, states = self.Moss.solve_heat(dt, ebal_forcing, Ts, self.Moss.Wliq, self.Moss.Wice)

            """  --- solve scalar profiles (H2O, CO2, T) """
            if Switch_WMA is False:
                # to recognize oscillation
                if iter_no > 1:
                    T_prev2 = T_prev.copy()
                T_prev = T.copy()

                H2O, CO2, T, err_h2o, err_co2, err_t = self.micromet.scalar_profiles(
                        gam, H2O, CO2, T, forcing['air_pressure'],
                        source=sources,
                        lbc={'H2O': fluxes_ffloor['evaporation'],
                             'CO2': Fc_gr,
                             'T': fluxes_ffloor['sensible_heat']},
                        Ebal=self.Switch_Ebal)

                # to recognize oscillation
                if iter_no > 5 and np.mean((T_prev - T)**2) > np.mean((T_prev2 - T)**2):
                    T = (T_prev + T) / 2
                    gam = max(gam / 2, 0.25)

                if (iter_no == max_iter or any(np.isnan(T)) or
                    any(np.isnan(H2O)) or any(np.isnan(CO2))):

                    if (any(np.isnan(T)) or any(np.isnan(H2O)) or any(np.isnan(CO2))):
                        logger.debug('%s Solution of profiles blowing up, T nan %s, H2O nan %s, CO2 nan %s',
                                         parameters['date'],
                                         any(np.isnan(T)), any(np.isnan(H2O)), any(np.isnan(CO2)))
                    elif max(err_t, err_h2o, err_co2, err_Tl, err_Ts) < 0.05:
                        if max(err_t, err_h2o, err_co2, err_Tl, err_Ts) > 0.01:
                            logger.debug('%s Maximum iterations reached but error tolerable < 0.05',
                                         parameters['date'])
                        break

                    Switch_WMA = True  # if no convergence, re-compute with WMA -assumption

                    logger.debug('%s Switched to WMA assumption: err_T %.4f, err_H2O %.4f, err_CO2 %.4f, err_Tl %.4f, err_Ts %.4f',
                                 parameters['date'],
                                 err_t, err_h2o, err_co2, err_Tl, err_Ts)

                    # reset values
                    iter_no = 0
                    err_t, err_h2o, err_co2, err_Tl, err_Ts = 999., 999., 999., 999., 999.
                    T, H2O, CO2, Tleaf = self._restore(forcing)
            else:
                err_h2o, err_co2, err_t = 0.0, 0.0, 0.0

        """ --- update state variables --- """
        self.interception.update()
        self.forestfloor.update()


        if self.Switch_Ebal:
            # energy closure of canopy  -- THIS IS EQUAL TO frsource (the error caused by linearizing sigma*ef*T^4)
            energy_closure =  sum((radiation_profiles['sw_absorbed'] +
                                   radiation_profiles['lw']['net_leaf']) * self.lad * self.dz) - (  # absorbed radiation
                              sum(sources['sensible_heat'] * self.dz)  # sensible heat
                              + sum(sources['latent_heat'] * self.dz))  # latent heat


        if self.Switch_WMA is False:
            state_canopy.update({'h2o': H2O,
                          'co2': CO2,
                          'temperature': T,
                          'WMA_assumption': 1.0*Switch_WMA})

        if self.Switch_Ebal:
            # layer - averaged leaf temperatures are averaged over plant-types
            Tleaf_sl = np.where(self.lad > 0.0,
                                sum([pt_st['Tleaf_sl'] for pt_st in pt_stats]) / (self.lad + EPS),
                                np.nan)
            Tleaf_sh = np.where(self.lad > 0.0,
                                sum([pt_st['Tleaf_sh'] for pt_st in pt_stats]) / (self.lad + EPS),
                                np.nan)
            Tleaf_wet = np.where(self.lad > 0.0,
                                 self.interception.Tl_wet,
                                 np.nan)
            SWnet = (radiation_profiles['nir']['down'][-1] - radiation_profiles['nir']['up'][-1] +
                     radiation_profiles['par']['down'][-1] - radiation_profiles['par']['up'][-1])
            LWnet = (radiation_profiles['lw']['down'][-1] - radiation_profiles['lw']['up'][-1])

            state_canopy.update({
                    'Tleaf_wet': Tleaf_wet,
                    'Tleaf_sl': Tleaf_sl,
                    'Tleaf_sh': Tleaf_sh,
                    'Tleaf': np.where(self.lad > 0.0, Tleaf, np.nan)
                    })

            fluxes_canopy.update({
                    'leaf_SW_absorbed': radiation_profiles['sw_absorbed'],
                    'leaf_net_LW': radiation_profiles['lw']['net_leaf'],
                    'sensible_heat_flux': flux_sensible_heat,  # [W m-2]
                    'energy_closure': energy_closure,
                    'SWnet': SWnet,
                    'LWnet': LWnet,
                    'PARdn': radiation_profiles['par']['down'],
                    'PARup': radiation_profiles['par']['up'],
                    'NIRdn': radiation_profiles['nir']['down'],
                    'NIRup': radiation_profiles['nir']['up'],
                    'LWdn':  radiation_profiles['lw']['down'],
                    'LWup':  radiation_profiles['lw']['up'],
                    'fr_source': sum(sources['fr'] * self.dz)})

        return fluxes_canopy, state_canopy, fluxes_ffloor, states_ffloor

    def _restore(self, forcing):
        """ initialize well-mixed profiles """
        T = self.ones * ([forcing['air_temperature']])
        H2O = self.ones * ([forcing['h2o']])
        CO2 = self.ones * ([forcing['co2']])

        return T, H2O, CO2

#%% ---sub-models as classes
        
class MossRadiation(object):
    """ handles SW-radiation in multi-layer moss canopy"""
    
    from canopy.radiation import canopy_sw_ZhaoQualls as sw_model
    from canopy.radiation import canopy_lw as lw_model
    
    def __init__(self, para, Lz):
        
        self.PAR_albedo = para['PAR_albedo']
        self.NIR_albedo = para['NIR_albedo']
        self.emissivity = para['emissivity']
        self.leaf_angle = para['leaf_angle'] # (1=spherical, 0=vertical, inf.=horizontal) (-)
        self.Clump = para['clump']
        self.soil_albedo = 0.0
        self.max_water_content = para['max_water_content'] # gH2O g-1 dry mass
        
        self.Lz = Lz # layerwise leaf-area per unit ground area (m2m-2)   
        
        
    def sw_profiles(self, forcing):
        """ --- SW profiles within canopy ---
        Args:
            forcing - dict
                bulk water_content [gg-1] 
                zenith_angle [rad]
                dir_par, dif_par - direct & diffuse par [Wm-2]
                dir_nir, dif_nir - direct & diffuse nir [Wm-2]
        Returns:
            SWabs - absorbed SW [Wm-2 (leaf)]
            PAR_incident - incident PAR [Wm-2 (leaf)]
            PAR_alb, NIR_alb - canopy albedos [-]
        """
        
        # here adjust albedo for water content; this is from Antti's code and represents bulk moss
        f = 1.0 + 3.5 / (1.0 + np.power(10.0, 4.53 * forcing['water_content'] / self.max_water_content))
        
        # PAR
        _, _, _, PAR_sl, PAR_sh, q_sl, q_sh, q_soil, f_sl, PAR_alb = sw_model(self.Lz,
            self.Clump, self.leaf_angle, forcing['zenith_angle'], forcing['dir_par'], forcing['dif_par'],
            self.PAR_albedo * f, SoilAlbedo=self.soil_albedo, PlotFigs=False)
        
        SWabs = q_sl * f_sl + q_sh * (1 - f_sl)
        PAR_incident = PAR_TO_UMOL * (PAR_sl * f_sl + PAR_sh * (1 - f_sl)) # umolm-2s-1

        # NIR
        _, _, _, NIR_sl, NIR_sh, q_sl, q_sh, q_soil, f_sl, NIR_alb = sw_model(self.Lz,
            self.Clump, self.leaf_angle, forcing['zenith_angle'], forcing['dir_nir'], forcing['dif_nir'],
            self.NIR_albedo * f, SoilAlbedo=self.soil_albedo, PlotFigs=False)
        
        SWabs += q_sl * f_sl + q_sh * (1 - f_sl)
        
        SWabs *= self.Lz # Wm-2 ground per layer
        return SWabs, PAR_incident, PAR_alb, NIR_alb
    

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
        """
        forcing['lw_up'] = 0.98 * STEFAN_BOLTZMANN *(forcing['soil_temperature'] + DEG_TO_KELVIN)**4.0
        LWlayer, LWdn, LWup, gr = lw_model(self.Lz, self.Clump, self.leaf_angle, forcing['T'], 
                                          forcing['lw_in'], forcing['lw_up'], leaf_emi=self.emissivity, PlotFigs=False)
        
        LWlayer *= self.Lz # Wm-2 ground per layer
        
        return LWlayer, LWdn, LWup, gr

def e_sat(T):
    """
    Computes saturation vapor pressure (Pa), slope of vapor pressure curve
    [Pa K-1]  and psychrometric constant [Pa K-1]
    IN:
        T - air temperature (degC)
    OUT:
        esa - saturation vapor pressure in Pa
        s - slope of saturation vapor pressure curve (Pa K-1)
    SOURCE:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics. (p.41)
    """

    esa = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    s = 17.502 * 240.97 * esa / ((240.97 + T)**2)

    return esa, s        
        # EOF