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
from canopy.constants import MOLAR_MASS_H2O, PAR_TO_UMOL, EPS

from canopy.radiation import canopy_sw_ZhaoQualls as sw_model
from canopy.radiation import canopy_lw as lw_model

from .moss_flow import closure_model_U_moss, closure_1_model_scalar
from .heat_and_water_flows import frozen_water, water_retention
from canopy.interception import Interception


logger = logging.getLogger(__name__)


class MLM_Moss(object):
    r"""Multi-layer moss canopy
    """

    def __init__(self, para, ini_cond):
        r""" Initializes multi-layer moss object and submodel objects using given parameters.

        Args:
            para - dict of parameters
            ini_cond - dict of initial conditions

        """

        # grid
        self.z = np.linspace(0, para['grid']['zmax'], para['grid']['Nlayers'])  # grid [m] above ground
        self.dz = self.z[1] - self.z[0]  # gridsize [m]
        self.ones = np.ones(len(self.z))  # dummy
        self.zref = para['zref'] # height of forcing data [m]
        
        # moss properties
        self.hc = para['hc'] # canopy height (m)
        self.lad = para['lad']  # shoot-area density (m2m-3)
        self.LAI = sum(self.lad*self.dz)
        
        self.canopy_nodes = np.where(self.lad > 0)[0]
        
        # hydraulic
        self.porosity = para['hydraulic']['porosity']
        self.pF = para['hydraulic']['pF']
        self.Ksat = para['hydraulic']['Ksat']
        self.freezing_curve = para['hydraulic']['freezing_curve']
        
        # radiation
        self.albedo = para['radiation'] # 'PAR', 'NIR'
        self.emissivity = para['radiation']['emissivity']
        self.clump = para['radiation']['clumping']
        self.leaf_angle = para['radiation']['leaf_angle']
        
        #self.radiation = para['radiation']
        
        # compute non-dimensional flow velocity Un = U/ust and momentum diffusivity
        Utop = ini_cond['Utop'] # U/ust at zref
        Ubot = 0.0 # no-slip
        self.Sc = para['Schmidt_nr']
        _, self.Un, self.Kmn, _ = closure_model_U_moss(self.z, self.lad, self.hc, Utop, Ubot)        
        
        self.U = None
        self.Ks = None
        self.length_scale = para['length_scale']
        
        self.Switch_WMA = False
        
        # initial states
        self.T = ini_cond['T']
        self.Wtot = ini_cond['Wtot']
        self.Wliq, self.Wice, _ = frozen_water(self.T, self.Wot, fp=self.freezing_curve, To=0.0)
        self.h = water_retention(self.pF, theta=self.Wliq)
        
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

        self.U = self.Un * forcing['friction_velocity']
        self.Ks = self.Sc * self.Km * forcing['friction_velocity'] # scalar diffusivity

        # --- SW profiles within canopy ---
        # PAR
        _, _, _, PAR_sl, PAR_sh, q_sl, q_sh, q_soil, f_sl, pPAR_alb = sw_model(self.lad*self.dz,
            self.Clump, self.leaf_angle, forcing['zenith_angle'], forcing['dir_par'], forcing['dif_par'],
            self.albedo['PAR'], SoilAlbedo=0.0, PlotFigs=False)
        
        SWabs = q_sl * f_sl + q_sh * (1 - f_sl)
        PAR_incident = PAR_TO_UMOL * (PAR_sl * f_sl + PAR_sh * (1 - f_sl)) # umolm-2s-1

        # NIR
        _, _, _, NIR_sl, NIR_sh, q_sl, q_sh, q_soil, f_sl, NIR_alb = sw_model(self.lad*self.dz,
            self.Clump, self.leaf_angle, forcing['zenith_angle'], forcing['dir_nir'], forcing['dif_nir'],
            self.albedo['NIR'], SoilAlbedo=0.0, PlotFigs=False)
        
        SWabs += q_sl * f_sl + q_sh * (1 - f_sl)

        
        # --- iterative solution of H2O, CO2, T, Tleaf and Tsurf ---

        max_err = 0.01  # maximum relative error
        max_iter = 25  # maximum iterations
        gam = 0.5  # weight for new value in iterations
        err_t, err_h2o,err_co2 = 999., 999., 999.
        Switch_WMA = self.Switch_WMA

        # initialize air-space state variables
        T, H2O, CO2 = self._restore(forcing)
        sources = {
            'h2o': None,  # [mol m-3 s-1]
            'co2': None,  # [umol m-3 s-1]
            'sensible_heat': None,  # [W m-3]
            'latent_heat': None,  # [W m-3]
            'fr': None  # [W m-3]
        }

        iter_no = 0
        while (err_t > max_err or err_h2o > max_err or
               err_co2 > max_err or err_Tl > max_err or
               err_Ts > max_err) and iter_no <= max_iter:

            iter_no += 1
            Tleaf_prev = Tleaf.copy()
            Tsurf_prev = self.forestfloor.temperature

            if self.Switch_Ebal:
                # ---  LW profiles within canopy ---
                ff_longwave = self.forestfloor.longwave_radiation()
                lw_forcing = {
                    'lw_in': forcing['lw_in'],
                    'lw_up': ff_longwave['radiation'],
                    'leaf_temperature': Tleaf_prev,
                }

                lw_params = {
                    'LAIz': self.lad * self.dz,
                    'ff_emissivity': ff_longwave['emissivity']
                }

                radiation_profiles['lw'] = self.radiation.longwave_profiles(
                    forcing=lw_forcing,
                    parameters=lw_params
                )

            # --- heat, h2o and co2 source terms
            for key in sources.keys():
                sources[key] = 0.0 * self.ones

            # --- wet leaf water and energy balance ---
            interception_forcing = {
                'h2o': H2O,
                'wind_speed': U,
                'air_temperature': T,
                'air_pressure': forcing['air_pressure'],
                'leaf_temperature': Tleaf_prev,
                'precipitation': forcing['precipitation'],
            }

            if self.Switch_Ebal:
                interception_forcing.update({
                    'sw_absorbed': radiation_profiles['sw_absorbed'],
                    'lw_radiative_conductance': radiation_profiles['lw']['radiative_conductance'],
                    'net_lw_leaf': radiation_profiles['lw']['net_leaf'],
                })

            interception_params = {
                'LAIz': self.lad * self.dz,
                'leaf_length': self.leaf_length
            }

            interception_controls = {
                'energy_balance': self.Switch_Ebal,
                'logger_info': 'date: {} iteration: {}'.format(
                    parameters['date'],
                    iter_no
                )
            }

            if self.Switch_Ebal:
                interception_forcing.update({
                    'sw_absorbed': radiation_profiles['sw_absorbed'],
                    'lw_radiative_conductance': radiation_profiles['lw']['radiative_conductance'],
                    'net_lw_leaf': radiation_profiles['lw']['net_leaf'],
                })

            wetleaf_fluxes = self.interception.run(
                dt=dt,
                forcing=interception_forcing,
                parameters=interception_params,
                controls=interception_controls
            )

            # dry leaf fraction
            df = self.interception.df

            # update source terms
            for key in wetleaf_fluxes['sources'].keys():
                sources[key] += wetleaf_fluxes['sources'][key] / self.dz

            # canopy leaf temperature
            Tleaf = self.interception.Tl_wet * (1 - df) * self.lad

            # --- dry leaf gas-exchange ---
            pt_stats = []
            for pt in self.planttypes:

                forcing_pt = {
                    'h2o': H2O,
                    'co2': CO2,
                    'air_temperature': T,
                    'air_pressure': forcing['air_pressure'],
                    'wind_speed': U,
                    'par': radiation_profiles['par'],
                    'leaf_temperature': Tleaf_prev,
                }

                if self.Switch_Ebal:
                    forcing_pt.update({
                        'nir': radiation_profiles['nir'],
                        'lw': radiation_profiles['lw']
                    })

                parameters_pt = {
                    'dry_leaf_fraction': self.interception.df,
                    'sunlit_fraction': sunlit_fraction
                }

                controls_pt = {
                    'energy_balance': self.Switch_Ebal,
                    'logger_info': 'date: {} iteration: {}'.format(
                        parameters['date'],
                        iter_no
                    )
                }

                if self.Switch_Ebal:
                    forcing_pt.update({
                        'nir': radiation_profiles['nir'],
                        'lw': radiation_profiles['lw'],
                    })

                # --- sunlit and shaded leaves
                pt_stats_i, pt_sources = pt.run(
                    forcing=forcing_pt,
                    parameters=parameters_pt,
                    controls=controls_pt
                )

                # update source terms
                # Dictionary IS modiefied in a loop. Wrapped in a list.
                for key in pt_sources.keys():
                    sources[key] += pt_sources[key]

                # append results
                pt_stats.append(pt_stats_i)

                # canopy leaf temperature
                Tleaf += pt_stats_i['Tleaf'] * df * pt.lad

            # canopy leaf temperature as weighted average
            Tleaf = Tleaf / (self.lad + EPS)

            err_Tl = max(abs(Tleaf - Tleaf_prev))

            # -- solve forest floor ---

            ff_controls = {
                'energy_balance': self.Switch_Ebal,
                'logger_info': 'iteration: {}'.format(iter_no)
            }

            ff_params = {
                'height': self.z[1],  # height to first calculation node
                'soil_depth': parameters['soil_depth'],
                'nsteps': 20,
                'soil_hydraulic_conductivity': parameters['soil_hydraulic_conductivity'],
                'soil_thermal_conductivity': parameters['soil_thermal_conductivity'],
                'iteration': iter_no,
            }

            ff_forcing = {
                'throughfall_rain': wetleaf_fluxes['throughfall_rain'],
                'throughfall_snow': wetleaf_fluxes['throughfall_snow'],
                'par': radiation_profiles['par']['ground'],
                'air_temperature': T[1],
                'h2o': H2O[1],
                'precipitation_temperature': T[1],
                'air_pressure': forcing['air_pressure'],
                'wind_speed': U[1],
                'friction_velocity': ustar,
                'soil_temperature': forcing['soil_temperature'],
                'soil_water_potential': forcing['soil_water_potential'],
                'soil_volumetric_water': forcing['soil_volumetric_water'],
                'soil_volumetric_air': forcing['soil_volumetric_water'],
                'soil_pond_storage': forcing['soil_pond_storage']
            }

            if self.Switch_Ebal:

                ff_forcing.update({
                    'nir': radiation_profiles['nir']['ground'],
                    'lw_dn': radiation_profiles['lw']['down'][0],
                    'lw_up': radiation_profiles['lw']['up'][0]
                })

            fluxes_ffloor, states_ffloor = self.forestfloor.run(
                dt=dt,
                forcing=ff_forcing,
                parameters=ff_params,
                controls=ff_controls
            )

            err_Ts = abs(Tsurf_prev - self.forestfloor.temperature)

            # check the sign of photosynthesis
            Fc_gr = -fluxes_ffloor['bryo_photosynthesis'] + fluxes_ffloor['respiration']

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

    def layer_energy_balance(self, SWabs, LWabs, U, T, H2O, T0=None, h2o0=None):
        """
        computes layer energy balance
        Args:
            self - object
            SWabs - absorbed SW [Wm-2] per layer
            LWabs - absorbed isothermal LW radiation per layer
            U - flow velocity [ms-1]
            T - air-space temperature [degC]
            H2O - air-space humidity [mol mol-1]
        """
        LAIz = self.lad*dz # total surface area [m2 m-2] per node
        
        # boundary-layer conductance [molm-2s-1]
        gh = 0.135 * np.sqrt(U / self.length_scale)
        gv = 0.147 * np.sqrt(U / self.length_scale)
        
        if T0 is None:
            T0 = self.T
        if h2o0 is None:
            h2o0 = self.equilibrium_humidity
        
        
        # moss temperature from last timestep/iteration
        # moss surface humidity
        # maximum evaporation / 
        
    def _restore(self, forcing):
        """ initialize state variables """
        T = self.ones * ([forcing['air_temperature']])
        H2O = self.ones * ([forcing['h2o']])
        CO2 = self.ones * ([forcing['co2']])

        return T, H2O, CO2

# EOF
