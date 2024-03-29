# -*- coding: utf-8 -*-
"""
.. module: pyAPES
    :synopsis: APES-model component
.. moduleauthor:: Kersti Haahti

Model framework for soil-bottom layer-canopy-atmosphere interactions.
Based on MatLab implementation by Samuli Launiainen.

Created on Tue Oct 02 09:04:05 2018

Note:
    migrated to python3
    - print on same line
    - dict.keys(), but these are iterated after in for-each-loop

References:
Launiainen, S., Katul, G.G., Lauren, A. and Kolari, P., 2015. Coupling boreal
forest CO2, H2O and energy flows by a vertically structured forest canopy –
Soil model with separate bryophyte layer. Ecological modelling, 312, pp.385-405.

To call model and read results:
    from tools.iotools import read_results
    from pyAPES import driver
    outputfile = driver(create_ncf=True, dbhfile="letto2014.txt")
    results = read_results(outputfile)
"""

import numpy as np
import time
from copy import deepcopy as copy

from tools.iotools import initialize_netcdf,  write_ncf
from tools.iotools import read_hyde_forcing as read_forcing
#from tools.iotools import jsonify
from canopy.canopy import CanopyModel
from soil.soil import Soil

from parameters.parametersets import iterate_parameters

import logging


def driver(create_ncf=False,
           result_file=None,
           parametersets={}):
    """
    Args:
        create_ncf (bool): results saved to netCDF4 file
        result_file (str): name of result file
        parametersets (dict): parameter sets to overwrite default parameters
    """
    # --- LOGGING ---
    from parameters.general import logging_configuration
    from logging.config import dictConfig
    dictConfig(logging_configuration)
    #mpl_logger = logging.getLogger('matplotlib')
    #mpl_logger.setLevel(logging.WARNING)

    # --- PARAMETERS ---
    # Import general parameters
    from parameters.general import gpara
    # Import canopy model parameters
    from parameters.canopy import cpara
    # Import soil model parameters
    from parameters.soil import spara

    if parametersets == {}:
        Nsim = 1
    else:
        Nsim = parametersets['count']

    # --- FORCING ---
    # Read forcing
    forcing = read_forcing(gpara['forc_filename'],
                           gpara['start_time'],
                           gpara['end_time'],
                           dt=gpara['dt'])

    default_params = {
            'canopy': cpara,
            'soil': spara
            }

    param_space = [iterate_parameters(parametersets, copy(default_params), count) for count in range(Nsim)]

    logger = logging.getLogger(__name__)

    logger.info('Simulation started. Number of simulations: {}'.format(Nsim))


    tasks = []

    for k in range(Nsim):
        tasks.append(Model(gpara, param_space[k]['canopy'], param_space[k]['soil'], forcing, nsim=k))

    if create_ncf:
        timestr = time.strftime('%Y%m%d%H%M')
        if result_file:
            filename = result_file
        else:
            filename = timestr + '_pyAPES_results.nc'

        ncf, _ = initialize_netcdf(
                gpara['variables'],
                Nsim,
                tasks[k].Nsoil_nodes,
                tasks[k].Ncanopy_nodes,
                tasks[k].Nplant_types,
                forcing,
                filepath=gpara['results_directory'],
                filename=filename)

        for task in tasks:
            logger.info('Running simulation number (start time %s): %s' % (
                        time.strftime('%Y-%m-%d %H:%M'), task.Nsim))
            running_time = time.time()
            results = task.run()
            logger.info('Running time %.2f seconds' % (time.time() - running_time))
            write_ncf(nsim=task.Nsim, results=results, ncf=ncf)

            del results

        output_file = gpara['results_directory'] + filename
        logger.info('Ready! Results are in: ' + output_file)
        ncf.close()

    else:
        running_time = time.time()
        results = {task.Nsim: task.run() for task in tasks}
        output_file = results
        logger.info('Running time %.2f seconds' % (time.time() - running_time))
    
    return output_file, tasks[0]


class Model(object):
    """
    pyAPES full model
    """
    def __init__(self,
                 gen_para,
                 canopy_para,
                 soil_para,
                 forcing,
                 nsim=0):

        self.dt = gen_para['dt']

        self.Nsteps = len(forcing)
        self.forcing = forcing
        self.Nsim = nsim

        self.Nsoil_nodes = len(soil_para['grid']['dz'])
        self.Ncanopy_nodes = canopy_para['grid']['Nlayers']

        # create soil model instance
        self.soil = Soil(soil_para)
        
        # initial delayed temperature and degreedaysum for pheno & LAI-models
        if 'X' in forcing:
            for pt in list(canopy_para['planttypes'].keys()):
                canopy_para['planttypes'][pt]['phenop'].update({'Xo': forcing['X'].iloc[0]})
        if 'DDsum' in forcing:
            for pt in list(canopy_para['planttypes'].keys()):
                canopy_para['planttypes'][pt]['laip'].update({'DDsum0': forcing['DDsum'].iloc[0]})           

        # create canopy model instance
        self.canopy_model = CanopyModel(canopy_para, self.soil.grid['dz'])

        self.Nplant_types = len(self.canopy_model.planttypes)

        self.results = _create_results(gen_para['variables'],
                                       self.Nsteps,
                                       self.Nsoil_nodes,
                                       self.Ncanopy_nodes,
                                       self.Nplant_types)

    def run(self):
        """ Runs atmosphere-canopy-soil--continuum model"""

        logger = logging.getLogger(__name__)
        logger.info('Running simulation {}'.format(self.Nsim))

        #print('RUNNING')
        k_steps=np.arange(0, self.Nsteps, int(self.Nsteps/10))
        for k in range(0, self.Nsteps):
            
            # progress bar
            if k in k_steps[:-1]:
                s = str(np.where(k_steps==k)[0][0]*10) + '%'
                print('{0}..'.format(s), end=' ')

            """ Canopy, moss and snow """
            # run daily loop (phenology, seasonal LAI and drought response)
            if self.forcing['doy'].iloc[k] != self.forcing['doy'].iloc[k-1] or k == 0:
                self.canopy_model.run_daily(
                        self.forcing['doy'].iloc[k],
                        self.forcing['Tdaily'].iloc[k],
                        Rew=self.forcing['Rew'].iloc[k])
            
            # compile canopy forcing from ubc and uppermost soil layer state
            canopy_forcing = {
                'soil_temperature': self.soil.heat.T[0],
                'soil_water_potential': self.soil.water.h[0],
                'soil_volumetric_water': self.soil.heat.Wliq[0],
                'soil_volumetric_air': self.soil.heat.Wair[0],
                'soil_pond_storage': self.soil.water.h_pond,
                'wind_speed': self.forcing['U'].iloc[k],
                'air_temperature': self.forcing['Tair'].iloc[k],
                'precipitation': self.forcing['Prec'].iloc[k],
                'h2o': self.forcing['H2O'].iloc[k],
                'co2': self.forcing['CO2'].iloc[k],
                'PAR': {'direct': self.forcing['dirPar'].iloc[k],
                        'diffuse': self.forcing['diffPar'].iloc[k]},
                'NIR': {'direct': self.forcing['dirNir'].iloc[k],
                        'diffuse': self.forcing['diffNir'].iloc[k]},
                'lw_in': self.forcing['LWin'].iloc[k],
                'friction_velocity': self.forcing['Ustar'].iloc[k],
                'air_pressure': self.forcing['P'].iloc[k],
                'zenith_angle': self.forcing['Zen'].iloc[k],
            }

            canopy_parameters = {
                'soil_depth': self.soil.grid['z'][0],
                'soil_hydraulic_conductivity': self.soil.water.Kv[0],
                'soil_thermal_conductivity': self.soil.heat.thermal_conductivity[0],
                'date': self.forcing.index[k]
            }

            
            # run self.canopy_model
            canopy_flux, canopy_state, ffloor_flux, ffloor_state = self.canopy_model.run_timestep(
                dt=self.dt,
                forcing=canopy_forcing,
                parameters=canopy_parameters
            )

            # --- Water and Heat flows in soil ---
            # potential infiltration and evaporation from ground surface
            soil_forcing = {
                'potential_infiltration': ffloor_flux['potential_infiltration'],
                'potential_evaporation': (
                    ffloor_flux['evaporation_soil'] + ffloor_flux['capillar_rise']
                ),
                'atmospheric_pressure_head': -1000.0,  # approx. water potential in ambient air.
                'ground_heat_flux': -ffloor_flux['ground_heat'],
                'date': self.forcing.index[k]}

            # FOR SINGLE SOIL LAYER DRIVEN BY MEASUREMENTS UPDATE SOIL STATE
            # FROM FORCING DATA
            
            if self.soil.solve_heat == False:
                soil_forcing.update({'state_heat':
                    {'temperature': self.forcing['Ts'].iloc[k]}})
            if self.soil.solve_water == False:
                soil_forcing.update({'state_water':
                    {'volumetric_water_content': self.forcing['Ws'].iloc[k]}})
                
            # transpiration sink [m s-1]
            rootsink =  self.canopy_model.rad * canopy_flux['transpiration']

            # run self.soil
            soil_flux, soil_state = self.soil.run(
                    dt=self.dt,
                    forcing=soil_forcing,
                    water_sink=rootsink)
            
            forcing_ubc = {
                    'wind_speed': self.forcing['U'].iloc[k],
                    'friction_velocity': self.forcing['Ustar'].iloc[k],
                    'air_temperature': self.forcing['Tair'].iloc[k],
                    'precipitation': self.forcing['Prec'].iloc[k],
                    'h2o': self.forcing['H2O'].iloc[k],
                    'co2': self.forcing['CO2'].iloc[k],
                    'pressure':self.forcing['P'].iloc[k]}

            canopy_state.update(canopy_flux)
            ffloor_state.update(ffloor_flux)
            soil_state.update(soil_flux)

            self.results = _append_results('forcing', k, forcing_ubc, self.results)
            self.results = _append_results('canopy', k, canopy_state, self.results)
            self.results = _append_results('ffloor', k, ffloor_state, self.results)
            self.results = _append_results('soil', k, soil_state, self.results)

        print('100%')
        self.results = _append_results('canopy', None, {'z': self.canopy_model.z}, self.results)

        self.results = _append_results('soil', None, {'z': self.soil.grid['z']}, self.results)

        return self.results


def _create_results(variables, Nstep, Nsoil_nodes, Ncanopy_nodes, Nplant_types):
    """
    Creates temporary results dictionary to accumulate simulation results
    """

    results = {}

    for var in variables:

        var_name = var[0]
        dimensions = var[2]

        if 'canopy' in dimensions:
            if 'date' in dimensions:
                var_shape = [Nstep, Ncanopy_nodes]
            else:
                var_shape = [Ncanopy_nodes]

        elif 'soil' in dimensions:
            if 'date' in dimensions:
                var_shape = [Nstep, Nsoil_nodes]
            else:
                var_shape = [Nsoil_nodes]

        elif 'planttype' in dimensions:
            if 'date' in dimensions:
                var_shape = [Nstep, Nplant_types]
            else:
                var_shape = [Nplant_types]


        else:
            var_shape = [Nstep]

        results[var_name] = np.full(var_shape, np.NAN)

    return results


def _append_results(group, step, step_results, results):
    """
    Adds results from each simulation steps to temporary results dictionary
    """

    results_keys = results.keys()
    step_results_keys = step_results.keys()

    for key in step_results_keys:
        variable = group + '_' + key

#        print(variable)

        if variable in results_keys:
            if variable == 'z':
                results[variable] = step_results[key]
            else:
#                print(type(step_results[key]))
                results[variable][step] = step_results[key]

    return results

#if __name__ == 'main':
#    import logging
#    from parameters.general import logging_configuration
#    from logging.config import dictConfig
#    dictConfig(logging_configuration)

#    driver(create_ncf=True, dbhfile='letto2014.txt')

