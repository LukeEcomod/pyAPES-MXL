# -*- coding: utf-8 -*-
"""
I/O tools for handling CCFPeat simulations.

Created on Fri Jun 08 10:32:44 2018

Note:
    migrated to python3
    - print >>out to print(file=out)

@author: Kersti Haahti
"""

import sys
import os
import pandas as pd
import numpy as np
import json


def write_ncf(nsim=None, results=None, ncf=None):
    """ Writes model simultaion results in netCDF4-file

    Args:
        index (int): model loop index
        results (dict): calculation results from group
        ncf (object): netCDF4-file handle
    """

    keys = results.keys()
    variables = ncf.variables.keys()

    for key in keys:

        if key in variables and key != 'time':
            length = np.asarray(results[key]).ndim

            if length > 1:
                ncf[key][:, nsim, :] = results[key]
            elif key == 'soil_z' or key == 'canopy_z':
                if nsim == 0:
                    ncf[key][:] = results[key]
            else:
                ncf[key][:, nsim] = results[key]


def initialize_netcdf(variables,
                      sim,
                      soil_nodes,
                      canopy_nodes,
                      plant_nodes,
                      forcing,
                      filepath='results/',
                      filename='climoss.nc',
                      description='Simulation results'):
    """ Climoss netCDF4 format output file initialization

    Args:
        variables (list): list of variables to be saved in netCDF4
        sim (int): number of simulations
        soil_nodes (int): number of soil calculation nodes
        canopy_nodes (int): number of canopy calculation nodes
        forcing: forcing data (pd.dataframe)
        filepath: path for saving results
        filename: filename
    """
    from netCDF4 import Dataset, date2num
    from datetime import datetime

    # dimensions
    date_dimension = None
    simulation_dimension = sim
    soil_dimension = soil_nodes
    canopy_dimension = canopy_nodes
    ptypes_dimension = plant_nodes

    pyAPES_folder = os.getcwd()
    filepath = os.path.join(pyAPES_folder, filepath)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    ff = os.path.join(filepath, filename)

    # create dataset and dimensions
    ncf = Dataset(ff, 'w')
    ncf.description = description
    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncf.source = 'pyAPES_beta2018'

    ncf.createDimension('date', date_dimension)
    ncf.createDimension('simulation', simulation_dimension)
    ncf.createDimension('soil', soil_dimension)
    ncf.createDimension('canopy', canopy_dimension)
    ncf.createDimension('planttype', ptypes_dimension)

    time = ncf.createVariable('date', 'f8', ('date',))
    time.units = 'days since 0001-01-01 00:00:00.0'
    time.calendar = 'standard'
#    tvec = [k.to_datetime() for k in forcing.index] is depricated
    tvec = [pd.to_datetime(k) for k in forcing.index]
    time[:] = date2num(tvec, units=time.units, calendar=time.calendar)

    for var in variables:

        var_name = var[0]
        var_unit = var[1]
        var_dim = var[2]

        variable = ncf.createVariable(
                var_name, 'f4', var_dim)

        variable.units = var_unit

    return ncf, ff

def read_mxl_forcing(forc_filename, start_time=None, end_time=None,
                      dt0=1800.0, dt=1800.0, na_values='NaN'):
    """
    Reads forcing or other data from csv file to dataframe
    Args:
        forc_filename (str): forcing file name with comma separator
        start_time (str): starting time [yyyy-mm-dd], if None first date in
            file used
        end_time (str): ending time [yyyy-mm-dd]. to get whole day, use yyyy-mm-dd+1 
                        if None last date in file used.
        dt0 (float): time step of forcing file [s]
        dt (float): time step of output [s]
        na_values (str/float): nan value representation in file
    Returns:
        Forc (dataframe): dataframe interpolated to dt
    """

    # filepath
    forc_fp = "forcing/" + forc_filename
    dat = pd.read_csv(forc_fp, sep=';', header='infer', na_values=na_values)

    # set to dataframe index
    tvec = pd.to_datetime(dat[['year', 'month', 'day', 'hour', 'minute']])
    tvec = pd.DatetimeIndex(tvec)
    dat.index = tvec

    dat['doy'] = dat['doy'].astype(float)
    dat['Prec'] = dat['Prec'] / dt0 # mm s-1
    
    if start_time == None:
        start_time = dat.index[0]
    if end_time == None:
        end_time = dat.index[-1]
    
    dat = dat[start_time:end_time]
    
    # interpolate to dt
    if dt != dt0:
        step = '%dS' %dt
        sampler = dat.resample(step)
        dat = sampler.interpolate(method='nearest')
        
    return dat


def read_results(outputfiles):
    """
    Opens simulation results netcdf4 dataset in xarray
    (or multiple in list of xarrays)
    Args:
        outputfiles (str or list of str):
            outputfilename or list of outputfilenames
    Returns:
        results (xarray or list of xarrays):
            simulation results from given outputfile(s)
    """

    import xarray as xr

    if type(outputfiles) != list:
        outputfiles = [outputfiles]

    results = []
    for outputfile in outputfiles:
        fp = outputfile
        result = xr.open_dataset(fp)
        result.coords['simulation'] = result.simulation.values
        result.coords['soil'] = result.soil_z.values
        result.coords['canopy'] = result.canopy_z.values
#        result.coords['planttype'] = ['pine','spruce','decid','shrubs']
        results.append(result)

    if len(results) == 1:
        return results[0]
    else:
        return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def jsonify(params, file_name='parameter_space.json'):
    """ Dumps simulation parameters into json format
    """

    os.path.join(file_name)

    with open(file_name, 'w+') as fp:
        json.dump(params, fp, cls=NumpyEncoder)


def open_json(file_path):
    """ Opens a json file
    """
    os.path.join(file_path)
    with open(file_path) as json_data:
        data = json.load(json_data)

    return data

