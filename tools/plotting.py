# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:09:06 2018

@author: L1656
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from timeseries_tools import diurnal_cycle, yearly_cumulative
from iotools import read_forcing, xarray_to_df

import seaborn as sns
pal = sns.color_palette("hls", 5)

prop_cycle = plt.rcParams['axes.prop_cycle']
default = prop_cycle.by_key()['color']

def plot_results(results, sim_idx=0):

    # Read snow

    # Read weir
    weir = read_forcing("Lettosuo_weir.csv", cols=['runf'])

    # Read gwl
    gwl_meas = read_forcing("Lettosuo_gwl.csv", cols=['WT_E','WT_N','WT_W','WT_S'], na_values=-999)

    plt.figure(figsize=(10,9))
    ax = plt.subplot(711)
    plot_timeseries_xr(results, 'forcing_air_temperature', colors=pal, xticks=False, sim_idx=sim_idx)
    plt.subplot(712, sharex=ax)
    plot_timeseries_xr(results, ['forcing_precipitation', 'canopy_throughfall'], sim_idx=sim_idx,
                       colors=pal[3:], xticks=False, unit_conversion={'unit':'mm h-1', 'conversion':1e3*3600})
    plt.subplot(713, sharex=ax)
    plot_timeseries_xr(results, ['canopy_moss_evaporation', 'canopy_transpiration', 'canopy_evaporation',
                       'soil_subsurface_drainage', 'soil_surface_runoff'], sim_idx=sim_idx,
                       colors=pal, cum=True, stack=True, xticks=False,
                       unit_conversion={'unit':'mm', 'conversion':1e3},)
    plt.subplot(714, sharex=ax)
    plot_timeseries_xr(results, ['canopy_evaporation', 'canopy_transpiration', 'canopy_moss_evaporation'],
                       sim_idx=sim_idx, colors=[pal[1]] + [pal[2]] + [pal[0]], xticks=False,
                       unit_conversion={'unit':'mm h-1', 'conversion':1e3*3600})
    plt.subplot(715, sharex=ax)
    plot_timeseries_df(weir, 'runf', unit_conversion = {'unit':'mm h-1', 'conversion':3600},
                       labels='measured runoff', colors=['k'], xticks=False)
    plot_timeseries_xr(results, 'soil_total_runoff', colors=pal, xticks=False, sim_idx=sim_idx,
                      unit_conversion={'unit':'mm h-1', 'conversion':1e3*3600})
    plt.subplot(716, sharex=ax)
    plot_timeseries_xr(results, 'canopy_snow_water_equivalent', colors=['gray'], xticks=False, stack=True,
                       sim_idx=sim_idx, unit_conversion={'unit':'mm', 'conversion':1e3})
    plt.subplot(717, sharex=ax)
    plot_timeseries_df(gwl_meas, ['WT_E','WT_N','WT_W','WT_S'], colors=[pal[1]], xticks=True)
    plot_timeseries_xr(results, 'soil_ground_water_level', colors=pal[3:], xticks=True, sim_idx=sim_idx)
    plt.ylim(-1.0, 0.0)

    plt.tight_layout(rect=(0, 0, 0.8, 1))

def plot_fluxes(results, sim_idx=0):
    Data = read_forcing("Lettosuo_data_2010_2018.csv", cols=['NEE','GPP','Reco','ET'],
                        start_time=results.date[0].values, end_time=results.date[-1].values,)
    Data.ET = Data.ET / 1800 * 3600
    Data.GPP = -Data.GPP

    variables=['canopy_NEE','canopy_GPP','canopy_Reco','canopy_transpiration','forcing_precipitation']
    df = xarray_to_df(results, variables, sim_idx=sim_idx)
    Data = Data.merge(df, how='outer', left_index=True, right_index=True)
    Data.canopy_transpiration = Data.canopy_transpiration * 1e3 * 3600
    Data.canopy_GPP = Data.canopy_GPP * 44.01 * 1e-3
    Data.canopy_Reco = Data.canopy_Reco * 44.01 * 1e-3

    dates = Data.index

    ix = pd.rolling_mean(Data.forcing_precipitation.values, 48, 1)
    dryc = np.ones(len(dates))
    f = np.where(ix > 0)[0]  # wet canopy indices
    dryc[f] = 0.0

    labels=['Measured', 'Modelled']

    plt.figure(figsize=(10,6))
    plt.subplot(341)
    plot_xy(Data.GPP, Data.canopy_GPP, color=pal[0], axislabels={'x': '', 'y': 'Modelled'})

    plt.subplot(345)
    plot_xy(Data.Reco, Data.canopy_Reco, color=pal[1], axislabels={'x': '', 'y': 'Modelled'})

    months = Data.index.month
    fmonth = 4
    lmonth = 9
    f = np.where((months >= fmonth) & (months <= lmonth) & (dryc == 1))[0]

    plt.subplot(349)
    plot_xy(Data.ET[f], Data.canopy_transpiration[f], color=pal[2], axislabels={'x': 'Measured', 'y': 'Modelled'})

    ax = plt.subplot(3,4,(2,3))
    plot_timeseries_df(Data, ['GPP','canopy_GPP'], colors=['k', pal[0]], xticks=False,
                       labels=labels)
    plt.title('GPP [mg CO2 m-2 s-1]', fontsize=10)
    plt.legend(bbox_to_anchor=(1.6,0.5), loc="center left", frameon=False, borderpad=0.0)

    plt.subplot(3,4,(6,7), sharex=ax)
    plot_timeseries_df(Data, ['Reco','canopy_Reco'], colors=['k', pal[1]], xticks=False,
                       labels=labels)
    plt.title('Reco [mg CO2 m-2 s-1]', fontsize=10)
    plt.legend(bbox_to_anchor=(1.6,0.5), loc="center left", frameon=False, borderpad=0.0)

    plt.subplot(3,4,(10,11), sharex=ax)
    plot_timeseries_df(Data, ['ET','canopy_transpiration'], colors=['k', pal[2]], xticks=True,
                       labels=labels)
    plt.title('ET [mm h-1]', fontsize=10)
    plt.legend(bbox_to_anchor=(1.6,0.5), loc="center left", frameon=False, borderpad=0.0)

    ax =plt.subplot(344)
    plot_diurnal(Data.GPP, color='k', legend=False)
    plot_diurnal(Data.canopy_GPP, color=pal[0], legend=False)
    plt.setp(plt.gca().axes.get_xticklabels(), visible=False)
    plt.xlabel('')

    plt.subplot(348, sharex=ax)
    plot_diurnal(Data.Reco, color='k', legend=False)
    plot_diurnal(Data.canopy_Reco, color=pal[1], legend=False)
    plt.setp(plt.gca().axes.get_xticklabels(), visible=False)
    plt.xlabel('')

    plt.subplot(3,4,12, sharex=ax)
    plot_diurnal(Data.ET[f], color='k', legend=False)
    plot_diurnal(Data.canopy_transpiration[f], color=pal[2], legend=False)

    plt.tight_layout(rect=(0, 0, 0.88, 1), pad=0.5)

def plot_pt_results(results, variable):
    if variable == 'canopy_pt_transpiration':
        unit1={'unit':'mm h-1', 'conversion':1e3*3600}
        unit2={'unit':'mm', 'conversion':1e3}
    else:
        unit1={'unit':'mg CO2 m-2 s-1', 'conversion': 44.01 * 1e-3}
        unit2={'unit':'kg CO2 m-2', 'conversion': 44.01 * 1e-9}
    plt.figure(figsize=(10,4))
    ax = plt.subplot(211)
    plot_timeseries_pt(results, variable, sim_idx=0,
                       unit_conversion=unit1,
                       xticks=False, stack=False, cum=False)
    plt.subplot(212, sharex=ax)
    plot_timeseries_pt(results, variable, sim_idx=0,
                       unit_conversion=unit2,
                       xticks=True, stack=True, cum=True)
    plt.title('')
    plt.tight_layout(rect=(0, 0, 0.9, 1))

def plot_timeseries_pt(results, variable, sim_idx=0, unit_conversion={'unit':None, 'conversion':1.0},
                    xticks=True, stack=True, cum=True):
    """
    Plot results by plant type.
    Args:
        results (xarray): xarray dataset
        variables (str): variable name to plot
        sim_idx (int): index of simulation in xarray dataset
        unit_converrsion (dict):
            'unit' (str): unit of plotted variable (if different from unit in dataset)
            'conversion' (float): conversion needed to get to 'unit'
        xticks (boolean): False for no xticklabels
        stack (boolean): stack plotted timeseries (fills areas)
        cum (boolean): plot yearly cumulative timeseries
    """
    species=['Pine','Spruce','Decid']
    labels=[]
    colors = sns.color_palette("hls", len(results.planttype))
    sub_planttypes = (len(results.planttype) - 1) / 3
    if sub_planttypes > 1:
        for sp in species:
            labels += [sp + '_' + str(k) for k in range(sub_planttypes)]
    else:
        labels = species
    labels.append('Shrubs')
    plot_timeseries_xr([results.isel(planttype=i) for i in range(len(results.planttype))],
                       variable, colors=colors, xticks=xticks,
                       stack=stack, cum=cum, sim_idx=sim_idx,
                       unit_conversion=unit_conversion, labels=labels)

def plot_timeseries_xr(results, variables, sim_idx=0, unit_conversion = {'unit':None, 'conversion':1.0},
                       colors=default, xticks=True, stack=False, cum=False, labels=None):
    """
    Plot timeseries from xarray results.
    Args:
        results (xarray or list of xarray): xarray dataset or list of xarray datasets
        variables (str or list of str): variable names to plot
        sim_idx (int): index of simulation in xarray dataset
        unit_converrsion (dict):
            'unit' (str): unit of plotted variable (if different from unit in dataset)
            'conversion' (float): conversion needed to get to 'unit'
        colors (list): list of color codes
        xticks (boolean): False for no xticklabels
        stack (boolean): stack plotted timeseries (fills areas)
        cum (boolean): plot yearly cumulative timeseries
        labels (list of str): labels corresponding to results[0], results[1],...
    """
    if type(results) != list:
        results = [results]
    if type(variables) != list:
        variables = [variables]
    if len(results) > 1 & len(variables) > 1:
        print "Provide either multiple results and one variable or one result and multiple variables"
    values_all=[]
    for result in results:
        if cum:
            values = yearly_cumulative(result.isel(simulation=sim_idx), variables)
            for k in range(len(values)):
                values_all.append(values[k])
        else:
            for var in variables:
                values_all.append(result[var].isel(simulation=sim_idx).values)
    if len(results) > 1:
        if labels == None:
            labels = [result.description for result in results]
        title = results[0][variables[0]].units.split('[')[0]
    else:
        labels = [results[0][var].units.split('[')[0] for var in variables]
        title = ''

    unit = '[' + results[0][variables[0]].units.split('[')[-1]
    if cum:
        unit = unit.split(' ')[-2]+']'
    if unit_conversion['unit'] != None:
        unit = '[' + unit_conversion['unit'] + ']'
        values_all = [val * unit_conversion['conversion'] for val in values_all]

    if len(values_all) > len(colors):
        colors = colors * (len(values_all) / len(colors) + 1)

    if stack:
        plt.stackplot(results[0].date.values, values_all, labels=labels, colors=colors)
        ymax = max(sum(values_all))
    else:
        for i in range(len(values_all)):
            plt.plot(results[0].date.values, values_all[i], color=colors[i], linewidth=1, label=labels[i])
        ymax = max([max(val) for val in values_all])
    plt.title(title)
    plt.xlim([results[0].date.values[0], results[0].date.values[-1]])
    plt.ylim(min([min(val) for val in values_all]), ymax)
    plt.ylabel(unit)
    plt.setp(plt.gca().axes.get_xticklabels(), visible=xticks)
    plt.xlabel('')
    plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", frameon=False, borderpad=0.0, fontsize=8)

def plot_timeseries_df(data, variables, unit_conversion = {'unit':None, 'conversion':1.0},
                       labels=None, colors=default, xticks=True, stack=False, cum=False, limits=True):
    """
    Plot timeseries from dataframe data.
    Args:
        data (DataFrame): Dataframe of datasets with datetime index
        variables (str or list of str): variable names to plot
        unit_converrsion (dict):
            'unit' (str): unit of plotted variable (if different from unit in dataset)
            'conversion' (float): conversion needed to get to 'unit'
        labels (str or list of str): labels corresponding to variables (if None uses variable names)
        colors (list): list of color codes
        xticks (boolean): False for no xticklabels
        stack (boolean): stack plotted timeseries (fills areas)
        cum (boolean): plot yearly cumulative timeseries
        limits (boolean): set axis limits based on plotted data
    """
    if type(variables) != list:
        variables = [variables]
    values_all=[]
    if cum:
        values = yearly_cumulative(data, variables)
        for k in range(len(values)):
            values_all.append(values[k])
    else:
        for var in variables:
            values_all.append(data[var].values)
    if labels == None:
        labels = variables
    if type(labels) != list:
        labels = [labels]

    if unit_conversion['unit'] != None:
        unit = '[' + unit_conversion['unit'] + ']'
        values_all = [val * unit_conversion['conversion'] for val in values_all]
    else:
        unit = ''

    if len(values_all) > len(colors):
        colors = colors * (len(values_all) / len(colors) + 1)

    if stack:
        plt.stackplot(data.index, values_all, labels=labels, colors=colors)
        ymax = max(sum(values_all))
    else:
        for i in range(len(values_all)):
            plt.plot(data.index, values_all[i], color=colors[i], linewidth=1, label=labels[i])
        ymax = max([max(val) for val in values_all])

    if limits:
        plt.xlim([data.index[0], data.index[-1]])
        plt.ylim(min([min(val) for val in values_all]), ymax)

    plt.ylabel(unit)
    plt.setp(plt.gca().axes.get_xticklabels(), visible=xticks)
    plt.xlabel('')
    plt.legend(bbox_to_anchor=(1.01,0.5), loc="center left", frameon=False, borderpad=0.0, fontsize=8)

def plot_columns(data, col_index=None):
    """
    Plots specified columns from dataframe as timeseries and correlation matrix.
    Args:
        data (DataFrame): dataframe including data to plot
        col_index (list of int): indices of columns to plot as list
            (if None plot all colums in data)
    """
    col_names=[]
    if col_index == None:
        col_index = range(len(data.columns))
    for i in col_index:
        col_names.append(data.columns[i])
    data[col_names].plot(kind='line',marker='o',markersize=1)
    plt.legend()
    axes = pd.plotting.scatter_matrix(data[col_names], figsize=(10, 10), alpha=.2)
    corr = data[col_names].corr().as_matrix()
    lim = [data[col_names].min().min(), data[col_names].max().max()]
    for i in range(len(col_index)):
        for j in range(len(col_index)):
            if i != j:
                idx = np.isfinite(data[col_names[i]]) & np.isfinite(data[col_names[j]])
                if idx.sum() != 0.0:
                    p = np.polyfit(data[col_names[j]][idx], data[col_names[i]][idx], 1)
                    axes[i, j].annotate("y = %.2fx + %.2f \n R2 = %.2f" % (p[0], p[1], corr[i,j]**2), (0.4, 0.9), xycoords='axes fraction', ha='center', va='center')
                    axes[i, j].plot(lim, [p[0]*lim[0] + p[1], p[0]*lim[1] + p[1]], 'r', linewidth=1)
                axes[i, j].plot(lim, lim, 'k--', linewidth=1)
            axes[i, j].set_ylim(lim)
            axes[i, j].set_xlim(lim)

def plot_lad_profiles(filename="letto2016_partial.txt", normed=False, quantiles = [1.0]):
    """
    Plots stand leaf area density profiles from given file.
    Args:
        filename (str): name of file with dbh series
        normed (boolean): normalized profiles
        quantiles (list): cumulative frequency limits for grouping trees, e.g. [0.5, 1.0]
    """
    from parameters.parameter_utils import model_trees

    z = np.linspace(0, 30.0, 100)
    lad_p, lad_s, lad_d, _, _, _, lai_p, lai_s, lai_d = model_trees(z, quantiles,
        normed=normed, dbhfile="parameters/runkolukusarjat/" + filename, plot=True)
    if normed == False:
        lad = z * 0.0
        for k in range(len(quantiles)):
            lad += lad_p[:,k] + lad_s[:,k] +lad_d[:,k]
        lai_tot = sum(lai_p) + sum(lai_s) + sum(lai_d)
        plt.plot(lad, z,':k', label='total, %.2f m$^2$m$^{-2}$' % lai_tot)
    plt.legend(frameon=False, borderpad=0.0, labelspacing=0.1)

def plot_xy(x, y, color=default[0], title='', axislabels={'x':'', 'y':''}):
    """
    Plot x,y scatter with linear regression line, info of relationship and 1:1 line.
    Args:
        x,y (array): arrays for x and y data
        color (str or tuple): color code
        title (str): title of plot
        axislabels (dict)
            'x' (str): x-axis label
            'y' (str): y-axis label
    """
    plt.scatter(x, y, marker='o', color=color, alpha=.2)
    idx = np.isfinite(x) & np.isfinite(y)
    p = np.polyfit(x[idx], y[idx], 1)
    corr = np.corrcoef(x[idx], y[idx])
    plt.annotate("y = %.2fx + %.2f \nR2 = %.2f" % (p[0], p[1], corr[1,0]**2), (0.45, 0.85), xycoords='axes fraction', ha='center', va='center', fontsize=9)
    lim = [min(min(y), min(x)), max(max(y), max(x))]
    plt.plot(lim, [p[0]*lim[0] + p[1], p[0]*lim[1] + p[1]], 'r', linewidth=1)
    plt.plot(lim, lim, 'k--', linewidth=1)
    plt.ylim(lim)
    plt.xlim(lim)
    plt.title(title)
    plt.xlabel(axislabels['x'])
    plt.ylabel(axislabels['y'])

def plot_diurnal(var, quantiles=False, color=default[0], title='', ylabel='', label='median', legend=True):
    """
    Plot diurnal cycle (hourly) for variable
    Args:
        var (Dataframe): dataframe with datetime index and variable to plot
        quantiles (boolean): plot quantiles as ranges
        color (str or tuple): color code
        title (str): title of plot
        ylabel (str): y-axis label
    """
    
    var_diurnal = diurnal_cycle(var)[var.name]
    if quantiles:
        plt.fill_between(var_diurnal['hour'], var_diurnal['5th'], var_diurnal['95th'],
                         color=color, alpha=0.2, label='5...95%')
        plt.fill_between(var_diurnal['hour'], var_diurnal['25th'], var_diurnal['75th'],
                         color=color, alpha=0.4, label='25...75%')
    plt.plot(var_diurnal['hour'], var_diurnal['median'],'o-', markersize=3, color=color,label=label)
    plt.title(title)
    plt.xlabel('Time [h]')
    plt.xlim(0,24)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
        plt.legend(frameon=False, borderpad=0.0,loc="center right")