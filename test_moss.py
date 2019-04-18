# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:11:32 2019

@author: slauniai
"""

import numpy as np
import matplotlib.pyplot as plt

from moss.heat_and_water_flows import MOLECULAR_DIFFUSIVITY_H2O, THERMAL_DIFFUSIVITY_AIR
from moss.heat_and_water_flows import saturation_vapor_density, water_heat_flow, \
    surface_energy_balance, water_retention, hydraulic_conductivity, thermal_conductivity, \
    molecular_diffusivity_porous_media
    
height = 0.20 # m
dz = 0.01

z = np.arange(0, -(height +dz), -dz)
g = np.ones(len(z))

# pleurozium
para = {'porosity': 0.98,
        'pF': {'ThetaS': 0.17, 'ThetaR': 0.026, 
               'alpha': 0.13, 'n': 2.17, 'l': -2.37},
        'Ksat': 1.2e-8,
        'freezing_curve': 0.25,
        'albedo': 0.15,
        'emissivity': 0.98,
        'zref': 0.2 # 1st node above ground
        }

# conductivities
decay = np.exp(30*z)
Kvf = 100 * MOLECULAR_DIFFUSIVITY_H2O * decay
Ktf = 100 * THERMAL_DIFFUSIVITY_AIR * decay
#Ktf[-2:] = THERMAL_DIFFUSIVITY_AIR
para.update({'Kvf': Kvf, 'Ktf': Ktf})

rh = 0.5
c_vap = rh* saturation_vapor_density(20.0)
dz_soil = 0.01
Ksoil = 0.5 # Wm-1K-1
gsoil = 0.5 / dz_soil # Wm-2K-1





#%% test solving surface temperature
forcing = {'Ta': 20.0, 'ust': 0.001, 'hsoil': -10.0, 'Tsoil': +10.0,
           'SWdn': 200.0, 'LWdn': 100.0, 'c_vap': c_vap}

res = surface_energy_balance(forcing, para, forcing['Tsoil'], gsoil)

#%% test pF-curve and hydraulic conductivity

w = np.linspace(0.03, 0.2, 50)

h = water_retention(para['pF'], theta=w)
wnew = water_retention(para['pF'], psi=h)

kliq = hydraulic_conductivity(para['pF'], w)
plt.figure()
plt.subplot(221);
plt.semilogx(-h, w, 'ro', -h, wnew, 'k-')
plt.subplot(222)
plt.loglog(-h, kliq, 'r-')

#%% test thermal conductivity

kh = thermal_conductivity(w)
print(kh)
T = 20.0
khm = molecular_diffusivity_porous_media(T, w, para['porosity'], scalar='co2')
plt.subplot(223)
plt.plot(w, 1e6*khm)

#%%
from moss.heat_and_water_flows import frozen_water, volumetric_heat_capacity, \
    moss_atm_conductance, air_density

w = 0.18, 
T = np.linspace(-20, 5, 100)
wliq, wice, gamma = frozen_water(T, w, fp=0.25)

plt.plot(T, wliq, '-', T, wice, '--')

#%%

from moss.heat_and_water_flows import solve_vapor

ctop = forcing['c_vap']
T = 20*np.exp(3.0*z)
h = g * water_retention(para['pF'], theta=0.15)
Kvap = para['Kvf']*g

E, c_vap = solve_vapor(z, h, T, Kvap, ctop)

#%%
    
    
t_solution = 5*3600.0 #s
T_ini = T = 20*np.exp(3.0*z)
T_ini = 10*g
initial_state =  {'volumetric_water_content': 0.15*g,
                  'volumetric_ice_content': 0.0*g,
                  'temperature': T_ini}

rh = 0.5
c_vap = rh* saturation_vapor_density(20.0)
dz_soil = 0.01
Ksoil = 0.5 # Wm-1K-1
gsoil = 0.5 / dz_soil # Wm-2K-1

forcing = {'Ta': 10.0, 'ust': 0.001, 'hsoil': -0.1, 'Tsoil': 5.0,
           'SWdn': 200.0, 'LWdn': 100.0, 'c_vap': c_vap, 'Gsurf': 0.0}

#%%
from moss.heat_water import water_heat_flow
fluxes, states, dto = water_heat_flow(t_solution, z, initial_state, forcing, para, steps=10)

#%%

#    pF = parameters['pF']
#    Ksat = parameters['Ksat']
#    fp = parameters['freezing_curve']
#    poros = parameters['porosity']
#    
#    Kvf = parameters['Kvt'] # [m2s-1], vapor conductivity with flow
#    Ktf = parameters['Kht'] # [m2s-1], heat conductivity with flow
#
#Pleurozium = {
#        "species": "Pleurozium schreberi",  # (Brid.) Mitt.
#        "ground_coverage": 0.0,
#        "height": 0.095,  # Soudziloskaia et al (2013)
#        "roughness_height": 0.01,
#        "leaf_area_index": 1.212,
#        "specific_leaf_area": 262.1,  # Bond-Lamberty and Gower (2007)
#        "bulk_density": 17.1,  # Soudziloskaia et al (2013)
#        "max_water_content": 10.0,
#        "min_water_content": 1.5,
#        "porosity": 0.98,
#        "photosynthesis": {
#            'Amax': 1.8,  # check from Rice et al. 2011 Bryologist Table 1
#            'b': 150.  # check from Rice et al. 2011 Bryologist Table 1
#        },
#        "respiration": {  # [2.0, 1.1]
#            'Q10': 2.0,  # check from Rice et al. 2011 Bryologist Table 1
#            'Rd10': 0.7  # check from Rice et al. 2011 Bryologist Table 1
#        },
#        'optical_properties': {  # [0.1102, 0.2909, 0.98]
#            'emissivity': 0.98,
#            'albedo_PAR': 0.1102,  # [-]
#            'albedo_NIR': 0.2909,  # [-]
#        },
#        "water_retention": {
#            # 'theta_s': 0.17,  # max_water_content * 1e-3 * bulk_density
#            # 'theta_r': 0.026,  # min_water_content * 1e-3 * bulk_density
#            'alpha': 0.13,
#            'n': 2.17,
#            'saturated_conductivity': 1.16e-8,  # [m s-1]
#            'pore_connectivity': -2.37
#        }
#}