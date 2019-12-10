# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:21:16 2019

@author: slauniai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%Short and Long-wave radiaiton in plant canopies:
""" 
module canopy.radiation contains functions for radiation transfer. These
can be imported and used directly.
canopy.radiation.Radiation is class definition for 'canopy RT model' used in pyAPES.  
"""

# import modules
from canopy.radiation import Radiation
from tools.utilities import lad_weibul, lad_constant
from mxl.utils import read_mxl_forcing # function for reading forcing data

# Read forcing for demos
forcing = read_mxl_forcing("forcing/FIHy_mxl_forcing_2014.dat",
                           start_time="2014-07-03 08:00",
                           end_time="2014-07-03 18:00",
                           dt0=1800.0, # forcing data timestep (s)
                           dt=1800.0) # model timestep (s)

# grid
grid = {'zmax': 20.0,  # heigth of grid from ground surface [m]
        'Nlayers': 50  # number of layers in grid [-]
        }

# radiation model
radi_param = {'clump': 0.7,  # clumping index [-]
             'leaf_angle': 1.0,  # leaf-angle distribution [-]
             'Par_alb': 0.12,  # shoot Par-albedo [-]
             'Nir_alb': 0.55,  # shoot NIR-albedo [-]
             'leaf_emi': 0.98,
             'SWmodel': 'ZHAOQUALLS', #'SPITTERS'
             'LWmodel': 'ZHAOQUALLS', #'FLERCHINGER'
             }


# create leaf-area density profile from weibul-distribution
z = np.linspace(0, grid['zmax'], grid['Nlayers'])  # grid [m] above ground
dz = z[1] - z[0]

lad = lad_weibul(z, LAI=4.0, h=15.0, hb=2.0, b=0.9, c=2.7)
print('lai', sum(lad*dz))
plt.figure(1); plt.plot(lad, z); plt.ylabel('z'); plt.xlabel('lad (m2m-3)')

# create Radiation instance
radi = Radiation(radi_param, Ebal=True) # Ebal=True solves long-wave balance

print(radi.__dict__)
#%%
"""
run radi: save outputs to dict
"""

for k in len(forcing):
        
    # canopy model want forcing in following format
        canopy_forcing = {
            'zenith_angle': forcing['Zen'].iloc[k],
            'PAR': {'direct': forcing['dirPar'].iloc[k],
                    'diffuse': forcing['diffPar'].iloc[k]},
            'NIR': {'direct': forcing['dirNir'].iloc[k],
                    'diffuse': forcing['diffNir'].iloc[k]},
            'air_pressure': 101300.0,
            'precipitation': 0.0, # 1e-3*self.forcing['Prec'].iloc[k], # precip in m/s = 1e-3 kg/m2
            
            'lw_in': forcing['LW_in'].iloc[k],
            'wind_speed': forcing['U'].iloc[k],
            'air_temperature': forcing['T'].iloc[k],
            'h2o': forcing['H2O'].iloc[k],
            'co2': forcing['CO2'].iloc[k],
            
            # soil 1st node state: don't affect anything here
            'soil_temperature': 10.0, #degC
            'soil_volumetric_water': 0.3, #m3/m3
            'soil_water_potential': -10.0 # m
        }
        
        ground_params = {}
results =  Radiation.shortwave_profiles()

