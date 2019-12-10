# -*- coding: utf-8 -*-
"""
Defines all parameters for running mxl-apes!

Created on Fri Apr  5 20:38:51 2019

@author: slauniai
"""
import numpy as np
from tools.utilities import lad_weibul #, lad_constant

gpara = {
    # 'pyAPES_path': r'c:\repositories\pyAPES_Samuli',
    'dt0' : 1800.0,  # timestep in forcing data file [s]
    'dt': 1800.0, # model timestep [s]
    'start_time' : "2014-07-03 08:00",  # start time of simulation [yyyy-mm-dd]
    'end_time' : "2014-07-03 18:00",  #"2018-01-01",  # end time of simulation [yyyy-mm-dd]
    'forc_filename' : "forcing/FIHy_mxl_forcing_2014.dat",  # 'Hyde_forcing_1997_2016.csv', # forcing data file*
    'results_directory':'results/',
    'outputs': [
              ['forcing_Ta', 'above canopy air temperature [degC]', ('date', 'simulation')],
              ['forcing_Prec', 'precipitation [m s-1]', ('date', 'simulation')],
              ['forcing_P', 'ambient pressure [Pa]', ('date', 'simulation')],
              ['forcing_h2o','H2O concentration [mol mol-1]', ('date', 'simulation')],
              ['forcing_co2','CO2 concentration [ppm]', ('date', 'simulation')],
              ['forcing_U','wind speed [m s-1]', ('date', 'simulation')],
              ['forcing_ust','friction velocity [m s-1]', ('date', 'simulation')],
              #
              ['canopy_WMA_assumption','WMA assumed (1=True, 0=False)', ('date', 'simulation')],
              
              # canopy averages & sums
              ['canopy_LAI','canopy LAI [m2 m-2]', ('date', 'simulation')],
              ['canopy_phenostate','canopy phenological state [-]', ('date', 'simulation')],
              ['canopy_interception', 'canopy interception [m s-1]', ('date', 'simulation')],
              ['canopy_interception_storage', 'canopy interception storage [m]', ('date', 'simulation')],
              ['canopy_evaporation', 'evaporation from interception storage [m s-1]', ('date', 'simulation')],
              ['canopy_condensation', 'condensation to canopy interception storage [m s-1]', ('date', 'simulation')],
              ['canopy_condensation_drip', 'condensation to canopy that drips [m s-1]', ('date', 'simulation')],
              ['canopy_transpiration','transpiration [m s-1]', ('date', 'simulation')],
              ['canopy_SH', 'sensible heat flux [W m-2]', ('date', 'simulation')],
              ['canopy_LE', 'latent heat flux [W m-2]', ('date', 'simulation')],
              ['canopy_SWnet', 'net shortwave radiation [W m-2]', ('date', 'simulation')],
              ['canopy_LWnet', 'net longwave radiation [W m-2]', ('date', 'simulation')],
              ['canopy_NEE', 'net ecosystem exchage [umol m-2 s-1]', ('date', 'simulation')],
              ['canopy_GPP', 'ecosystem gross primary production [umol m-2 s-1]', ('date', 'simulation')],
              ['canopy_TER', 'ecosystem respiration [umol m-2 s-1]', ('date', 'simulation')],
              
              # canopy layer profiles
              ['canopy_h2o','H2O concentration [mol mol-1]', ('date', 'simulation', 'canopy')],
              ['canopy_co2','CO2 concentration [ppm]', ('date', 'simulation', 'canopy')],
              ['canopy_temperature','air temperature []degC]', ('date', 'simulation', 'canopy')],
              ['canopy_wind_speed','canopy wind speed [m s-1]', ('date', 'simulation', 'canopy')],
              ['canopy_friction_velocity','canopy friction velocity [m s-1]', ('date', 'simulation', 'canopy')],
              ['canopy_lad','leaf area density [m3 m-2]', ('date', 'simulation', 'canopy')],              
              
              ['canopy_Tleaf', 'leaf temperature [degC]', ('date', 'simulation', 'canopy')],
              ['canopy_Tleaf_wet', 'wet leaf temperature [degC]', ('date', 'simulation', 'canopy')],
              ['canopy_Tleaf_sl', 'layer sunlit leaf temperature [degC]', ('date', 'simulation', 'canopy')],
              ['canopy_Tleaf_sh', 'layer shaded leaf temperature [degC]', ('date', 'simulation', 'canopy')],
              ['canopy_sunlit_fraction','fraction of sunlit leafs [-]', ('date', 'simulation', 'canopy')],
              ['canopy_leaf_net_LW', 'net leaf longwave radiation [W m-2]', ('date', 'simulation', 'canopy')],
              ['canopy_leaf_SW_absorbed', 'leaf absorbed shortwave radiation [W m-2]', ('date', 'simulation', 'canopy')],
              ['canopy_PARdn', 'downwdard PAR [W m-2 ground]', ('date', 'simulation', 'canopy')],
              ['canopy_PARup', 'updard PAR [W m-2 ground]', ('date', 'simulation', 'canopy')],
              ['canopy_NIRdn', 'downwdard INR [W m-2 ground]', ('date', 'simulation', 'canopy')],
              ['canopy_NIRup', 'updard NIR [W m-2 ground]', ('date', 'simulation', 'canopy')], 
              ['canopy_LWdn', 'downwdard LW [W m-2 ground]', ('date', 'simulation', 'canopy')],
              ['canopy_LWup', 'updard LW [W m-2 ground]', ('date', 'simulation', 'canopy')],                      
              ['canopy_throughfall', 'throughfall to moss or snow [m s-1]', ('date', 'simulation')],
              ['canopy_evaporation_ml', 'evaporation from interception storage (condensation incl.) [m s-1]', ('date', 'simulation', 'canopy')],
              ['canopy_throughfall_ml', 'throughfall within canopy [m s-1]', ('date', 'simulation', 'canopy')],
              ['canopy_condensation_drip_ml', 'condensation drip within canopy [m s-1]', ('date', 'simulation', 'canopy')],
              ['canopy_co2_flux', 'co2 flux [umol m-2 s-1]', ('date', 'simulation', 'canopy')],
              ['canopy_latent_heat_flux', 'latent heat flux [W m-2]', ('date', 'simulation', 'canopy')],
              ['canopy_sensible_heat_flux', 'sensible heat flux [W m-2]', ('date', 'simulation', 'canopy')],
              
              # plant-type specific outputs
              ['canopy_pt_transpiration', 'transpiration [m s-1]', ('date', 'simulation', 'planttype')],
              ['canopy_pt_gpp', 'gross primary production [umol m-2 s-1]', ('date', 'simulation', 'planttype')],
              ['canopy_pt_respiration', 'dark respiration [umol m-2 s-1]', ('date', 'simulation', 'planttype')],
              ['canopy_pt_stomatal_conductance_h2o', 'stomatal conductance for H2O [mol m-2 leaf s-1]', ('date', 'simulation', 'planttype')],
              ['canopy_pt_boundary_conductance_h2o', 'boundary layer conductance for H2O [mol m-2 leaf s-1]', ('date', 'simulation', 'planttype')],
              ['canopy_pt_leaf_internal_co2', 'leaf internal CO2 mixing ratio [mol mol-1]', ('date', 'simulation', 'planttype')],
              ['canopy_pt_leaf_surface_co2', 'leaf surface CO2 mixing ratio [mol mol-1]', ('date', 'simulation', 'planttype')],
                  
              # mxl outputs
              # variables and their units
              ['mxl_h', 'mxl height [m]', ('date', 'simulation')],
              ['mxl_h_lcl', 'lcl height [m]', ('date', 'simulation')],
              ['mxl_T_lcl', 'lcl temperature [K]', ('date', 'simulation')],
              ['mxl_theta', 'mxl potential temperature [K]', ('date', 'simulation')],
              ['mxl_q', 'mxl specific humidity [kg/kg]', ('date', 'simulation')],
              ['mxl_thetav', 'mxl virtual potential temperature [K]', ('date', 'simulation')],
              ['mxl_ca', ' mxl CO2 mixing ratio [ppm]', ('date', 'simulation')],
              ['mxl_Ws', 'subsidene velocity [ms-1]', ('date', 'simulation')],
              ['mxl_wstar', 'convective velocity scale [ms-1]', ('date', 'simulation')],
              ['mxl_sigmaw', 'turbulent velocity scale [ms-1]', ('date', 'simulation')],
              ['mxl_u', ' mxl horizontal wind speed [ms-1]', ('date', 'simulation')],
              ['mxl_U', 'mxl wind speed [ms-1]', ('date', 'simulation')],
              ['mxl_vpd', 'surface vapor pressure deficit [kPa]', ('date', 'simulation')],
              ['mxl_rh', 'surface relative humidity [-]', ('date', 'simulation')],
              ['mxl_Psurf', 'surface pressure [kPa]', ('date', 'simulation')],
            
              # entrainment zone
              ['mxl_Pe', 'entrainment zone pressure [kPa]', ('date', 'simulation')],
              ['mxl_Te', 'entrainment zone temperature [K]', ('date', 'simulation')],
              ['mxl_theta_jump', 'potential temperature jump [K]', ('date', 'simulation')],
              ['mxl_q_jump', 'specific humidity jump [kg/kg]', ('date', 'simulation')],
              ['mxl_ca_jump', 'CO2 jump [ppm]', ('date', 'simulation')],
              ['mxl_thetav_jump', 'virtual potential temperature jump [K]', ('date', 'simulation')],
            
              # surface forcing to mxl
              ['mxl_wthetas', 'surface kinematic heat flux [Kms-1]', ('date', 'simulation')],
              ['mxl_wqs', 'surface kinematic moisture flux [kg kg-1 ms-1]', ('date', 'simulation')],
              ['mxl_wcs', 'surface kinematic CO2 flux [ppm ms-1]', ('date', 'simulation')],
              ['mxl_ust', 'surface friction velocity [ppm ms-1]', ('date', 'simulation')],
              ]
    }

# ---  mxl model parameters and initial state    

mxlpara = {'dt': 1800.0, # s 
           'f': 1e-4,  # s-1
           'beta': 0.2, # closure constant
           'divU': 0.0, # large-scale subsidence due horizontal wind divergence s-1
           'ctr': {'Wind': True}
            }

mxl_ini = {'h': 100.,           # m
           'theta': 288.0,      # K
           'q': 8.0e-3,         # kg kg-1
           'ca': 422.0,         # ppm
           'theta_jump': 1.0,   # K
           'gamma_theta': 6e-3, # K m-1
           'q_jump': -1.0e-3,   # kg kg-1
           'gamma_q': -1.45e-6, # kg kg-1 m-1
           'ca_jump': -40.0,    # ppm
           'gamma_ca': 0.0,     # ppm m-1
           'u': 5.0,            # m s-1
           'u_jump': 8.0,       # m s-1, geostrophic wind is u_jump + u
           'gamma_u': 0.0,      # s-1
           'Psurf': 101.3       # kPa
          }

""" --- compile canopy model parameters ---"""

# site location
loc = {'lat': 61.51,  # latitude
       'lon': 24.0  # longitude
       }

# grid
grid = {'zmax': 30.0,  # heigth of grid from ground surface [m]
        'Nlayers': 100  # number of layers in grid [-]
        }

# --- control flags (True/False) ---
ctr = {'Eflow': True,  # ensemble flow
       'WMA': False, # well-mixed assumption
       'Ebal': True,  # computes leaf temperature by solving energy balance
       'WaterStress': None, #'Rew',  # Rew or PsiL or None
       'seasonal_LAI': False,  # account for seasonal LAI dynamics
       'pheno_cycle': False  # account for phenological cycle
       }

# --- micrometeo ---
micromet = {'zos': 0.01,  # forest floor roughness length [m]  -- not used?
            'dPdx': 0.00, #0.01,  # horizontal pressure gradient
            'Cd': 0.15,  # drag coefficient
            'Utop': 5.0,  # U
            'Ubot': 0.0,  # m/s, no-slip
            'Sc': {'T': 1.0, 'H2O': 1.0, 'CO2': 1.0}  # turbulent Schmidt numbers
            }

# --- radiation ---
radiation = {'clump': 0.7,  # clumping index [-]
             'leaf_angle': 1.0,  # leaf-angle distribution [-]
             'Par_alb': 0.12,  # shoot Par-albedo [-]
             'Nir_alb': 0.55,  # shoot NIR-albedo [-]
             'leaf_emi': 0.98,
             'SWmodel': 'ZHAOQUALLS', #'SPITTERS'
             'LWmodel': 'ZHAOQUALLS', #'FLERCHINGER'
             }

# --- interception ---
interception = {'wmax': 0.2e-03,  # maximum interception storage capacity for rain [m per unit of LAI]  - Watanabe & Mizunani coniferous trees
                'wmaxsnow': 1.6e-03,  # maximum interception storage capacity for snow [m per unit of LAI]
                'w_ini': 0.0,  # initial canopy storage [m]
                'Tmin': 0.0,  # temperature below which all is snow [degC]
                'Tmax': 1.0,  # temperature above which all is water [degC]
                'leaf_orientation': 0.5 # leaf orientation factor for randomdly oriented leaves
                }

# planttype
z = np.linspace(0, grid['zmax'], grid['Nlayers'])  # grid [m] above ground

pt1 = { 'name': 'generic_tree',                                         
        'LAImax': 3.0, #2.1,  # maximum annual LAI m2m-2 
        'lad': lad_weibul(z, LAI=1.0, b=0.4, c=2.7, h=15.0, hb=0.0, species='pine'),  # leaf-area density m2m-3
        'phenop': {   # cycle of photosynthetic activity
            'Xo': 0.0,
            'fmin': 0.1,
            'Tbase': -4.67,  # Kolari 2007
            'tau': 8.33,  # Kolari 2007
            'smax': 18.0  # Kolari 2014
            },
        'laip': {    # cycle of annual LAI
            'lai_min': 0.8,
            'lai_ini': None,
            'DDsum0': 0.0,
            'Tbase': 5.0,
            'ddo': 45.0,
            'ddmat': 250.0,
            'sdl': 12.0,
            'sdur': 30.0
            },
        'photop': { # A-gs model
            'Vcmax': 45.0,
            'Jmax': 85.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 0.9,  # 0.023*Vcmax
            'tresp': { # temperature response parameters (Kattge and Knorr, 2007)
                'Vcmax': [72., 200., 649.],
                'Jmax': [50., 200., 646.],
                'Rd': [33.0]
                },
            'alpha': 0.2,   # quantum efficiency parameter -
            'theta': 0.7,   # curvature parameter
            #'La': 1600.0,  # marginal wue for Optimality-model
            'g1': 2.1,      # stomatal slope kPa^(0.5)
            'g0': 5.0e-3,   # residual conductance mol m-2 s-1
            'kn': 0.5,      # nitrogen attenuation coefficient -
            'beta': 0.95,   # co-limitation parameter -
            'drp': [0.39, 0.83, 0.31, 3.0] # Rew-based drought response
            },
        'leafp': {
            'lt': 0.02,     # leaf length scale m
            },
        'rootp': {
            'root_depth': 0.5,
            'beta': 0.943,
            'RAI_LAI_multiplier': 2.0,
            'fine_radius': 2.0e-3,
            'radial_K': 5.0e-8,
            }
        }

# --- ground surface (for testing, include soil model if needed)
ground = {'soildepth': 10.0,
          'temperature': 10.0, 
          'emissivity': 0.98,
          'albedo': {'PAR':0.05, 'NIR': 0.2}
         }

# compile all into single dict imported from test_mxl_apes

cpara = {'loc': loc,
         'ctr': ctr,
         'grid': grid,
         'radiation': radiation,
         'flow': micromet,
         'interception': interception,
         'planttypes': {'plant1': pt1},
         'ground': ground
         }



logging_configuration = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
                'default': {'format': '%(asctime)s %(levelname)s %(name)s %(message)s'},
                'model': {'format': '%(levelname)s %(name)s %(funcName)s %(message)s'},
                },
        'handlers': {
                'console': {
                        'class' : 'logging.StreamHandler',
                        'formatter': 'model',
                        'level': 'INFO'  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        },
                'file': {
                        'class': 'logging.FileHandler',
                        'level': 'DEBUG',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'formatter': 'model',
                        'filename': 'pyAPES.log',
                        'mode': 'w',  # a == append, w == overwrite
                        },
                },
        'loggers': {
                'pyAPES': {
                        'handlers': ['file', 'console'],
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagate': True,
                        },
                'canopy':{
                        'handlers': ['file', 'console'],
                        'level': 'DEBUG',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagate': True,
                        },
                'soil':{
                        'handlers': ['file', 'console'],
                        'level': 'DEBUG',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagate': True,
                        },
                },
        }

parallel_logging_configuration = {
        'version': 1,
        'formatters': {
                'default': {
                    'class': 'logging.Formatter',
                    'format': '%(asctime)s %(levelname)s %(name)s %(message)s'},
                'model': {
                    'class': 'logging.Formatter',
                    'format': '%(process)d %(levelname)s %(name)s %(funcName)s %(message)s'},
                },
        'handlers': {
                'console': {
                        'class' : 'logging.StreamHandler',
                        'formatter': 'model',
                        'level': 'INFO'  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        },
                'pyAPES_file': {
                        'class': 'logging.FileHandler',
                        'level': 'DEBUG',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'formatter': 'model',
                        'filename': 'pyAPES.log',
                        'mode': 'w',  # a == append, w == overwrite
                        },
                'parallelAPES_file': {
                        'class': 'logging.FileHandler',
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'formatter': 'default',
                        'filename': 'parallelAPES.log',
                        'mode': 'w',  # a == append, w == overwrite
                        },
        },
        'loggers': {
                'pyAPES': {
                        #'handlers': ['file'],
                        'level': 'INFO',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
                        'propagate': True,
                        },
        #        'canopy':{
        #                #'handlers': ['file'],
        #                'level': 'DEBUG',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
        #                },
        #        'soil':{
        #                #'handlers': ['file'],
        #                'level': 'DEBUG',  # CRITICAL, ERROR, WARNING, INFO, DEBUG
        #                },
                },
        'root': {
                'level': 'DEBUG',
                'handlers': ['console', 'parallelAPES_file']
                }
        }