# -*- coding: utf-8 -*-
"""
CANOPY MODEL PARAMETERS
"""
from copy import deepcopy
from parameter_utils import lad_profiles

# initialize dictionary to store parameters
cpara = {}

# --- control flags ---
ctr = {'multilayer_model': False,  # compute in multilayer mode
       'seasonal_LAI': True,  # account for seasonal LAI dynamics
       'pheno_cylcle': True  # account for phenological cycle
       }

# --- site location ---
loc = {'lat': 61.4,  # latitude
       'lon': 23.7  # longitude
       }

# --- aerodynamics ---
aero = {'w': 0.01,  # leaf length scale [m]
        'zmeas': 2.0,  # wind speed measurement height above canopy [m]
        'zg': 0.5,  # height above ground where Ug is computed [m]
        'zos': 0.01  # forest floor roughness length [m]
        }

# --- interception and snowmodel ---  SADANNAN KORJAUSKERTOIMET?
interc_snow = {'wmax': 0.0005,  # maximum interception storage capacity for rain [m per unit of LAI]
               'wmaxsnow': 0.004,  # maximum interception storage capacity for snow [m per unit of LAI]
               'kmelt': 2.8934e-08,  # Melting coefficient [m degC-1 s-1]
               'kfreeze': 5.79e-09,  # Freezing  coefficient [m degC-1 s-1]
               'retention': 0.05,  # max fraction of liquid water in snow [-]
               'Tmin': 0.0,  # temperature below which all is snow [degC]
               'Tmax': 1.0,  # temperature above which all is water [degC]
               'Tmelt': 0.0,  # temperature when melting starts [degC]
               'w_ini': 0.0,  # initial canopy storage [m]
               'swe_ini': 0.0  # initial snow water equivalent [m]
               }

# --- default values for plant characteristics ---
plant_default = {'phenop': {  # --- seasonal cycle of phenology ---
                            'Xo': 0.0,  # initial delayed temperature [degC]
                            'fmin': 0.01,  # minimum photocapacity [-]
                            'Tbase': -4.67,  # base temperature [degC]
                            'tau': 8.33,  # time constant [days]
                            'smax': 18.5  #threshold for full acclimation [degC]
                            },
                'laip': {  # --- leaf-area seasonal dynamics ---
                         'lai_min': 0.1,  # minimum LAI, fraction of annual maximum [-]
                         'lai_ini': None,  # initial LAI fraction, if None lai_ini = Lai_min * LAImax
                         'DDsum0': 0.0,  # degreedays at initial time [days]
                         'Tbase': 5.0,  # base temperature [degC]
                         'ddo': 45.0,  # degreedays at bud burst [days]
                         'ddur': 23.0,  # duration of recovery period [days]
                         'sso': 240.0,  # start doy of decrease, based on daylength
                         'sdur': 30.0, # duration of decreasing period [days]
                         },
                'photop': {  # --- leaf gas-exchange parameters ---
                        'Vcmax': None,  # maximum carboxylation velocity [umolm-2s-1]
                        'Jmax': None,  # maximum rate of electron transport [umolm-2s-1]
                        'Rd': None,  # dark respiration rate [umolm-2s-1]
                        'alpha': 0.16,  # quantum yield parameter [mol/mol]
                        'theta': 0.7,  # co-limitation parameter of Farquhar-model
                        'La':None,  # stomatal parameter (Lambda, m, ...) depending on model
                        'm':None,
                        'g0': 1.0e-3,  # residual conductance for CO2 [molm-2s-1]
                        'kn': 0.6,
                        'beta': 0.95,  # co-limitation parameter of Farquhar-model
                        'drp': 0.7,
                        'tresp': {  # --- temperature sensitivity parameters ---
                                'Vcmax': [],  # [Ha, Hd, Topt]; activation energy [kJmol-1], deactivation energy [kJmol-1], optimum temperature [degC]
                                'Jmax': [],  # [Ha, Hd, Topt];
                                'Rd': [33.0]}  #[Ha]; activation energy [kJmol-1)]
                        },
                'leafp': {  # --- leaf properties ---
                        'lt': 0.02,  # leaf lengthscale [m]
                        'par_alb': 0.12,  # leaf Par albedo [-]
                        'nir_alb': 0.55,  # leaf Nir albedo [-]
                        'emi': 0.98  # leaf emissivity [-]
                        },
                }
pine = deepcopy(plant_default)  # initialize with default
spruce = deepcopy(plant_default)
decid = deepcopy(plant_default)
shrubs = deepcopy(plant_default)

# --- stand characteristics ---
# specify name and maximum leaf-area index, LAImax [m2/m2], and 
# adjust values of 'phenop' and 'laip' if default values not suitable
pine.update({'name': 'pine', 'LAImax': [2.1]})
pine['phenop'].update({'fmin': 0.1})
pine['laip'].update({'lai_min': 0.8})

spruce.update({'name': 'spruce', 'LAImax': [1.0]})
spruce['phenop'].update({'fmin': 0.1})
spruce['laip'].update({'lai_min': 0.8})

decid.update({'name': 'decid', 'LAImax': [1.0]})

shrubs.update({'name': 'shrubs', 'LAImax': [0.7]})
shrubs['laip'].update({'lai_min': 0.5})

if ctr['multilayer_model'] is False:
    """parameters for simple (not multilayer) model"""
    # --- physiology for calculation of transpiration and Efloor---
    phys_para = {'ga': 40.0,  # [m s-1]          MISSÄ KÄYTETÄÄN?
                 'q50': 50.0,  # half-sat. of leaf light response [W m-2]
                 'kp': 0.6,  # parameter to calculate fraction of Rnet at ground [-]
                 'f': 0.8,  # fraction of local Rnet available for evaporation at ground [-]
                 'rw': 0.20,  # transpiration moisture response parameter
                 'rwmin': 0.02}  # transpiration moisture response parameter
    pine['gsref'] = 1.6e-6  # [m s-1]
    spruce['gsref'] = 1.6e-6
    decid['gsref'] = 2.6e-6
    # --- parameters describing canopy ---
    canopy_para = {'hc': 16.0,  # canopy height [m]
                   'cf': 0.6}  # canopy closure [-]
    # add to parameter dictionary
    cpara.update({'phys_para': phys_para, 'canopy_para': canopy_para})
    # plant types to list
    plant_types = [pine, spruce, decid]  # shrubs?? hs?
else:
    """parameters for multilayer model"""
    # grid
    grid = {'zmax': 30.0,  # heigth of grid from ground surface [m]
            'Nlayers': 200  # number of layers in grid [-]
            }
    # normed leaf area density profiles
    dbhfile = "hyde_runkolukusarjat.txt"  # filepath to dbhfile (pine, spruce, decid)
    quantiles = [1.0]  # quantiles used in creating species stand lad profiles
    hs = 0.5  # height of understory shrubs [m]
    pine['lad'], spruce['lad'], decid['lad'], shrubs['lad'] = lad_profiles(
            grid, dbhfile, quantiles, hs, plot=True)
    # adjust leaf gas-exchange parameters
    gfact = 1.2  # coefficient for adjusting (?)
    pine['photop'].update({'Vcmax': 55.0, 'Jmax': 104.0, 'Rd': 1.3,
                           'La': 1600.0, 'm': gfact*2.5})
    pine['photop']['tresp'].update({'Vcmax': [78, 200.0, 650.0],
                                    'Jmax': [56, 200.0, 647.0]})
    spruce['photop'].update({'Vcmax': 70.0, 'Jmax': 133.0, 'Rd': 1.6,
                             'La': 1600.0, 'm': gfact*2.5})
    spruce['photop']['tresp'].update({'Vcmax': [53.2, 202.0, 640.3],  # Tarvainen et al. 2013 Oecologia
                                      'Jmax': [38.4, 202.0, 655.8]})
    decid['photop'].update({'Vcmax': 60.0, 'Jmax': 114.0, 'Rd': 1.3,
                            'La': 600.0, 'm': gfact*4.5})
    decid['photop']['tresp'].update({'Vcmax': [77.0, 200.0, 636.7],  # Medlyn et al 2002.
                                     'Jmax': [42.8, 200.0, 637.0]})
    shrubs['photop'].update({'Vcmax': 60.0, 'Jmax': 114.0, 'Rd': 1.3,
                            'La': 600.0, 'm': gfact*4.5, 'kn': 0.3})
    shrubs['photop']['tresp'].update({'Vcmax': [77.0, 200.0, 636.7],
                                     'Jmax': [42.8, 200.0, 637.0]})
    plant_types = [pine, spruce, decid, shrubs]

cpara.update({'ctr': ctr, 'loc': loc, 'aero': aero,
              'interc_snow': interc_snow, 'plant_types': plant_types})

#        for computing aerodynamic resistances  -- yksiköt?
#        self.zmeas = 2.0
#        self.zground = 0.5
#        self.zo_ground = 0.01
#        self.soilrp = 300.0