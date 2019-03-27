# -*- coding: utf-8 -*-
"""
Planttype parameters: default parameterization which creates
"""

import numpy as np
#from .canopy import grid

grid = {'zmax': 30.0,  # heigth of grid from ground surface [m]
        'Nlayers': 100  # number of layers in grid [-]
        }

"""
planttypes (list):
    i. planttype_i (dict):
        'name' (str): name of planttype
        'LAImax' (list): leaf area index of planttype groups
        'lad' (array of lists): normalized leaf area density profiles of planttype groups
        'phenop' (dict): parameters for seasonal cycle of photosynthetic activity
            'Xo': initial delayed temperature [degC]
            'fmin': minimum photocapacity [-]
            'Tbase': base temperature [degC]
            'tau': time constant [days]
            'smax': threshold threshold temperature for full acclimation [degC]
        'laip' (dict): parameters for LAI seasonal dynamics
            'lai_min': minimum LAI, fraction of annual maximum [-]
            'lai_ini': initial LAI fraction, if None lai_ini = Lai_min * LAImax
            'DDsum0': degreedays at initial time [days]
            'Tbase': base temperature [degC]
            'ddo': degreedays at bud burst [days]
            'sdl':  daylength for senescence start [h]
            'sdur': duration of decreasing period [days]
        'photop' (dict): leaf gas-exchange parameters
            'Vcmax': maximum carboxylation velocity [umolm-2s-1]
            'Jmax': maximum rate of electron transport [umolm-2s-1]
            'Rd': dark respiration rate [umolm-2s-1]
            'alpha': quantum yield parameter [mol/mol]
            'theta': co-limitation parameter of Farquhar-model
            *'La': stomatal parameter (Lambda, m, ...) depending on model
            'g1':
            'g0': residual conductance for CO2 [molm-2s-1]
            'kn': used to scale photosynthetic capacity (vertical N gradient)
            'beta':  co-limitation parameter of Farquhar-model
            'drp': (dict): drought response parameters 
            'tresp' (dict): temperature sensitivity parameters
                'Vcmax': [Ha, Hd, dS]; activation energy [kJmol-1], deactivation energy [kJmol-1],  entropy factor [J mol-1]
                'Jmax': [Ha, Hd, dS];
                'Rd': [Ha]; activation energy [kJmol-1)]
        'leafp' (dict): leaf properties
            'lt': leaf lengthscale [m]
        'rootp' (dict): root zone properties
            'root_depth': root depth [m]
            'beta': shape parameter for root distribution model
            'RAI_LAI_multiplier': multiplier for total fine root area index (RAI = 2*LAImax)
            'fine_radius': fine root radius [m]
            'radial_K': maximum bulk root membrane conductance in radial direction [s-1]
"""

def lad_weibul(z, LAI, h, hb=0.0, b=None, c=None, species=None):
    """
    Generates leaf-area density profile from Weibull-distribution
    Args:
        z: height array (m), monotonic and constant steps
        LAI: leaf-area index (m2m-2)
        h: canopy height (m), scalar
        hb: crown base height (m), scalar
        b: Weibull shape parameter 1, scalar
        c: Weibull shape parameter 2, scalar
        species: 'pine', 'spruce', 'birch' to use table values
    Returns:
        LAD: leaf-area density (m2m-3), array \n
    SOURCE:
        Teske, M.E., and H.W. Thistle, 2004, A library of forest canopy structure for 
        use in interception modeling. Forest Ecology and Management, 198, 341-350. 
        Note: their formula is missing brackets for the scale param.
        Here their profiles are used between hb and h
    AUTHOR:
        Gabriel Katul, 2009. Coverted to Python 16.4.2014 / Samuli Launiainen
    """
    
    para = {'pine': [0.906, 2.145], 'spruce': [2.375, 1.289], 'birch': [0.557, 1.914]} 
    
    if (max(z) <= h) | (h <= hb):
        raise ValueError("h must be lower than uppermost gridpoint")
        
    if b is None or c is None:
        b, c = para[species]
    
    z = np.array(z)
    dz = abs(z[1]-z[0])
    N = np.size(z)
    LAD = np.zeros(N)

    a = np.zeros(N)

    # dummy variables
    ix = np.where( (z > hb) & (z <= h)) [0]
    x = np.linspace(0, 1, len(ix)) # normalized within-crown height

    # weibul-distribution within crown
    cc = -(c / b)*(((1.0 - x) / b)**(c - 1.0))*(np.exp(-((1.0 - x) / b)**c)) \
            / (1.0 - np.exp(-(1.0 / b)**c))

    a[ix] = cc
    a = np.abs(a / sum(a*dz))    

    LAD = LAI * a

    # plt.figure(1)
    # plt.plot(LAD,z,'r-')      
    return LAD

def lad_constant(z, LAI, h, hb=0.0):
    """
    creates constant leaf-area density distribution from ground to h.

    INPUT:
        z: height array (m), monotonic and constant steps
        LAI: leaf-area index (m2m-2)
        h: canopy height (m), scalar
        hb: crown base height (m), scalar
     OUTPUT:
        LAD: leaf-area density (m2m-3), array
    Note: LAD must cover at least node 1
    """
    if max(z) <= h:
        raise ValueError("h must be lower than uppermost gridpoint")

    z = np.array(z)
    dz = abs(z[1]-z[0])
    N = np.size(z)
    
#    # dummy variables
#    a = np.zeros(N)
#    x = z[z <= h] / h  # normalized heigth
#    n = np.size(x)
#
#    if n == 1: n = 2
#    a[1:n] = 1.0
    
    # dummy variables
    a = np.zeros(N)
    ix = np.where( (z > hb) & (z <= h)) [0]
    if ix.size == 0:
        ix = [1]

    a[ix] = 1.0
    a = a / sum(a*dz)
    LAD = LAI * a
    return LAD

# create grid and default plant types
z = np.linspace(0, grid['zmax'], grid['Nlayers'])  # grid [m] above ground

Pine = {
        'name': 'pine',                                         
        'LAImax': 3.0, #2.1,                                        # maximum annual LAI m2m-2 
        'lad': lad_weibul(z, LAI=1.0, b=0.4, c=2.7, h=15.0, hb=0.0, species='pine'),  # leaf-area density m2m-3
        'phenop': {                                             # cycle of photosynthetic activity
            'Xo': 0.0,
            'fmin': 0.1,
            'Tbase': -4.67,  # Kolari 2007
            'tau': 8.33,  # Kolari 2007
            'smax': 18.0  # Kolari 2014
            },
        'laip': {                                               # cycle of annual LAI
            'lai_min': 0.8,
            'lai_ini': None,
            'DDsum0': 0.0,
            'Tbase': 5.0,
            'ddo': 45.0,
            'ddmat': 250.0,
            'sdl': 12.0,
            'sdur': 30.0
            },
        'photop': {                                             # A-gs model
            'Vcmax': 45.0,
            'Jmax': 85.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 0.9,  # 0.023*Vcmax
            'tresp': {                      # temperature response parameters (Kattge and Knorr, 2007)
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

Understory = {
        'name': 'spruce',
        'LAImax': 0.5,
        'lad': lad_weibul(z, LAI=1.0, b=0.4, c=2.7, h=3.0, hb=0.0, species='underst'),
        'phenop': {
            'Xo': 0.0,
            'fmin': 0.1,
            'Tbase': -4.67,  # Kolari 2007
            'tau': 8.33,  # Kolari 2007
            'smax': 15.0  # Kolari 2014
            },
        'laip': {
            'lai_min': 0.8,
            'lai_ini': None,
            'DDsum0': 0.0,
            'Tbase': 5.0,
            'ddo': 45.0,
            'ddmat': 250.0,
            'sdl': 12.0,
            'sdur': 30.0
            },
        'photop': {
            'Vcmax': 55.0,
            'Jmax': 105.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 1.1,  # 0.023*Vcmax
            'tresp': {
                'Vcmax': [72., 200., 649.],  # (Kattge and Knorr, 2007)
                'Jmax': [50., 200., 646.],  # (Kattge and Knorr, 2007)
                'Rd': [33.0]
                },
            'alpha': 0.2,
            'theta': 0.7,
            #'La': 1600.0,
            'g1': 2.3,
            'g0': 5.0e-3,
            'kn': 0.5,
            'beta': 0.95,
            'drp': [0.39, 0.83, 0.31, 3.0]
            },
        'leafp': {
            'lt': 0.02,
            },
        'rootp': {
            'root_depth': 0.5,
            'beta': 0.943,
            'RAI_LAI_multiplier': 2.0,
            'fine_radius': 2.0e-3,
            'radial_K': 5.0e-8,
            }
        }

Deciduous = {
        'name': 'decidious',
        'LAImax': 1.2,
        'lad': lad_weibul(z, LAI=1.0, h=12.0, hb=0.5, species='birch'),
        'phenop': {
            'Xo': 0.0,
            'fmin': 0.01,
            'Tbase': -4.67,  # Kolari 2007
            'tau': 8.33,  # Kolari 2007
            'smax': 15.0  # Kolari 2014
            },
        'laip': {
            'lai_min': 0.1,
            'lai_ini': None,
            'DDsum0': 0.0,
            'Tbase': 5.0,
            'ddo': 45.0,
            'ddmat': 250.0,
            'sdl': 12.0,
            'sdur': 30.0
            },
        'photop': {
            'Vcmax': 40.0,
            'Jmax': 76.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 1.0,  # 0.023*Vcmax
            'tresp': {
                'Vcmax': [72., 200., 649.],  # (Kattge and Knorr, 2007)
                'Jmax': [50., 200., 646.],  # (Kattge and Knorr, 2007)
                'Rd': [33.0]
                },
            'alpha': 0.2,
            'theta': 0.7,
            #'La': 600.0,
            'g1': 4.0,
            'g0': 1.0e-2,
            'kn': 0.5,
            'beta': 0.95,
            'drp': [0.39, 0.83, 0.31, 3.0]
            },
        'leafp': {
            'lt': 0.05,
            },
        'rootp': {
            'root_depth': 0.5,
            'beta': 0.943,
            'RAI_LAI_multiplier': 2.0,
            'fine_radius': 2.0e-3,
            'radial_K': 5.0e-8,
            }
        }

Shrubs = {
        'name': 'shrubs',
        'LAImax': 0.7,
        'lad': lad_constant(z, LAI=1.0, h=0.5, hb=0.0),
        'phenop': {
            'Xo': 0.0,
            'fmin': 0.01,
            'Tbase': -4.67,  # Kolari 2007
            'tau': 8.33,  # Kolari 2007
            'smax': 15.0  # Kolari 2014
            },
        'laip': {
            'lai_min': 0.5,
            'lai_ini': None,
            'DDsum0': 0.0,
            'Tbase': 5.0,
            'ddo': 45.0,
            'ddmat': 250.0,
            'sdl': 12.0,
            'sdur': 30.0
            },
        'photop': {
            'Vcmax': 40.0,
            'Jmax': 76.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 0.8,  # 0.023*Vcmax
            'tresp': {
                'Vcmax': [72., 200., 649.],  # (Kattge and Knorr, 2007)
                'Jmax': [50., 200., 646.],  # (Kattge and Knorr, 2007)
                'Rd': [33.0]
                },
            'alpha': 0.2,
            'theta': 0.7,
            #'La': 600.0,
            'g1': 4.5,          
            'g0': 1.0e-2,
            'kn': 0.5,
            'beta': 0.95,
            'drp': [0.39, 0.83, 0.31, 3.0]
            },
        'leafp': {
            'lt': 0.05,
            },
        'rootp': {
            'root_depth': 0.3,
            'beta': 0.943,
            'RAI_LAI_multiplier': 2.0,
            'fine_radius': 2.0e-3,
            'radial_K': 5.0e-8,
            }
        }

planttypes = {'pine': Pine,
              'understory': Understory,
              'shrubs': Shrubs
              }
              #'spruce': Spruce,
              #'decidious': Deciduous,
              #'shrubs': Shrubs
              #