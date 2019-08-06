# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:05:33 2019

@author: slauniai

Try to replicate Elumeeva et al. 2011 J. Veget. Sci. experiment

"""

import numpy as np
import matplotlib as mpl
mpl.rc('image', cmap='coolwarm')
import matplotlib.pyplot as plt

from moss.MLM_Moss import MLM_Moss


def run_simulation(dt, z, para, initial_state, lbc, forcing, Nsteps):
    """ runs simulations, returns results and model instance """

    # create MLM_Moss instance
    Moss = MLM_Moss(z, para, initial_state)

    #%% test moss 
                
    c = len(Moss.Flow.z)
    d = len(Moss.z)
    
    #dt = 60.0
    #Nsteps = int(5*3600 / dt)
    
    results = {'T': np.ones((c, Nsteps))*np.NaN, 'H2O': np.ones((c, Nsteps))*np.NaN,
               'Tmoss': np.ones((d, Nsteps))*np.NaN, 'Wliq': np.ones((d, Nsteps))*np.NaN,
               'Wice': np.ones((d, Nsteps))*np.NaN, 'h': np.ones((d, Nsteps))*np.NaN,
               'LEz': np.ones((d, Nsteps))*np.NaN, 'Hz': np.ones((d, Nsteps))*np.NaN,
               'SWabs': np.ones((d, Nsteps))*np.NaN, 'LWabs': np.ones((d, Nsteps))*np.NaN,
               'Rnet': np.ones(Nsteps)*np.NaN, 'H': np.ones(Nsteps)*np.NaN, 'LE': np.ones(Nsteps)*np.NaN,
               'U': np.ones((c, Nsteps))*np.NaN, 'ust': np.ones((c, Nsteps))*np.NaN,
               'S': np.ones((d, Nsteps))*np.NaN, 'Sadv': np.ones((d, Nsteps))*np.NaN,
               'Trfall': np.ones(Nsteps)*np.NaN, 'Interc': np.ones(Nsteps)*np.NaN
              }  
    
    for n in range(0, Nsteps):
        #forc = forcing.iloc[start + n,:]
        forc = forcing.copy()
        state, canopyflx, T, H2O = Moss.run_timestep(dt, forc, lbc, sub_dt=60.0)
        Moss._update_state(state)
        
        results['T'][:, n] = T.copy()
        results['H2O'][:, n] = H2O.copy()
        results['Tmoss'][:, n] = Moss.T.copy()
        results['Wliq'][:, n] = Moss.Wliq.copy()
        results['Wice'][:, n] = Moss.Wice.copy()
        results['h'][:, n] = Moss.h.copy()
        
        results['SWabs'][:,n] = canopyflx['SWabs']
        results['LWabs'][:,n] = canopyflx['LWabs']
        results['LEz'][:,n] = canopyflx['LEz']
        results['Hz'][:,n] = canopyflx['Hz']
        results['U'][:,n] = Moss.Flow.U
        results['ust'][:,n] = Moss.Flow.ust
        
        results['Rnet'][n] = canopyflx['Rnet']
        results['H'][n] = canopyflx['H']
        results['LE'][n] = canopyflx['LE']
        results['S'][:,n] = canopyflx['S']
        results['Sadv'][:,n] = canopyflx['Sadv']
        results['Trfall'][n] = canopyflx['Trfall']
        results['Interc'][n] = canopyflx['Interc']
        
        print('n=',n )
        
    return results, Moss

def driver(p, Nsteps, wind_speed=None, dz=0.0025, dt=60.0, mosstype='feathermoss'):
    
    """
    ambient conditions in the lab experiment:
    """
    Pamb = 101300.0 # Pa
    Ta =  23.0 # -1, +3 degC
    RH = 46.0 # +/- 5%
    es = 611.0 * np.exp((17.502 * Ta) / (Ta + 240.97))  # Pa
    es = es / Pamb   
    H2Oa = RH/100 * es # mol/mol
    
    forcing = {'zenith_angle': 0.78, # 45degC
               'dir_par': 0.0, # 25.0,
               'dif_par': 90.0,
               'dir_nir': 0.0, 
               'dif_nir': 90.0, 
               'lw_in': 330.0, 
               'wind_speed': 0.1,
               'air_temperature': Ta,
               'h2o': H2Oa,
               'co2': 400.0,
               'air_pressure': 101300.0,
               'soil_temperature': 10.0,
               'precipitation': 0.0
               }
    
    
    lbc = {'heat': {'type': 'flux', 'value': 0.0},
           #'heat': {'type': 'temperature', 'value': 15.0},
           #'water': {'type': 'head', 'value': -0.01}
           'water': {'type': 'flux', 'value': 0.0}
          }
 
    # nominal parameters
    
    if mosstype == 'feathermoss':
        # pleurozium
        para0 = {'height': None,
                'lad': None,
                'bulk_density': 15.0, #15.0, #20.0,
                'max_water_content': 10.0,
                'min_water_content': 0.1,
                'pF':{'alpha': 0.13, 'n': 2.17, 'l': -2.37, 'Ksat': 1.2e-8},
                'porosity': 0.98,
                'length_scale': 0.005,
                'freezing_curve': 0.25,
                
                # radiation parameters
                'radiation': {'PAR_albedo': 0.2,
                              'NIR_albedo': 0.3,
                              'emissivity': 0.98,
                              'leaf_angle': 1.0,
                              'clump': 1.0,
                              'soil_albedo': 0.0,
                              'max_water_content': 10.0
                             },
                # flow parameters
                'flow': {'zref': 0.2,
                         'Sc': 1.0,
                         'gam': 0.1
                         }
                }
    if mosstype == 'sphagnum': 
        para0 = {'height': None,
                'lad': None,
                'bulk_density': 30.0,
                'max_water_content': 15.0,
                'min_water_content': 0.1,
                'pF':{'alpha': 0.10, 'n': 2.4, 'l': -2.37, 'Ksat': 1.2e-5},
                'porosity': 0.98,
                'length_scale': 0.005,
                'freezing_curve': 0.25,
                
                # radiation parameters
                'radiation': {'PAR_albedo': 0.2,
                              'NIR_albedo': 0.3,
                              'emissivity': 0.98,
                              'leaf_angle': 1.0,
                              'clump': 1.0,
                              'soil_albedo': 0.0,
                              'max_water_content': 20.0
                             },
                # flow parameters
                'flow': {'zref': 0.2,
                         'Sc': 1.0,
                         'gam': 0.1
                         }
                }        

    # create parameterset
    #z = np.arange(0, (p['height'] + dz), dz)
    z = np.linspace(0, p['height'], np.fix(p['height'] / dz) + 1)
    a = np.ones(np.shape(z)) 
    
    para = para0.copy()    
    para['height'] = p['height']
    
    if 'LAI' in p:
        para['lad'] = a / sum(a*dz) * p['LAI']
    if 'ladmean' in p:
        lai = p['ladmean'] * p['height']
        para['lad']  = a / sum(a*dz) * lai

    if 'bulk_density' in p:
        para['bulk_density'] = p['bulk_density']
    if 'max_water_content' in p:
        para['max_water_content'] = para['max_water_content']
    
    if wind_speed:
        forcing['wind_speed'] = wind_speed
        
    initial_state = {'temperature': a * forcing['air_temperature'],
                     'water_potential': a * -0.01, 
                     'Utop': 5.0,
                     'Ubot': 0.0
                     }
    
    # run
    res, model = run_simulation(dt, z, para, initial_state, lbc, forcing, Nsteps) 

    return res

def scenarios(Nsteps=600, mosstype='feathermoss'):
    
    h = np.arange(0.02, 0.1, 0.01)
    lai = np.linspace(1.0, 10, 10)
    #U = np.linspace(0.01, 1.0, 0.1)
    
    h_laic = []
    h_ladc = []
    for n in range(0, len(h)):
        p = {'height': h[n], 'LAI': 5.0}
        h_laic.append(driver(p, Nsteps))
        del p
        
        p = {'height': h[n], 'ladmean': 100.0}
        h_ladc.append(driver(p, Nsteps))
     
    r_lai = []
    
    for n in range(0, len(lai)):
        p = {'height': 0.05, 'LAI': lai[n]}
        r_lai.append(driver(p, Nsteps))
    
    return h_laic, h_ladc, r_lai

#%%

""" run for drying-hypothesis """

h_laic, h_ladc, r_lai = scenarios()

#%%
E1 = []
for k in range(0, len(h_laic)):
    E1.append(h_laic[k]['LE'][-1])

E2 = []
for k in range(0, len(h_laic)):
    E2.append(h_ladc[k]['LE'][-1])

E3 = []
for k in range(0, len(r_lai)):
    E3.append(r_lai[k]['LE'][-1])

x1 = np.arange(0.02, 0.1, 0.01)
x3 = np.linspace(1.0, 10, 10)

L = 1e3 * (3147.5 - 2.37 * (23.0 + DEG_TO_KELVIN)) # J kg-1 

plt.figure()
plt.plot(x1, 1e6*np.array(E1) / L, 'ko', label='LAI=5.0'); plt.ylabel('Evaporation rate (mgm-2s-1)'); plt.xlabel(' height (m)'); plt.title('LAI=5.0 m2m-2')
plt.plot(x1, 1e6*np.array(E2) / L, 'ro', label='$\Lambda_l$=100.0 m$^2$m$^{-3}$'); 
plt.legend()

#%%
#L = 1e3 * (3147.5 - 2.37 * (23.0 + DEG_TO_KELVIN)) # J kg-1
#E1 = []
#for k in range(0, len(a)):
#    E1.append(a[k]['LE'][-1] / L)
    
    #L = 1e3 * (3147.5 - 2.37 * (forcing['air_temperature'] + DEG_TO_KELVIN)) # J kg-1
#
##E = results['LE'] / L
#E = []
#for k in range(0, len(Res)):
#    E.append(1e6*Res[k]['LE'][-1] / L) # mg
#Res_lai = Res.copy()
##%% Loop over traits:
#allresults = []
#
#heights = np.arange(0.02, 0.08, 0.01)
#wind_speed = np.linspace(0.01, 2.0, 20)
#mean_lad = 80.0
#leaf_areas = np.arange(1.0, 8.0, 1.0)
#for nn in range(0, len(heights)):
#    print('nn=', nn)
#    # make grid and lad
#    dz = 0.0025 # grid size
#    height = heights[nn]
#    z = np.arange(0, (height + dz), dz)
#    LAI = 5.0 #leaf_areas[nn]
#    #LAI = mean_lad * height
#    a = np.ones(np.shape(z)) 
#    lad = a / sum(a*dz) * LAI
#    
#    # pleurozium
#    para = {'height': height,
#            'lad': lad,
#            'bulk_density': 15.0, #15.0, #20.0,
#            'max_water_content': 10.0,
#            'min_water_content': 0.2,
#            'pF':{'alpha': 0.13, 'n': 2.17, 'l': -2.37, 'Ksat': 1.2e-8},
#            'porosity': 0.98,
#            'length_scale': 0.005,
#            'freezing_curve': 0.25,
#            
#            # radiation parameters
#            'radiation': {'PAR_albedo': 0.2,
#                          'NIR_albedo': 0.3,
#                          'emissivity': 0.98,
#                          'leaf_angle': 1.0,
#                          'clump': 1.0,
#                          'soil_albedo': 0.0,
#                          'max_water_content': 10.0
#                         },
#            # flow parameters
#            'flow': {'zref': 0.2,
#                     'Sc': 1.0,
#                     'gam': 0.1
#                     }
#            }
#    
#    # sphagnum
#    #para = {'height': height,
#    #        'lad': lad,
#    #        'bulk_density': 30.0,
#    #        'max_water_content': 15.0,
#    #        'min_water_content': 1.5,
#    #        'pF':{'alpha': 0.10, 'n': 2.4, 'l': -2.37, 'Ksat': 1.2e-4},
#    #        'porosity': 0.98,
#    #        'length_scale': 0.005,
#    #        'freezing_curve': 0.25,
#    #        
#    #        # radiation parameters
#    #        'radiation': {'PAR_albedo': 0.1,
#    #                      'NIR_albedo': 0.3,
#    #                      'emissivity': 0.98,
#    #                      'leaf_angle': 1.0,
#    #                      'clump': 1.0,
#    #                      'soil_albedo': 0.0,
#    #                      'max_water_content': 20.0
#    #                     },
#    #        # flow parameters
#    #        'flow': {'zref': 0.2,
#    #                 'Sc': 2.0,
#    #                 'gam': 0.1
#    #                 }
#    #        }
#    
#    # initial state
#    initial_state = {'temperature': a * forcing['Ta'],
#                     'water_potential': a * -0.01, 
#                     'Utop': 5.0,
#                     'Ubot': 0.0
#                     }
#    
#
#
##%%
#L = 1e3 * (3147.5 - 2.37 * (forcing['air_temperature'] + DEG_TO_KELVIN)) # J kg-1
#
##E = results['LE'] / L
#E = []
#for k in range(0, len(Res)):
#    E.append(1e6*Res[k]['LE'][-1] / L) # mg
#Res_lai = Res.copy()

##%%
#
#t = np.arange(0, Nsteps) * dt / 3600
#
#fig = plt.figure()
#fig.set_size_inches(13.0, 16.0)
#
##plt.figure(12)
#plt.subplot(421); 
#plt.plot(t, results['Rnet'], '-', label='Rnet'); plt.plot(t, results['LE'], '-', label='LE'); 
#plt.plot(t, results['H'], '-', label='H'); plt.legend()
#
#plt.subplot(422); plt.plot(t, np.mean(results['Wliq'], axis=0), 'b-', label='Wbulk'); plt.ylabel('m3m-3')
#plt.legend()
#ax2 = plt.gca().twinx()
#ax2.plot(t, np.mean(results['Tmoss'], axis=0), 'r-', label='Tmoss ave'); plt.ylabel('degC')
#ax2.plot(t, results['T'][-1,:], 'k-', label='Ta top')
#ax2.plot(t,  results['T'][Moss.Nlayers,:], 'k:', label='Ta hc');
##ax2.plot(t,  results['T'][0,:], 'k--', label='Ta bot'); 
#
##ax2.plot(t,  np.mean(results['T'][0:Moss.Nlayers,:], axis=0), 'r:', label='Ta bulk');
#
#plt.legend()
#
#plt.subplot(423); plt.contourf(t, Moss.z, results['LEz']);plt.colorbar(); plt.title('LE [Wm-2]'); plt.ylabel('z [m]')
#plt.subplot(424); plt.contourf(t, Moss.z, results['Hz']); plt.colorbar(); plt.title('H [Wm-2]'); plt.ylabel('z [m]')
#plt.subplot(425); plt.contourf(t, Moss.z, results['Wliq']); plt.colorbar(); plt.title('Wliq [m3m-3]'); plt.ylabel('z [m]')
#plt.subplot(426); plt.contourf(t, Moss.z, results['Tmoss']); plt.colorbar(); plt.title('Tmoss [degC]'); plt.ylabel('z [m]')
#plt.subplot(427); plt.contourf(t, Moss.Flow.z, 1e3*results['H2O']); plt.colorbar(); plt.title('H2O [ppth] air'); plt.ylabel('z [m]'); plt.xlabel('hours')
#plt.plot([t[0], t[-1]], [Moss.z[-1], Moss.z[-1]], 'k--')
#plt.subplot(428); plt.contourf(t, Moss.Flow.z, results['T']); plt.colorbar(); plt.title('Tair [degC]'); plt.ylabel('z [m]'); plt.xlabel('hours')
#plt.plot([t[0], t[-1]], [Moss.z[-1], Moss.z[-1]], 'k--')
#    
#fig = plt.figure()
#fig.set_size_inches(13.0, 16.0)
#dy = results['Tmoss'] - results['T'][0:Moss.Nlayers,:]
#plt.subplot(421); plt.contourf(t, Moss.z, dy); plt.colorbar(); plt.title('Tm -Ta [degC]'); plt.ylabel('z [m]')
#dT = results['Tmoss'] - np.mean(results['Tmoss'], axis=0)
#plt.subplot(422); plt.contourf(t, Moss.z, dT); plt.colorbar(); plt.title('Tmoss - Tmoss, ave[degC]'); plt.ylabel('z [m]')
#plt.subplot(423); plt.contourf(t, Moss.z, results['SWabs']); plt.colorbar(); plt.title('SW abs'); plt.ylabel('z [m]')
#plt.subplot(424); plt.contourf(t, Moss.z, results['LWabs']); plt.colorbar(); plt.title('LW abs'); plt.ylabel('z [m]')
#plt.subplot(425); plt.contourf(t, Moss.z, results['LWabs']); plt.colorbar(); plt.title('LW abs'); plt.ylabel('z [m]')
#
## gravimetric water content
#plt.subplot(426);
#WATER_DENSITY = 1.0e3
#yy = results['Wliq'] * WATER_DENSITY / Moss.bulk_density
#plt.contourf(t, Moss.z, yy); plt.colorbar(); plt.title('water content [g/g]'); plt.ylabel('z [m]')
#
## bulk evaporation rate and bulk water content
#DEG_TO_KELVIN = 273.15
#
#L = 1e3 * (3147.5 - 2.37 * (forcing['air_temperature'] + DEG_TO_KELVIN)) # J kg-1
#
#E = results['LE'] / L
#
#plt.subplot(427)
#plt.plot(t, 1e6*E, 'r-'); plt.ylabel('E mgm-2 s-1')
#plt.subplot(428);
#plt.plot(t, np.mean(results['Wliq'], axis=0) * WATER_DENSITY / Moss.bulk_density)
#%% test radiation module
#g = np.ones(np.shape(Moss.Radiation.Lz))
#rad_forcing = {'zenith_angle': 0.5, 'dir_par': 150.0, 'dif_par': 150.0,
#               'dir_nir': 150.0, 'dif_nir': 150.0, 'lw_in': 300.0,
#               'soil_temperature': 10.0, 'T': g*10.0, 'water_content': np.mean(Moss.water_content)}
#
#SWabs, PARabs, PARz, PAR_alb, NIR_alb, f_sl = Moss.Radiation.sw_profiles(rad_forcing)
##SWabs = np.zeros(np.shape(SWabs))
#LWabs, LWdn, LWup, gr = Moss.Radiation.lw_profiles(rad_forcing)
##LWabs = np.zeros(np.shape(LWabs))
#
##
#dt = 20*1800.0
#Pamb = 101300.0
#rh = 0.5
#Ta = 20.0
#es = 611.0 * np.exp((17.502 * Ta) / (Ta + 240.97))  # Pa
#es = es / Pamb   
#H2Oa = rh * es # mol/mol
#Uo =0.3
#
#U = Uo * np.exp(Moss.LAI / 2 * (z / Moss.height - 1.0))
#
#ebal_forcing = {'SWabs': SWabs, 'LWabs': LWabs, 'T': Ta, 'U': U, 'H2O': H2Oa, 'Pamb': Pamb,
#                'radiative_conductance': gr, 'soil_temperature': 10.0, 'lw_in': 300.0}
##Ts = Moss.T
##S, E, H, LE, SWabs, LWabs, heat_fluxes, water_fluxes, state = Moss.solve_heat(dt, ebal_forcing, Ts, Moss.Wliq, Moss.Wice)
