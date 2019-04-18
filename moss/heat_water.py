# -*- coding: utf-8 -*-
"""
Coupled liquid - vapor - heat flow in porous media resembling moss canopy

Samuli Launiainen, Gaby Katul, Antti-Jussi Kieloaho, Kersti Haahti (2019)
Created on Mon Apr  8 01:07:25 2019

@author: slauniai
"""
import numpy as np
from scipy.optimize import fsolve
from tools.utilities import tridiag as thomas, spatial_average

#from moss_canopy import *

#: machine epsilon
EPS = np.finfo(float).eps

#: [J mol\ :sup:`-1`\ ], latent heat of vaporization at 20\ :math:`^{\circ}`\ C
LATENT_HEAT = 44100.0
#: [kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O
MOLAR_MASS_H2O = 18.015e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of CO\ :sub:`2`\
MOLAR_MASS_CO2 = 44.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of C
MOLAR_MASS_C = 12.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of air
MOLAR_MASS_AIR = 29.0e-3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of H\ :sub:`2`\ O
SPECIFIC_HEAT_H2O = 4.18e3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of organic matter
SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
#: [J mol\ :sup:`-1` K\ :sup:`-1`\ ], molar heat capacity of air at constant pressure
SPECIFIC_HEAT_AIR = 29.3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], mass-based heat capacity of air at constant pressure
SPECIFIC_HEAR_AIR_MASS = 1004.67  
#: [W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8
#: [-], von Karman constant
VON_KARMAN = 0.41
#: [K], zero degrees celsius in Kelvin
DEG_TO_KELVIN = 273.15
#: [K], zero degrees celsius in Kelvin
NORMAL_TEMPERATURE = 273.15
#: [mol m\ :sup:`-3`\ ], density of air at 20\ :math:`^{\circ}`\ C
AIR_DENSITY = 41.6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], kinematic viscosity of air at 20\ :math:`^{\circ}`\ C
AIR_VISCOSITY = 15.1e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], thermal diffusivity of air at 20\ :math:`^{\circ}`\ C
THERMAL_DIFFUSIVITY_AIR = 21.4e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of CO\ :sub:`2` at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of H\ :sub:`2`\ at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
#: [J mol\ :sup:`-1` K\ :sup:``-1], universal gas constant
GAS_CONSTANT = 8.314
#: [kg m\ :sup:`2` s\ :sup:`-1`\ ], standard gravity
GRAVITY = 9.81
#: [kg m\ :sup:`-3`\ ], water density
WATER_DENSITY = 1.0e3
#: [umol m\ :sup:`2` s\ :sup:`-1`\ ], conversion from watts to micromol
PAR_TO_UMOL = 4.56
#: [rad], conversion from deg to rad
DEG_TO_RAD = 3.14159 / 180.0
#: [umol m\ :sup:`-1`], O2 concentration in air
O2_IN_AIR = 2.10e5
# [-], von Karman constant
VON_KARMAN = 0.41
#: [J kg\ :sup:`-1`\ ], latent heat of freezing
LATENT_HEAT_FREEZING = 333700.0
#: [\ :math:`^{\circ}`\ C], freezing point of water
FREEZING_POINT_H2O = 0.0
#: [kg m\ :sup:`-3`\ ], densities
ICE_DENSITY = 917.0

#: [J m\ :sup:`-3`\ K \ :sup:`-1`\], thermal condutivities
K_WATER = 0.57
K_ICE = 2.2
K_AIR = 0.025
K_ORG = 0.25

#: volumetric heat capacieties  [J m\ :sup:`-3`\ K \ :sup:`-1`\]
CV_AIR = 1297.0  # air at 101kPa
CV_WATER = 4.18e6  # water
CV_ICE = 1.93e6  # ice
CV_ORGANIC = 2.50e6  # dry organic matter
#CV_MINERAL = 2.31e6  # soil minerals


def water_heat_flow(t_final, z, initial_state, forcing, parameters, steps=10):
    
    r""" Solves liquid water flow in 1-D using implicit, backward finite difference
    solution of Richard's equation.

    Args:
        t_final (float): solution timestep [s]
        grid (dict):
            'z': node elevations, top surface = 0.0 [m]
            'dz': thickness of computational layers [m]
            'dzu': distance to upper node [m]
            'dzl': distance to lower node [m]
        forcing (dict):
            q_sink (array): sink term from layers, e.g. evaporation sink [m3 m-3 s-1]
            #q_source (array): source due to rainfall interception per layer [m3 m-3 s-1]
        initial_state (dict):
            'water_potential': initial water potential [m]
            'volumetric_water_content' [m3 m-3]
            'volumetric_ice_content' [m3 m-3]
            'temperature' [degC]
        pF (dict): water retention parameters (van Genuchten)
            'ThetaS' (array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'ThetaR' (array):residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (array):air entry suction [cm\ :sup:`-1`]
            'n' (array):pore size distribution [-]
            'l' (array): pore connectivity
        Ksat (array): saturated hydraulic conductivity [m s-1]

        steps (int): initial number of subtimesteps used to proceed to 't_final'
    Returns:
        fluxes (dict): [m s-1]
            'evaporation'
            'drainage'
            'water_closure'
        state (dict):
            'water_potential': [m]
            'volumetric_water_content': [m3 m-3]
            'ground_water_level': [m]
            'pond_storage': [m]
        dto (float): timestep used for solving [s]

    References:
        vanDam & Feddes (2000): Numerical simulation of infiltration, evaporation and shallow
        groundwater levels with the Richards equation, J.Hydrol 233, 72-85.

    """
    global N, dz, dz2, pF, poros
    
    N = len(z) # zero at moss canopy top, z<= 0, constant steps
    dz = z[0] - z[1]
    dz2 = dz**2
    
    # initial and computational time step [s]
    dto = t_final / steps
    dt = dto # adjusted during solution
    dt_min = 1.0 # minimum timestep
    
    # convergence criteria
    crit_W = 1.0e-12  # moisture [m3 m-3] 
    crit_h = 1.0e-10  # head [m]
    crit_T = 1e-3 # [degC]
    crit_Wice = 1.0e-5 # ice content [m3 m-3]
    
    pF = parameters['pF']
    Ksat = parameters['Ksat']
    fp = parameters['freezing_curve']
    poros = parameters['porosity']
    
    Kvf = parameters['Kvf'] # [m2s-1], vapor conductivity with flow
    Ktf = parameters['Ktf'] # [m2s-1], heat conductivity with flow
           
    # cumulative fluxes for 0...t_final
    Evap = 0.0
    LEcum = 0.0
    Q_bot = 0.0
    Q_top = 0.0
    G_top = 0.0
    G_bot = 0.0
    Fheat = np.zeros(N+1)
    Eflx = np.zeros(N)
    
    # initial state    
    T_ini = initial_state['temperature']
    W_tot = initial_state['volumetric_water_content'] + initial_state['volumetric_ice_content']

    # Liquid and ice content, and dWliq/dTs
    W_ini, Wice_ini, gamma = frozen_water(T_ini, W_tot, fp=fp)
    h_ini = water_retention(pF, theta=W_ini) # h [m]
    del W_tot
    
    """ solve for t =[0; t_final] using implict iterative scheme """
    
    # variables updated during solution
    W = W_ini.copy()
    Wice = Wice_ini.copy()
    h = h_ini.copy()
    T = T_ini.copy()
    
    t = 0.0
    while t < t_final:
        # solution of previous timestep
        h_old = h.copy()
        W_old = W.copy()
        Wice_old = Wice.copy()
        T_old = T.copy()

        # state variables updated during iteration of time step dt
        h_iter = h.copy()
        W_iter = W.copy()
        Wice_iter = Wice.copy()
        T_iter = T.copy()
        
        del h, W, Wice, T
        
        # hydraulic condictivity [m s-1 = kg m-2 s-1]
        KLh = hydraulic_conductivity(pF, W_old, Ksat=Ksat)
        KLh = spatial_average(KLh, method='arithmetic')
        
        # bulk soil heat capacity [Jm-3K-1]
        CP_old = volumetric_heat_capacity(poros, W_old, Wice_old)
        
        # apparent heat capacity due to freezing/thawing [Jm-3K-1]
        A = ICE_DENSITY*LATENT_HEAT_FREEZING*gamma
        
        # thermal conductivity in gas and liquid phases [W m-2 K-1]
        Ktm = molecular_diffusivity_porous_media(T_old, W_old + Wice_old, poros, scalar='heat') 
        
        Kheat = air_density(T_old) * SPECIFIC_HEAR_AIR_MASS * (Ktm + Ktf) + \
                thermal_conductivity(W_old, wice=Wice_old)
        Kheat = spatial_average(Kheat, method='arithmetic')
                    

        # vapor conductivity [m2 s-1]
        Kvm = molecular_diffusivity_porous_media(T_old, W_old + Wice_old, poros, scalar='h2o')
        Kvap = Kvf + Kvm 
        #Kvap = spatial_average(Kvap, method='arithmetic')
                    
        
        # take bottom boundary conditions from previous timestep
        q_bot = -KLh[-1] * ((h_old[-1] - forcing['hsoil']) / dz + 1)
        #q_bot = 0.0
        #lbc_h = ('flux', 0.0)
        #lbc_h = ('temperature', forcing['Tsoil'])
#        lbc_h = ('flux', Kheat[-1] * (T_old[-1] - forcing['Tsoil']) / (2*dz)) # Wm-2
#        print(lbc_h[1])
        #--- solve vapor phase and return E [kg m-3 s-1] & c_vap[kg m-3]
        E, c_vap = solve_vapor(h_iter, T_iter, Kvap, ubc=forcing['c_vap'])
        #E = np.zeros(N)    
        # initiate iteration over dt
        err_h = 999.0
        err_W = 999.0
        err_Wice = 999.0
        err_T = 999.0
        iterNo = 0
        
        while (err_h > crit_h or err_W > crit_W or err_Wice > crit_Wice or err_T > crit_T):

            iterNo += 1
            #print(t)
            # previous iteration values
            h_iter0 = h_iter.copy()                                      
            W_iter0 = W_iter.copy()
            Wice_iter0 = Wice_iter.copy()
            T_iter0 = T_iter.copy()  
            
            # --- here add solution of surface energy balance!!
            Gsurf = forcing['Gsurf']
            ubc_h = ('flux', Gsurf) # Wm-2
            
            # solve liquid water flow
            Ew = 1e-3 * E # evaporation / condensation [s-1]
            #Ew = np.zeros(N)
            h_iter, W_iter = solve_liquid(dt, h_iter, W_iter, W_old, KLh, S=Ew, q_top=0.0, q_bot=q_bot)

            #--- solve heat equation & liquid and ice contents
            # heat advection with liquid flow Wm-3 # HOW TO IMPLEMENT boundary conditions??
            F = heat_advection(KLh, h_iter, T_old, q_bot, Ttop=T_old[0], Tbot=T_old[-1])

            # bulk soil heat capacity [Jm-3K-1]
            CP = volumetric_heat_capacity(poros, W_iter, Wice_iter)
        
            # heat capacity due to freezing/thawing [Jm-3K-1]
            A = ICE_DENSITY*LATENT_HEAT_FREEZING*gamma
            
            Lv = latent_heat_vaporization(T_iter) # J kg-1
            LE = Lv*E # Wm-3
            
            #lower boundary condition
            lbc_h = ('flux', Kheat[-1] * (T_iter[-2] - forcing['Tsoil']) / (2*dz)) # Wm-2
            #lbc_h = ('flux', 20.0)
            T_iter = solve_heat(dt, T_iter, T_old, Wice_iter, Wice_old, CP, CP_old, A,
                                Kheat, S=LE, F=F, ubc=ubc_h, lbc=lbc_h)
            
            W_iter, Wice_iter, gamma = frozen_water(T_iter, W_iter + Wice_iter, fp=fp)
            h_iter = water_retention(pF, theta=W_iter)
            
                         
            # check solution, if problem continue with smaller dt or break
            if any(abs(h_iter - h_iter0) > 1.0) or any(np.isnan(h_iter)):
                if dt > dt_min:
                    dt = max(dt / 3.0, dt_min)
                    print('(iteration %s) Solution blowing up, retry with dt = %.1f s' %(iterNo, dt))

                    iterNo = 0
                    h_iter = h_old.copy()
                    W_iter = W_old.copy()
                    Wice_iter = Wice_old.copy()
                    T_iter = T_old.copy()
                    continue
                else:  # convergence not reached with dt=30s, break
                    print('(iteration %s)  No solution found (blow up), h and Wtot set to old values' %(iterNo))
                    h_iter = h_old.copy()
                    W_iter = W_old.copy()
                    Wice_iter = Wice_old.copy()
                    T_iter = T_old.copy()
                    break

            # if problems reaching convergence devide time step and retry
            if iterNo == 20:
                if dt > dt_min:
                    dt = max(dt / 3.0, dt_min)
                    print('iteration %s) More than 20 iterations, retry with dt = %.1f s' %(iterNo, dt))
                    iterNo = 0
                    continue
                else:  # convergence not reached with dt=30s, break
                    print('(iteration %s) Solution not converging, err_h: %.5f, errW: %.5f, errT: %.5f,, errWice: %.5f' 
                          %(iterNo, err_h, err_W, err_T, err_Wice))
                    break

            # errors for determining convergence
            err_h = max(abs(h_iter - h_iter0))
            err_W = max(abs(W_iter - W_iter0))
            err_Wice = max(abs(Wice_iter - Wice_iter0))
            err_T = max(abs(T_iter - T_iter0))

        # end of iteration loop
            
        # new state
        h = h_iter.copy()
        W = W_iter.copy()
        Wice = Wice_iter.copy()
        T = T_iter.copy()
        
        del h_iter, W_iter, Wice_iter, T_iter

        # cumulative fluxes over integration time
        Q_bot += q_bot * dt
        Q_top += E[0] * dt
        Evap += sum(Ew*dz)*dt
        LEcum += sum(LE*dz)*dt
        G_top += ubc_h[1] * dt
        G_bot += lbc_h[1] * dt

        Eflx += Ew*dt
        # Heat flux [J m-2]
        Fheat[1:-1] += -Kheat[1:-1]*(T[1:] - T[:-1])/ dz * dt
        Fheat[0] += ubc_h[1] * dt
        Fheat[-1] += lbc_h[1] * dt
            
        # solution time and new timestep
        t += dt

        dt_old = dt  # save temporarily
        # select new time step based on convergence
        if iterNo <= 3:
            dt = dt * 2
        elif iterNo >= 6:
            dt = dt / 2
        # limit to minimum of 30s
        dt = max(dt, 30)
        
#        # save dto for output to be used in next run
        if dt_old == t_final or t_final > t:
            dto = min(dt, t_final)
        # limit by time left to solve
        dt = min(dt, t_final-t)

    """ time loop ends """

    # mass balance error [m]
    mbe = (sum(W_ini + Wice_ini) - sum(W + Wice))*dz + Q_bot + Evap
    
    
    # energy closure [Wm-2]
    CP_ini = volumetric_heat_capacity(poros, W_ini, Wice_ini)
    CP = volumetric_heat_capacity(poros, W, Wice)
    
    ebe = sum((CP_ini * T_ini - CP * T)*dz + (Fheat[:-1] - Fheat[1:])) - LEcum
    ebe = ebe / t_final
    
    fluxes = {'Evap': Evap / t_final,  # evaporation/condensation at surface (kg m-2 s-1)
              'LE': LEcum / t_final,
              'Qbot': Q_bot / t_final, # liquid water flow at bottom boundary
              'Gtop': G_top / t_final, # heat flux at surface (Wm-2)
              'Gbot': G_bot / t_final, # heat flux at bottom boundary (Wm-2)
              'mbe': mbe,              # mass balance error (m)
              'ebe': ebe,               # energy balance error (Wm-2)
              'Eflx': Eflx / t_final
              }
    
    states = {'water_potential': h,
              'volumetric_water_content': W,
              'volumetric_ice_content': Wice,
              'temperature': T,
              'h2o': c_vap
            }
    return fluxes, states, dto

# ---- functions to set and solve tridiagonal matrices during iteration---

def solve_liquid(dt, h_iter, W_iter, W_old, KLh, S, q_top, q_bot):
    # solves tridiagonal system for liquid-phase    
    # N, dz, dz2 are global
    
    # differential water capacity [m-1]
    C = diff_wcapa(pF, h_iter)    

    """ set up tridiagonal matrix """
    a = np.zeros(N)  # sub diagonal
    b = np.zeros(N)  # diagonal
    g = np.zeros(N)  # super diag
    f = np.zeros(N)  # rhs

    # intermediate nodes i=1...N-1
    b[1:-1] = C[1:-1] + dt / dz2 * (KLh[1:N-1] + KLh[2:N])
    a[1:-1] = - dt / dz2 * KLh[1:N-1]
    g[1:-1] = - dt / dz2 * KLh[2:N]
    f[1:-1] = C[1:-1] * h_iter[1:-1] - (W_iter[1:-1] - W_old[1:-1]) + dt / dz \
                * (KLh[1:N-1] - KLh[2:N]) - S[1:-1] * dt

    # top node i=0 is zero flux
    b[0] = C[0] + dt / dz2 * KLh[1]
    a[0] = 0.0
    g[0] = -dt / dz2 * KLh[1]
    f[0] = C[0] * h_iter[0] - (W_iter[0] - W_old[0]) + dt / dz \
            * (-q_top - KLh[1]) - S[0] * dt

    # bottom node i=N is prescribed flux
    b[-1] = C[-1] + dt / dz2 * KLh[N-1]
    a[-1] = -dt / dz2 * KLh[N-1]
    g[-1] = 0.0
    f[-1] = C[-1] * h_iter[-1] - (W_iter[-1] - W_old[-1]) + dt / dz \
            * (KLh[N-1] + q_bot) - S[-1] * dt
    #else:  # head boundary
    #    h_bot = lbc_liq
    #    b[-1] = C[-1] + dt / dz2 * (KLh[N-1] + KLh[N])
    #    a[-1] = - dt / dz * KLh[N-1]
    #    g[-1] = 0.0
    #    f[-1] = C[-1] * h_iter[-1] - (W_iter[-1] - W_old[-1]) + dt / dz\
    #            * ((KLh[N-1] - KLh[N]) + KLh[N] / dz * h_bot) - S[-1] * dt

    # solve new pressure head and corresponding moisture
    h_new = thomas(a, b, g, f)
    W_new = water_retention(pF, psi=h_new)
    
    return h_new, W_new

def solve_heat(dt, T_iter, T_old, Wice_iter, Wice_old, CP, CP_old, A, Kt, S, F, ubc, lbc):
    # solves heat flow as tridiagonal system
    # N, dz, dz2 are global

    #print('Wice, Wice_old:',  Wice_iter, Wice_old, 'CP, CP_old:', CP, CP_old)
    #print('A', A, 'S', S, 'Kt', Kt)
    """ set up tridiagonal matrix """
    a = np.zeros(N)
    b = np.zeros(N)
    g = np.zeros(N)
    f = np.zeros(N)

    # intermediate nodes
    b[1:-1] = CP[1:-1] + A[1:-1] + dt / dz2 * (Kt[1:N-1] + Kt[2:N])
    a[1:-1] = - dt / dz2 * Kt[1:N-1]
    g[1:-1] = - dt / dz2 * Kt[2:N]
    f[1:-1] = CP_old[1:-1] * T_old[1:-1] + A[1:-1] * T_iter[1:-1] \
            + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[1:-1] - Wice_old[1:-1]) - (S[1:-1] + F[1:-1]) * dt

    # top node i=0
    if ubc[0] == 'flux':  # flux bc
        F_top = ubc[1] 
        b[0] = CP[0] + A[0] + dt / dz2 * Kt[1]
        a[0] = 0.0
        g[0] = -dt / dz2 * Kt[1]
        f[0] = CP_old[0]*T_old[0] + A[0]*T_iter[0] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[0] - Wice_old[0])\
                - dt / dz * F_top - dt*(S[0] + F[0])

    if ubc[0] == 'temperature':  # temperature bc
        T_sur = ubc[1]
        b[0] = CP[0] + A[0] + dt / dz2 * (Kt[0] + Kt[1])
        a[0] = 0.0
        g[0] = -dt / dz2 * Kt[1]
        f[0] = CP_old[0]*T_old[0] + A[0]*T_iter[0] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[0] - Wice_old[0])\
                + dt / dz2 * Kt[0]*T_sur - dt*(S[0] + F[0])

    # bottom node i=N
    if lbc[0] == 'flux':  # flux bc
        F_bot = lbc[1]
        b[-1] = CP[-1] + A[-1] + dt / dz2 * Kt[N-1]
        a[-1] = -dt / dz2 * Kt[N-1]
        g[-1] = 0.0
        f[-1] = CP_old[-1]*T_old[-1] + A[-1]*T_iter[-1] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[-1] - Wice_old[-1])\
                - dt / dz * F_bot - dt*(S[-1] + F[-1])

    if lbc[0] == 'temperature':  # temperature bc
        T_bot = lbc[1]
        b[-1] = CP[-1] + A[-1] + dt / dz2 * (Kt[N-1] + Kt[N])
        a[-1] = -dt / dz2 * Kt[N-1]
        g[-1] = 0.0
        f[-1] = CP_old[-1]*T_old[-1] + A[-1]*T_iter[-1] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[-1] - Wice_old[-1])\
                + dt / dz2 * Kt[N]*T_bot - dt*(S[-1] + F[-1])

    # solve new temperature
    T_new = thomas(a, b, g, f)
    
    return T_new

def heat_advection(KLh, h, T, q_bot, Ttop, Tbot):
    """
    approximates heat advection with liquid water flow
    """
    # liquid flux m/s
    q = np.zeros(N+1)
    q[1:-1] = -KLh[1:-1] * ((h[0:-1] - h[1:]) / dz + 1.0)
    q[0] = 0.0
    q[-1] = q_bot
    
    # heat advection m/s*K
    qT = np.zeros(N+1)
    qT[0] = q[0]*Ttop
    qT[-1] = q[-1]*Tbot
    qT[1:-1] = q[1:-1]*T[0:-1]
    
    # advective heat source/sink per layer Wm-3 
    F = -CV_WATER* np.diff(qT) / dz 

    return F

def solve_vapor(h, T, Kvap, ubc):
    """
    solves vapor concentration profile in air-space of porous media assuming
    equilibrium with liquid water in soil and steady-state conditions
    Args:
        h - pressure head (m)
        T - temperature (degC)
        Kvap - conductivity (m2 s-1)
        ubc - upper bc (concentration at surface air)
        
    Returns:
        E - evaporation/condensation rate at each layer [kg m-3 s-1]
        c - concentration profile [kg m-3]
    """
    # N, dz, dz2 are global
    #N = len(z); dz=z[1]-z[0]; dz2=dz**2
    # assume vapor density c in equilibrium with soil h
    csat = saturation_vapor_density(T) # kgm-3
    c = csat * relative_humidity(h, T)
    
    # E = 1/dz*(dc/dz) with centered finite difference scheme
    E = np.zeros(N)
 
    # intermediate nodes
    #E[1:-1] = (Kvap[1:N-1] * (c[0:-2] - c[1:-1]) - Kvap[1:-1] * (c[1:-1 - c[2:N]])) / dz2
    E[1:-1] = -(Kvap[1:N-1] * (c[0:-2] - c[1:-1]) - Kvap[1:-1] * (c[1:-1] - c[2:N])) / dz2
    # bc's
    E[0] = -Kvap[0] * (ubc - c[0]) * 2 / dz # forward-difference
    E[-1] = 0.0 # zero-flux
    
    return E, c
 
def water_retention(pF, theta=None, psi=None):
    """
    Water retention curve vanGenuchten - Mualem
    Args:
        pF - parameter dict
        theta - vol. water content (m3m-3)
        psi - matrix potential (m), <=0
    Returns:
        theta or psi
    """

    Ts = np.array(pF['ThetaS'], ndmin=1)
    Tr = np.array(pF['ThetaR'], ndmin=1)
    alfa = np.array(pF['alpha'], ndmin=1)
    n = np.array(pF['n'], ndmin=1)
    m = 1.0 - np.divide(1.0, n)

    def theta_psi(x):
        # converts water content (m3m-3) to potential (m)
        x = np.minimum(x, Ts)
        x = np.maximum(x, Tr)  # checks limits
        s = (Ts - Tr) / ((x - Tr) + EPS)
        Psi = -1e-2 / alfa*(s**(1.0 / m) - 1.0)**(1.0 / n)  # m
        Psi[np.isnan(Psi)] = 0.0
        return Psi

    def psi_theta(x):
        # converts water potential (m) to water content (m3m-3)
        x = 100*np.minimum(x, 0)  # cm
        Th = Tr + (Ts - Tr) / (1 + abs(alfa*x)**n)**m
        return Th

    if theta is not None:
        return theta_psi(theta)
    else:
        return psi_theta(psi)

def hydraulic_conductivity(pF, wliq, Ksat=1.0):
    r""" Unsaturated liquid-phase hydraulic conductivity following 
    vanGenuchten-Mualem -model.

    Args:
        pF (dict):
            'ThetaS' (float/array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'ThetaR' (float/array): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (float/array): air entry suction [cm\ :sup:`-1`]
            'n' (float/array): pore size distribution [-]
        wliq (float or array): liquid water content
        Ksat (float or array): saturated hydraulic conductivity [units]
    Returns:
        Kh (float or array): hydraulic conductivity (if Ksat ~=1 then in [units], else relative [-])

    """

    w = np.array(wliq)
    
    # water retention parameters
    l = pF['l']
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)
    
    # saturation ratio
    S = np.minimum(1.0, (w - pF['ThetaR']) / (pF['ThetaS'] - pF['ThetaR']) + EPS)

    Kh = Ksat * S**l * (1 - (1 - S**(1/m))**m)**2

    return Kh

def diff_wcapa(pF, h):
    r""" Differential water capacity calculated numerically.
    Args:
        pF (dict):
            'ThetaS' (array): saturated water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'ThetaR' (array): residual water content [m\ :sup:`3` m\ :sup:`-3`\ ]
            'alpha' (array): air entry suction [cm\ :sup:`-1`]
            'n' (array): pore size distribution [-]
        h (array): pressure head [m]
    Returns:
        dwcapa (array): differential water capacity dTheta/dhead [m\ :sup:`-1`]

    Kersti Haahti, Luke 8/1/2018
    """

    dh = 1e-5
    theta_plus = water_retention(pF, psi=h + dh)
    theta_minus = water_retention(pF, psi=h- dh)

    # differential water capacity
    dwcapa = (theta_plus - theta_minus) / (2 * dh)

    return dwcapa

def relative_humidity(psi, T):
    """
    relative humidity in equilibrium with water potential
    Args:
        psi - water potential [m]
        T - temperature [degC]
    Returns
        rh - relative humidity [-]
    """

    rh = np.exp(GRAVITY*MOLAR_MASS_H2O*psi / (GAS_CONSTANT * (T + DEG_TO_KELVIN)))

    return rh

def latent_heat_vaporization(T):
    """ latent heat of vaporization of water
    Arg:
        T - temperature (degC)
    Returns:
        Lv - lat. heat. of vaporization (J kg-1)
    """
    return 1.0e6*(2.501 - 2.361e-3*T)  # J kg-1

def saturation_vapor_density(T):
    """
    Args:
        T [degC]
    Returns
        rhos [kg m-3]
    """
    es = 611.0 * np.exp((17.502 * T) / (T + 240.97)) #Pa
    
    rhos = MOLAR_MASS_H2O / (GAS_CONSTANT * (T + DEG_TO_KELVIN)) * es #kg m-3
    return rhos

def air_density(T, P=101300.0):
    """
    Args:
        T - temperature [degC]
        P - ambient pressure [Pa]
    OUTPUT:
        rhoa - density of dry air [kg m-3]
    """
    T = T + DEG_TO_KELVIN

    rhoa = P * MOLAR_MASS_AIR / (GAS_CONSTANT * T) # kg m-3

    return rhoa

def volumetric_heat_capacity(poros, wliq=0.0, wice=0.0):
    r""" Computes volumetric heat capacity of porous organic matter.

    Args:
        poros: porosity [m3 m-3]
        wliq: volumetric water content [m3 m-3]
        wice: volumetric ice content [m3 m-3]
    Returns:
        cv: volumetric heat capacity [J m-3 K-1]
    """
    wair = poros - wliq - wice
    cv = CV_ORGANIC * (1. - poros) + CV_WATER * wliq + CV_ICE * wice + CV_AIR * wair

    return cv

def thermal_conductivity(wliq, wice=0.0):
    """ thermal conductivity in organic matter"""
    # o'Donnell et al. 2009
    k = 0.032 + 0.5 * wliq
    return k

def frozen_water(T, wtot, fp=0.25, To=0.0):
    r""" Approximates ice content from soil temperature and total water content.

    Args:
        T : soil temperature [degC]
        wtot: total volumetric water content [m3 m-3]
        fp: parameter of freezing curve [-]
            2...4 for clay and 0.5-1.5 for sandy soils
            < 0.5 for peat soils (Nagare et al. 2012 HESS)
        To: freezing temperature of soil water [degC]
    Returns:
        wliq: volumetric water content [m3 m-3]
        wice: volumetric ice content [m3 m-3]
        gamma: dwice/dT [K-1]
    References:
        For peat soils, see experiment of Nagare et al. 2012:
        http://scholars.wlu.ca/cgi/viewcontent.cgi?article=1018&context=geog_faculty
    """
    wtot = np.array(wtot)
    T = np.array(T)
    fp = np.array(fp)

    wice = wtot*(1.0 - np.exp(-(To - T) / fp))
    # derivative dwliq/dT
    gamma = (wtot - wice) / fp

    ix = np.where(T > To)[0]
    wice[ix] = 0.0
    gamma[ix] = 0.0

    wliq = wtot - wice

    return wliq, wice, gamma

def molecular_diffusivity_porous_media(T, wtot, porosity, scalar, P=101300.0):
    r""" Estimates molecular diffusivity in porous media.
    Args:
        T - temperature [degC]
        vol_water + ice: [m3m-3]
        porosity: [m3m-3]
        scalar: 'h2o', 'co2', 'heat' (add' o2', 'ch4')
        P: [Pa]

    Returns:
        D - molecular diffusivity [m2s-1]
    Ref: Millington and Quirk (1961)

    """
    T =T + DEG_TO_KELVIN
    
    # temperature sensitivity of diffusivity in air
    t_adj = (101300.0 / P) * (T / 293.16)**1.75
             
    if scalar.lower() == 'h2o':
        Do = MOLECULAR_DIFFUSIVITY_H2O * t_adj
    if scalar.lower() == 'co2':
        Do = MOLECULAR_DIFFUSIVITY_CO2 * t_adj
    if scalar.lower() == 'heat':
        Do = THERMAL_DIFFUSIVITY_AIR * t_adj 
    
    # [m3/m3], air filled porosity
    afp = np.maximum(0.0, porosity - wtot)

    # [mol m-3], air molar density
    #cair = P / (GAS_CONSTANT * T)
    
    # D/Do, diffusivity relative to free air Millington and Quirk (1961)
    f = np.power(afp, 10.0/3.0) / porosity**2
                        
    return f * Do

def surface_energy_balance(forcing, params, Ts, gsoil):
    
    alb = params['albedo']
    emi = params['emissivity']
    zref = params['zref']
    
    #SWabs = forcing['SWabs']
    SWup= alb * forcing['SWdn']
    SWabs = (1-alb) * forcing['SWdn']
    LWin = (1-emi) * forcing['LWdn']
    ust = forcing['ust']
    Ta = forcing['Ta'] + DEG_TO_KELVIN
    Ts = Ts + DEG_TO_KELVIN
    
    
    # coeffs
    a = SWabs + LWin
    b = emi*STEFAN_BOLTZMANN
    d = gsoil
    
    def surface_temperature(T, *para):
        a, b, c, d, Ta, Ts = para
        f = a - b * np.power(T, 4) - c*(T - Ta) - d *(T - Ts)
        
        return f

    err = 999.0
    dT = 0.0
    T = 0.5 * (Ta + Ts)
    while err > 0.01:
        gh, _, _ = moss_atm_conductance(zref, ust=ust, dT=dT) # molm-2s-1
        c = SPECIFIC_HEAT_AIR * gh
        para = (a, b, c, d, Ta, Ts)
        
        Told = np.copy(T)
        
        # --- find surface temperature
        T = fsolve(surface_temperature, T, para)
        
        err = abs(T - Told)
    
    # -- fluxes (Wm-2)
    LWup = b * np.power(T, 4)
    H = c * (T - Ta)
    G = d * (T - Ts)
    
    ebe = SWabs + LWin - LWup - H - G
    res = {'H': H, 'G': G, 'SWup': SWup, 'LWup': LWup,
           'Tsurf': T - DEG_TO_KELVIN, 'ebe': ebe}
    return res

def moss_atm_conductance(zref, ust=None, U=None, zom=None, dT=0.0, b=1.1e-3):
    """
    Soil surface - atmosphere transfer conductance for scalars. Two paralell
    mechanisms: forced and free convection
    Args:
        U - wind speed (m/s) at zref
        zref - reference height (m). Log-profile assumed below zref.
        zom - roughness height for momentum (m), ~0.1 x canopy height
        ustar - friction velocity (m/s) at log-regime. if ustar not given,
                it is computed from Uo, zref and zom
        b - parameter for free convection. b=1.1e-3 ... 3.3e-3 from smooth...rough surface
    Returns:
        gh, gv, gc - conductances for heat, water vapor and CO2
    References:
        Schuepp and White, 1975:Transfer Processes in Vegetation by Electrochemical Analog,
        Boundary-Layer Meteorol. 8, 335-358.
        Schuepp (1977): Turbulent transfer at the ground: on verification of
        a simple predictive model. Boundary-Layer Meteorol., 171-186
        Kondo & Ishida, 1997: Sensible Heat Flux from the Earth’s Surface under
        Natural Convective Conditions. J. Atm. Sci.    
    """
    
    Sc_v = AIR_VISCOSITY / MOLECULAR_DIFFUSIVITY_H2O  
    Sc_c = AIR_VISCOSITY / MOLECULAR_DIFFUSIVITY_CO2
    Pr = AIR_VISCOSITY / THERMAL_DIFFUSIVITY_AIR 

    d = 0.0 # displacement height
    
    if ust == None:
        ust = U * VON_KARMAN / np.log((zref - d) / zom)

    delta = AIR_VISCOSITY / (VON_KARMAN * ust)
    
    gb_h = (VON_KARMAN * ust) / (Pr - np.log(delta / zref))
    gb_v = (VON_KARMAN * ust) / (Sc_v - np.log(delta / zref))
    gb_c = (VON_KARMAN * ust) / (Sc_c - np.log(delta / zref))
    
    # free convection as parallel pathway, based on Condo and Ishida, 1997.
    #b = 1.1e-3 #ms-1K-1 b=1.1e-3 for smooth, 3.3e-3 for rough surface
    dT = np.maximum(dT, 0.0)
    
    gf_h = b * dT**0.33  # ms-1

    # mol m-2 s-1    
    gb_h = (gb_h + gf_h) * AIR_DENSITY
    gb_v = (gb_v + Sc_v / Pr * gf_h) * AIR_DENSITY
    gb_c = (gb_c + Sc_c / Pr * gf_h) * AIR_DENSITY
    
    return gb_h, gb_v, gb_c

def forward_diff(y, dx):
    """
    computes gradient dy/dx using forward difference
    assumes dx is constant
    """
    N = len(y)
    dy = np.ones(N) * np.NaN
    dy[0:-1] = np.diff(y)
    dy[-1] = dy[-2]
    return dy / dx

def central_diff(y, dx):
    """
    computes gradient dy/dx with central difference method
    assumes dx is constant
    """
    N = len(y)
    dydx = np.ones(N) * np.NaN
    # -- use central difference for estimating derivatives
    dydx[1:-1] = (y[2:] - y[0:-2]) / (2 * dx)
    # -- use forward difference at upper boundary
    dydx[0] = (y[1] - y[0]) / dx
    # -- use backward difference at lower boundary
    dydx[-1] = (y[-1] - y[-2]) / dx

    return dydx