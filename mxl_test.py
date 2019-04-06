# -*- coding: utf-8 -*-
"""
SCRIPT FOR TESTING MIXED-LAYER MODEL USING ONE DAYTIME DATA FROM HYYTIÄLÄ

Created on Thu Sep 20 14:55:56 2018

@author: slauniai
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
EPS = np.finfo(float).eps  # machine epsilon

# --- import constants from module mxl
from mxl.mxl import CP_AIR_MASS, MAIR_DRY, MH2O, NT, R
# --- import model class and utility functions from module mxl
from mxl.mxl import MXLmodel
# --- import tools to read forcing data
from mxl.utils import read_mxl_forcing

# --- import parameters & initial conditions for mxl model
from mxl.parameters import mxlpara, ini

wdir = r'c:\repositories\pyAPES-MXL'
os.chdir(wdir)
#print('---- working dir: ' + os.getcwd())

print('---- reading forcing ---')

# --- read forcing for testing mxl growth
ffile = r'forcing\FIHy_mxl_forcing_2014.dat'

fday = '2014-07-03'
lday = '2014-07-03'

forc = read_mxl_forcing(ffile, fday, lday)

# select one day and conditions when H > 0
#forc = forcing_data.loc[fday:lday][['doy','ust', 'wt', 'wq', 'wc', 'P', 'Ta', 'U']]

# this selects periods when H >0 from forcing.
forc = forc[forc['H'] > 0]

# plot figure
plt.figure()
plt.plot(forc['wt']); plt.ylabel('wt (Kms-1)')

#%% create model instance

print('---- creating MXL-model----')
# --- Create model instance
run1 = MXLmodel(ini, mxlpara)
## print run1.__dict_

print('---- running MXL-model----')
nsteps = len(forc) # len(F_h)
tt = 30.*np.arange(0,nsteps) # time vector, min

# initialize results dictionany, fill with NaN's
res = {'h': np.ones(nsteps)*np.NaN, 'theta': np.ones(nsteps)*np.NaN, 
       'q':np.ones(nsteps)*np.NaN, 'ca': np.ones(nsteps)*np.NaN,
       'h_lcl': np.ones(nsteps)*np.NaN, 'vpd': np.ones(nsteps)*np.NaN,
       'U': np.ones(nsteps)*np.NaN, 'u': np.ones(nsteps)*np.NaN
      }

# run model for nsteps
for k in range(nsteps):
    wt = forc['wt'][k]
    wq = forc['wq'][k]
    wc = forc['wc'][k]
    ust = forc['ust'][k]
    
    run1.run_timestep(wt, wq, wc, ust)

    res['h'][k] = run1.h
    res['theta'][k] = run1.theta
    res['q'][k] = run1.q
    res['ca'][k] = run1.ca
    res['h_lcl'][k] = run1.h_lcl
    res['vpd'][k] = run1.vpd
    res['U'][k] = run1.U
    res['u'][k] = run1.u

print('---- making graphs----')

plt.figure(1)
plt.subplot(221); plt.plot(tt, forc['H'], 'r'); plt.ylabel('H (Wm-2)')
plt.subplot(222); plt.plot(tt, forc['LE'], 'b'); plt.ylabel('E (Wm-2)')
plt.subplot(223); plt.plot(tt, forc['NEE'], 'g'); plt.ylabel('NEE (umol m-2 s-1)')
plt.subplot(224); plt.plot(tt, forc['ust'], 'k'); plt.ylabel('ustar (m s-1)')

plt.savefig('forc.png')

plt.figure(2)
plt.subplot(321); plt.plot(tt, res['h'], label='mxl'); plt.plot(tt, res['h_lcl'], label='lcl'); 
plt.ylabel('mxl and lcl height (m)'); plt.legend(fontsize=8)
plt.subplot(322); plt.plot(tt, res['theta']); plt.ylabel('Theta (K)')
plt.subplot(323); plt.plot(tt, res['q']); plt.ylabel('q (kg/kg)')
plt.subplot(324); plt.plot(tt, res['vpd']); plt.ylabel('vpd (kPa)')
plt.subplot(325); plt.plot(tt, res['ca']); plt.ylabel('ca (ppm)'); plt.xlabel('time (min)')
plt.subplot(326); plt.plot(tt, res['U'], label='U'); plt.plot(tt, res['u'], label='u horiz')
plt.ylabel('velocity (ms-1)'); plt.xlabel('time (min)'); plt.legend(fontsize=8)
plt.savefig('mxl-test.png')

print('---- done ! ----')