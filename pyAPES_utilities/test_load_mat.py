# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:06:03 2019

@author: slauniai
"""

import numpy as np
import scipy.io as sio

fname = 'Hy2005APES.mat'

dat = sio.loadmat(fname)
a = dat['Hy05APES']
a.keys()