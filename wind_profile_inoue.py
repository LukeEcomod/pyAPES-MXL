# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:45:13 2019

@author: slauniai
"""

import numpy as np
import matplotlib.pyplot as plt
#: machine epsilon
EPS = np.finfo(float).eps

Uo = 0.1
usto = 0.1

LAI = 5.0
a = LAI / 2.0
hc = 0.1
N = 20
dz = hc / N
z = np.arange(0, hc + dz, dz)

tau = usto**2 * np.exp(2*a*(z / hc - 1.0))

U = Uo * np.exp(a * (z / hc - 1.0))

dUdz = Uo * a / hc * np.exp(a * (z / hc - 1.0))

dU = np.diff(U)/dz

b = (usto / Uo)**2
L = 2*b*hc/LAI

plt.figure()
plt.subplot(221); plt.plot(U, z)
plt.subplot(222); plt.plot(tau, z)
plt.subplot(223); plt.plot(dUdz, z)
plt.subplot(223); plt.plot(dU, z[1:])

