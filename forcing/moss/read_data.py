# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 17:05:29 2017

@author: slauniai
"""
#reading near-surface microclimate into pandas dataframe with datetime index
import pandas as pd
from datetime import datetime

fname='bryo_microclimate_LAI1.0.csv'
#def read_file(fname):
    #import data
dat=pd.read_csv(fname, sep=',', header='infer', na_values=-999)

#parse to datetime
N=len(dat)

t=[]
for k in range(N): 
    t.append( datetime(dat['yyyy'].iloc[k],dat['mo'].iloc[k],dat['dd'].iloc[k],dat['hh'].iloc[k],dat['mm'].iloc[k]) )
#set to dataframe index
dat.index=t



  
