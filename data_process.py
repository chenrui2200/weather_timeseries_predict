# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df=pd.read_excel('./data/weather.xls', engine='xlrd')

'''
rho (g/m**3) 密度
sh (g/kg) 比湿
T (degC) 温度 
Tdew (degC) 露点温度
Tlog (degC) 记录器温度
Tpot (K) 潜在温度
VPact (mbar) 实际蒸气压
VPmax (mbar) 最大蒸气压
'''
data=pd.concat([np.floor((df['rho (g/m**3)'])),
            np.floor((df['sh (g/kg)'])),
            np.floor((df['T (degC)'])),
            np.floor((df['Tdew (degC)'])),
            np.floor((df['Tlog (degC)'])),
            np.floor((df['Tpot (K)'])),
            np.floor((df['VPact (mbar)'])),
            np.floor((df['VPmax (mbar)']))], axis=1)

data.to_pickle('./data/weather.pkl')
