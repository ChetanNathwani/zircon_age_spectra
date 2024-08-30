import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import glob as glob
import os as os
import ot
import seaborn as sb
from scipy.stats import ks_2samp
import geochron as geochron

tims_data = pd.read_csv('zircon_tims_comp.csv',encoding = "ISO-8859-1")
tims_data['NormalizedAge'] = tims_data.groupby('Unit')['age68'].transform(geochron.normalize_age)
tims_data = tims_data[(tims_data['Type'] == 'Volcanic') | (tims_data['Type'] == 'Porphyry') | (tims_data['Type'] == 'Plutonic')]
tims_data = tims_data[tims_data['Notes'] != 'x']

n_cutoff = 10
dt_sigma_cutoff = 10
std_weights_cutoff = 0.08
tims_data = tims_data.groupby('Unit').filter(lambda x: x['age68'].mean() < 130) # Filter those with average age less than 500 Ma
tims_data = tims_data.groupby('Unit').filter(lambda x: len(x['age68']) >= n_cutoff) # Filter those with length < threshold
tims_data = tims_data.groupby('Unit').filter(lambda x: np.std((1/x['2s_68']**2)/np.sum(1/x['2s_68']**2)) < std_weights_cutoff) # Filter those with very variable weights
tims_data = tims_data.reset_index(drop=True)

tims_duration = tims_data.groupby('Unit').apply(lambda grp: np.max(grp['age68'])
                                                - np.min(grp['age68']))

groups = [tims_data.groupby('Unit').get_group(x) for x in tims_data.groupby('Unit').groups]

for n, group in enumerate(groups):
    group = group.sort_values(['age68'])
    groups[n] = group.loc[geochron.filter_older_ages(group['age68'], gradient_cut_off = 0.3, tflatmax = 0.25).index] # Filter inheritance   

filtered = pd.concat(groups)
filtered = filtered.groupby('Unit').filter(lambda x: (np.max(x['age68'])-np.min(x['age68']))/(np.mean(x['2s_68'])/2) >= dt_sigma_cutoff) # Filter those with low dt/s
filtered = filtered.groupby('Unit').filter(lambda x: np.std((1/x['2s_68']**2)/np.sum(1/x['2s_68']**2)) < std_weights_cutoff) # Filter those with very variable weights
filtered = filtered.groupby('Unit').filter(lambda x: len(x['age68']) >= n_cutoff) # Filter those with number zircons < threshold

filtered.to_csv('zircon_tims_comp_filtered.csv', index = False)