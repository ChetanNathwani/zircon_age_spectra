import numpy as np
from scipy import stats
import ot
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import warnings

def normalize_age(group):
    min_age = group.min()
    max_age = group.max()
    return (group - min_age) / (max_age - min_age)

def calc_delT(data):
    # Insert a numpy array of dates and return the age normalised to the min and max
    deltaT = (data-np.min(data))/(np.max(data)-np.min(data))
    return(deltaT)

def check_data(ages, unc):
    if len(ages) < 10:
        warnings.warn("Distributions with low number of dates may not provide useful results")
    delta_T = np.max(ages) - np.min(ages)
    avg_sigma = np.mean(unc)
    if delta_T/avg_sigma < 10:
        warnings.warn("Distributions with low ΔT/σ may not provide useful results")
    if np.mean(ages) > 120:
        warnings.warn("Age distributions greater than 120 Ma may not provide useful results, check for Pb loss")

def plot_kde(ages, uncertainty=None, normalize = False):
    if normalize is True:
        ages = (ages - np.min(ages)) / (np.max(ages) - np.min(ages))
    if uncertainty is None:
        kde = stats.gaussian_kde(ages)
    else:
        weights = (1/uncertainty**2)/np.sum(1/uncertainty**2)
        kde = stats.gaussian_kde(ages, weights = weights)
    bw = kde.covariance_factor()*np.std(ages) # extract bw
    # eval_points = np.linspace(np.min(ages)-0.05, np.max(ages)+bw)
    eval_points = np.linspace(np.min(ages), np.max(ages))
    y_sp = kde.pdf(eval_points)
    return(eval_points, y_sp)

def ecdf(ages, uncertainties=None):
    
    # Normalize the data between 0 and 1
    data_min = min(ages)
    data_max = max(ages)
    normalized_data = (ages - data_min) / (data_max - data_min)

    # Sort the data and weights accordingly
    sorted_indices = np.argsort(normalized_data)
    sorted_data = np.array(normalized_data)[sorted_indices]
    
    if uncertainties is None:
        weights = np.ones(len(ages))
    else:
        weights = (1/uncertainties**2)/np.sum(1/uncertainties**2)
        weights = np.array(weights)[sorted_indices]
    
    # Compute the weighted ECDF
    weighted_cumsum = np.cumsum(weights)
    ecdf_y = weighted_cumsum / weighted_cumsum[-1]
    
    return(sorted_data, ecdf_y)
    

def calc_w2(x,y,x_err = None, y_err = None, normalize = False, p = 2):
    
    # Input pandas series which is converted to array
    
    # x = pandas series of age distributions x
    # y = pandas series of age distributions y
    # x_err = pandas series of age uncertainities of x
    # y_err = pandas series of age uncertainties of y
    # normalize = normalize ages between min and max age
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    x_err = np.array(x_err).flatten()
    y_err = np.array(y_err).flatten()
    
    if normalize is True:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Calculate W2
    if np.all(x_err == None):
        w2 = ot.wasserstein_1d(x, y, p=p)
    else:
        x_weights = (1/x_err**2)/np.sum(1/x_err**2)
        y_weights = (1/y_err**2)/np.sum(1/y_err**2)
        w2 = ot.wasserstein_1d(x, y, p=p, u_weights = x_weights, v_weights = y_weights)
    
    w2 = np.sqrt(w2)
    
    return(w2)

def calc_flatness(ages, uncertainty = None, gradient_cut_off = 0.3, young_index = 2):
    
    # ages is an array of the unnormalised age distribution
    # threshold is the gradient of an ECDF below which inheritance is marked
    # f_threshold is the maximum relative age below the threshold which is deemed acceptable for an age distribution
    # young_index is the index from which look at the gradient of the ECDF (to avoid younger outliers)   
    x,y = ecdf(ages, uncertainty)
    dx = np.gradient(x)
    dy = np.gradient(y)
    gradient_ecdf = dy/dx
    gradient_ecdf = gradient_ecdf[young_index:] # Exclude the gradient of the youngest part of the curve
    flatness = np.sum(np.diff(x[young_index:][np.argwhere(gradient_ecdf<gradient_cut_off)].flatten()))
    
    return flatness

def filter_older_ages(age_dist, unc=None, weighted = False, gradient_cut_off = 0.3, young_index = 2, tflatmax = 0.25):
    if not weighted or unc is None:
        while calc_flatness(age_dist, gradient_cut_off = gradient_cut_off, young_index = young_index) >= tflatmax:
            age_dist = age_dist[:-1]
            if unc is None:
                unc = unc
            else:
                unc = unc[:-1]
    else:
        while calc_flatness(age_dist, uncertainty = unc, gradient_cut_off = gradient_cut_off, young_index = young_index) >= tflatmax:
            age_dist = age_dist[:-1]
            unc = unc[:-1]
    
    return(age_dist,unc)

# Import data for TIMS age vs unc parameterisation

sava = pd.read_csv('../data/zircon_tims_comp.csv',encoding = "ISO-8859-1")
sava = sava[sava['age68'] < 1000]

# Define function to fit to data
def TIMS_func(x, a, b, c):
    output_2s = a * np.power(x, b) + c
    return output_2s

TIMS_popt, TIMS_pcov = curve_fit(TIMS_func, sava['age68'], sava['2s_68']) # Fit regression
def tims_func_unc(params, sigma):
    # Calculate the sigma error on fit parameters by taking the diagonal of the covariance matrix
    pcov_sigma = np.sqrt(np.diag(sigma))

    # Update parameters with gaussian uncertainty
    TIMS_popt_prop = [np.random.normal(params[0], pcov_sigma[0], 1),
                  np.random.normal(params[1], pcov_sigma[1], 1),
                  np.random.normal(params[2], pcov_sigma[2], 1)]
    return TIMS_popt_prop

def laser_func_unc(params, sigma):
    # Calculate the sigma error on fit parameters by taking the diagonal of the covariance matrix
    pcov_sigma = np.sqrt(np.diag(sigma))
    
    # Update parameters with gaussian uncertainty
    laser_popt_prop = [np.random.normal(params[0], pcov_sigma[0], 1),
                       np.random.normal(params[1], pcov_sigma[1], 1)]
    return laser_popt_prop


def generate_pca_scores():

    data = pd.read_csv('../data/zircon_tims_comp_filtered.csv',encoding = "ISO-8859-1")
    localities = [data.groupby('Locality').get_group(x) for x in data.groupby('Locality').groups]

    all_samples = []

    mapper = pd.Series(data.Locality.values,index=data.Unit).to_dict()
    mapper2 = pd.Series(data.Type.values,index=data.Unit).to_dict()

    ws = []
    dic = {}
    count = 0
    for n,locality in enumerate(localities):
        name = locality['Unit'].unique()
        units = [locality.groupby('Unit').get_group(x) for x in locality.groupby('Unit').groups]
        for unit in units:
            all_samples.append(name)
            # s, p = stats.skewtest(unit['age68'])
            sample_name = unit['Unit'].iloc[0]
            unit = unit.drop(['Unit'], axis = 1)
            dic[sample_name] = unit
            count = count + 1


    values = dic.values()
    keys = list(dic.keys())


    for df in values:
        for i in np.arange(0,len(values)):
            df2 = list(values)[i]
            # Calculate Wasserstein metric
            w2 = calc_w2(df['age68'],df2['age68'],x_err = df['2s_68'], y_err = df2['2s_68'], normalize = True)
            ws.append(w2)


    ws = np.array(ws).reshape(len(values), len(values))


    PCA_ws = PCA(n_components = 2)

    names = [*map(mapper.get, keys)]
    types = [*map(mapper2.get, keys)]

    types = pd.Series(types, name = 'Type')
    names = pd.Series(names, name = 'Locality')

    PC_scores = pd.DataFrame(PCA_ws.fit_transform(ws), columns = ['PC1','PC2'])
    PC_scores['Type'] = types
    PC_scores['Locality'] = names

    pca_types = [PC_scores.groupby('Type').get_group(x) for x in PC_scores.groupby('Type').groups]
    
    return PC_scores['PC1'], PC_scores['PC2'], PC_scores['Type']

def calc_W_PCA(ages, unc):
    check_data(ages, unc)
    data = pd.read_csv('../data/zircon_tims_comp_filtered.csv',encoding = "ISO-8859-1")
    localities = [data.groupby('Locality').get_group(x) for x in data.groupby('Locality').groups]

    all_samples = []

    mapper = pd.Series(data.Locality.values,index=data.Unit).to_dict()
    mapper2 = pd.Series(data.Type.values,index=data.Unit).to_dict()

    ws = []
    dic = {}
    count = 0
    for n,locality in enumerate(localities):
        name = locality['Unit'].unique()
        units = [locality.groupby('Unit').get_group(x) for x in locality.groupby('Unit').groups]
        for unit in units:
            all_samples.append(name)
            # s, p = stats.skewtest(unit['age68'])
            sample_name = unit['Unit'].iloc[0]
            unit = unit.drop(['Unit'], axis = 1)
            dic[sample_name] = unit
            count = count + 1


    values = dic.values()
    keys = list(dic.keys())


    for df in values:
        for i in np.arange(0,len(values)):
            df2 = list(values)[i]
            # Calculate Wasserstein metric
            w2 = calc_w2(df['age68'],df2['age68'],x_err = df['2s_68'], y_err = df2['2s_68'], normalize = True)
            ws.append(w2)


    ws = np.array(ws).reshape(len(values), len(values))


    PCA_ws = PCA(n_components = 2)

    names = [*map(mapper.get, keys)]
    types = [*map(mapper2.get, keys)]

    types = pd.Series(types, name = 'Type')
    names = pd.Series(names, name = 'Locality')

    PC_scores = pd.DataFrame(PCA_ws.fit_transform(ws), columns = ['PC1','PC2'])
    PC_scores['Type'] = types
    PC_scores['Locality'] = names

    pca_types = [PC_scores.groupby('Type').get_group(x) for x in PC_scores.groupby('Type').groups]


    w2_user = []
    for i in np.arange(0,len(values)):
        df2 = list(values)[i]
        w2 = calc_w2(ages,df2['age68'],x_err = unc, y_err = df2['2s_68'], normalize = True)
        w2_user.append(w2)
    pc_user = PCA_ws.transform(X = np.array(w2_user).reshape(1, -1)).flatten()
    return(pc_user)


laser = pd.read_csv('../data/laicpms_comp.csv')
# Let's fit an exponential function.  
# This looks like a line on a lof-log plot.
def laser_func(x, a, b):
    return a * np.power(x, b)
laser_popt, laser_pcov = curve_fit(laser_func, laser['68age'], laser['68 2s'])