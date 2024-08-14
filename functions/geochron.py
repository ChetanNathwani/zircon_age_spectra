import numpy as np
from scipy import stats
import ot
import pandas as pd
from scipy.optimize import curve_fit

def normalize_age(group):
    min_age = group.min()
    max_age = group.max()
    return (group - min_age) / (max_age - min_age)

def calc_delT(data):
    # Insert a numpy array of dates and return the age normalised to the min and max
    deltaT = (data-np.min(data))/(np.max(data)-np.min(data))
    return(deltaT)

def plot_kde(ages, uncertainty=None, normalize = False):
    ages = ages.dropna()
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

def filter_older_ages(age_dist, uncertainty=None, gradient_cut_off = 0.3, young_index = 2, tflatmax = 0.25):
    while calc_flatness(age_dist, uncertainty, gradient_cut_off = gradient_cut_off, young_index = young_index) >= tflatmax:
        age_dist = age_dist[:-1]
    
    return(age_dist)

# Import data for TIMS age vs unc parameterisation
sava = pd.read_csv('TIMS_COMPILATION_v5.csv',encoding = "ISO-8859-1")
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


laser = pd.read_csv('LAICPMS_Comp_Cyril.csv')
# Let's fit an exponential function.  
# This looks like a line on a lof-log plot.
def laser_func(x, a, b):
    return a * np.power(x, b)
laser_popt, laser_pcov = curve_fit(laser_func, laser['68age'], laser['68 2s'])
        
