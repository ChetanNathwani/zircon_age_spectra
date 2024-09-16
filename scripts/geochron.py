import numpy as np
from scipy import stats
import ot
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import warnings

def normalize_age(ages):
    """
    Normalise zircon absolute ages to t_rel

    Args:
        ages (np.array): zircon ages in Ma

    Returns:
        np.array of zircon ages normalised between t_max and t_min
    """
    min_age = ages.min()
    max_age = ages.max()
    return (ages - min_age) / (max_age - min_age)

def calc_delT(data):

    """
    Calculate deltaT (max versus minimum zircon age)

    Args:
        ages (np.array): list of zircon ages in Ma

    Returns:
        np.array of zircon ages normalised between t_max and t_min
    """

    deltaT = (data-np.min(data))/(np.max(data)-np.min(data))
    return(deltaT)

def check_data(ages, unc):

    """
    Check zircon U-Pb age distributions for potential issues such as low delta_t/avg_sgima, high absolute age, low number of zircons, variable uncertainties

    Args:
        ages (np.array): list of zircon ages in Ma
        unc (np.array): list of zircon 2 sigma uncertainties in Ma 

    Returns:
        warnings where raised
    """
    
    if len(ages) < 10:
        warnings.warn("Distributions with low number of dates may not provide useful results")
    delta_T = np.max(ages) - np.min(ages)
    avg_sigma = np.mean(unc)
    if delta_T/avg_sigma < 10:
        warnings.warn("Distributions with low Δt/σ may not provide useful results")
    if np.mean(ages) > 120:
        warnings.warn("Old age distributions may not provide useful results, check for Pb loss")
    stdev_weights = np.std(((1/(unc/2)**2)/np.sum(1/(unc/2)**2)))
    if stdev_weights > 0.10:
        warnings.warn("Relative uncertainties show large variation, shape of age distribution may be dominated by variation in analytical uncertainty")

def plot_kde(ages, uncertainty=None, normalize = False):

    """
    Calculate x, y co-ordinates to plot kernel density plot from zircon age distribution

    Args:
        ages (np.array): list of zircon ages in Ma
        uncertainty (np.array): list of zircon 2 sigma uncertainties in Ma , default = None
        normalize (bool): whether to normalise the data between 0 and 1, default = False
    
    Returns:
        x (np.array): x-coordinates of kernel density plot
        y (np.array): y-coordinates of kernel density plot
    """
    
    
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

    """
    Calculate x, y co-ordinates to plot empirical cumulative distribution frequency curve from zircon age distribution

    Args:
        ages (np.array): list of zircon ages in Ma
        uncertainty (np.array): list of zircon 2 sigma uncertainties in Ma , default = None
    
    Returns:
        x (np.array): x-coordinates of ecdf
        y (np.array): y-coordinates of ecdf
    """
    
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

def TIMS_func(x, a, b, c):
    output_2s = a*x**2 + b*x + c
    return output_2s

tims_compilation = pd.read_csv('../data/zircon_tims_comp.csv',encoding = "ISO-8859-1")
tims_compilation = tims_compilation[tims_compilation['age68'] < 1000]
TIMS_popt, TIMS_pcov = curve_fit(TIMS_func, tims_compilation['age68'], tims_compilation['2s_68']) # Fit regression
# Define function to fit to data
def TIMS_uncer(age):

    pcov_sigma = np.sqrt(np.diag(TIMS_pcov))

    # Update parameters with gaussian uncertainty
    TIMS_popt_prop = [np.random.normal(TIMS_popt[0], pcov_sigma[0], 1),
                      np.random.normal(TIMS_popt[1], pcov_sigma[1], 1),
                      np.random.normal(TIMS_popt[2], pcov_sigma[2], 1)]
    
    output_2s = TIMS_popt_prop[0] * age **2  + TIMS_popt_prop[1] * age + TIMS_popt_prop[2]
    
    return float(output_2s)

def laser_func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

laser_compilation = pd.read_csv('../data/laicpms_comp.csv')
laser_popt, laser_pcov = curve_fit(laser_func, laser_compilation['68age'], laser_compilation['68 2s'])

def laser_uncer(age):
    
    pcov_sigma = np.sqrt(np.diag(laser_pcov))

    # Update parameters with gaussian uncertainty
    laser_popt_prop = [np.random.normal(laser_popt[0], pcov_sigma[0], 1),
                       np.random.normal(laser_popt[1], pcov_sigma[1], 1),
                       np.random.normal(laser_popt[2], pcov_sigma[2], 1),
                      np.random.normal(laser_popt[3], pcov_sigma[3], 1)]

    output_2s = laser_popt_prop[0] * age**3 + laser_popt_prop[1] * age**2 + laser_popt_prop[2] * age +laser_popt_prop[3] 

    return(output_2s)


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
    
    return PC_scores

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

def calc_W_PCA(ages, unc = None, check = True):

    if unc is None:
        unc = np.ones(len(ages))
    if check == True:
        check_data(ages, unc)

    w2_user = []
    for i in np.arange(0,len(values)):
        df2 = list(values)[i]
        w2 = calc_w2(ages,df2['age68'],x_err = unc, y_err = df2['2s_68'], normalize = True)
        w2_user.append(w2)
    pc_user = PCA_ws.transform(X = np.array(w2_user).reshape(1, -1)).flatten()
    return(pc_user)

def bootstrap_sampling(age, n_zircon, n_simulations, distribution = 'MELTS', method = 'ID-TIMS', truncation = 1.0, n_inh = 0, dx = 1):

    """
    Bootstrap sampling of a zircon age distribution for a given age.

    Args:
        age (float): The age to sample at in Ma.
        n_zircon (int): The number of zircons to sample from the age distribution.
        n_simulations (int): The number of simulations of bootstap sampling to perform.
        distribution (string or int): The underlying age distribution to sample from (default = MELTS):
            - string options (distributions of Keller C. B. (2018) chron.Jl library): MELTS, Triangular, Uniform, Volcanic Low Crystallinity, Volcanic, Half Normal,   Reverse Triangular
            - int options: an integer between 0 and 20 to sample from the Magma Chamber Simulator outputs of Tavazzani et al. (2023)
        method (string): whether to sample for uncertainties relevant to ID-TIMS or LA-ICP-MS (default = ID-TIMS)
        truncation (float): value between 0 and 1 for truncating the age distribution, (default = 1 i.e. no truncation)
        n_inh (int): number of inherited/antecrystal zircons to add
        dx (float or list): distance in relative time the antecrystal population is from the main distribution, if a list is given should be (min, max)
        and a random value is drawn from this range

    Returns:
        np.array of an a
    """
    
    synthetic_distributions = np.zeros((n_simulations, n_zircon+n_inh))
    
    if distribution == 'MELTS':
        dist = pd.read_csv('../data/synthetic/MeltsTZircDistribution.tsv', header = None)[0]
    if distribution == 'Triangular':
        dist = pd.read_csv('../data/synthetic/TriangularDistribution.tsv', header = None)[0]
    if distribution == 'Uniform':
        dist = pd.read_csv('../data/synthetic/UniformDistribution.tsv', header = None)[0]
    if distribution == 'Volcanic Low Crystallinity':
        dist = pd.read_csv('../data/synthetic/VolcanicZirconLowXDistribution.tsv', header = None)[0]
    if distribution == 'Volcanic':
        dist = pd.read_csv('../data/synthetic/VolcanicZirconDistribution.tsv', header = None)[0]
    if distribution == 'Half Normal':
        dist = pd.read_csv('../data/synthetic/HalfNormalDistribution.tsv', header = None)[0]
    if distribution == 'Reverse Triangular':
        dist = pd.read_csv('../data/synthetic/ReverseTriangularDistribution.tsv', header = None)[0]
    if isinstance(distribution, int):
        mcs_distributions = pd.read_csv('../data/synthetic/Tavazzani2023MCS.csv', header = None)
        dist = mcs_distributions[distribution][::-1]
        dist = dist.dropna()

    dist = dist.iloc[:int(np.rint(len(dist)*truncation))]
    dist = dist.iloc[::-1]
    probabilities = dist/np.sum(dist)
    if method == 'ID-TIMS':
        sigma = TIMS_uncer(age)/2 # 1 sigma at the given age
    if method == 'LA-ICP-MS':
        sigma = laser_uncer(age)/2 # 1 sigma at the given age
    n = 0

    while n < n_simulations:
        synthetic = pd.Series(np.random.choice(np.linspace(start = 0, stop = 1.0, num = len(probabilities)), n_zircon, p=list(probabilities)))
        sigmas = np.random.normal(0, sigma, size=(len(synthetic),)) # Randomly generate analytical uncertainties 
        synthetic = synthetic + sigmas

        if n_inh != 0:
            if isinstance(dx, list):
                dxn = np.random.uniform(dx[0],dx[1], 1)
            else:
                dxn = dx

            
            synthetic_2 = pd.Series(np.random.choice(np.linspace(start = 0, stop = 1.0, num = len(probabilities))+dxn, n_inh, p=list(probabilities)))
            sigmas = np.random.normal(0, sigma, size=(len(synthetic_2),)) # Randomly generate analytical uncertainties 
            synthetic_2 = synthetic_2 + sigmas
            # New distribution 
            synthetic = pd.concat([synthetic,synthetic_2])
        
        synthetic = np.sort(normalize_age(synthetic))
        synthetic_distributions[n] = synthetic
        n = n + 1

    return synthetic_distributions


def synthetic_distribution(distribution):
    if distribution == 'MELTS':
        dist = pd.read_csv('../data/synthetic/MeltsTZircDistribution.tsv', header = None)[0]
    if distribution == 'Triangular':
        dist = pd.read_csv('../data/synthetic/TriangularDistribution.tsv', header = None)[0]
    if distribution == 'Uniform':
        dist = pd.read_csv('../data/synthetic/UniformDistribution.tsv', header = None)[0]
    if distribution == 'Volcanic Low Crystallinity':
        dist = pd.read_csv('../data/synthetic/VolcanicZirconLowXDistribution.tsv', header = None)[0]
    if distribution == 'Volcanic':
        dist = pd.read_csv('../data/synthetic/VolcanicZirconDistribution.tsv', header = None)[0]
    if distribution == 'Half Normal':
        dist = pd.read_csv('../data/synthetic/HalfNormalDistribution.tsv', header = None)[0]
    if distribution == 'Reverse Triangular':
        dist = pd.read_csv('../data/synthetic/ReverseTriangularDistribution.tsv', header = None)[0]
    if isinstance(distribution, int):
        mcs_distributions = pd.read_csv('../data/synthetic/Tavazzani2023MCS.csv', header = None)
        dist = mcs_distributions[distribution][::-1]
        dist = dist.dropna()

    dist = dist.iloc[::-1]
    probabilities = dist/np.sum(dist)
    synthetic = pd.Series(np.random.choice(np.linspace(start = 0, stop = 1.0, num = len(probabilities)), 10000, p=list(probabilities)))

    x,y = plot_kde(synthetic, normalize = True)
    
    return(x,y)

