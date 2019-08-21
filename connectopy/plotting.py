from glob import glob
import os

import numpy as np

import lmfit
from lmfit import Model, models
from sklearn import linear_model as lm

from fragmenter import RegionExtractor as re
from niio import loaded, write

import plotnine
from plotnine import ggplot, geom_point, geom_density_2d, geom_line, aes, ggtitle, geom_boxplot, xlab

import matplotlib.pyplot as plt

import pandas as pd


"""
Set of methods for plotting the results of source-to-target
correlation vs. dispersion data.

csv_2_matrix:
    Convert aggregated coefficient csv files to directed adjacency matrices.
    
plot_regional:
    Visualize results of model fits using single region as source OR target

aggregate_model_coefficients:
    Combines all source-target pair coefficients into single file.

plot_dispersion:
    Visualize results of correlation as a function of spatial dispersion.
    Fits a Power Law model to data and saves model coefficients.

pair:
    Visualize results of correlation as function of spatial dispersion.
    Plots scatterplot and 2d-density.

fit:
    Sub-method.  Fits Power Law model.

plot_model_fit:
    Sub-method.  Plots fitted model.
"""


def csv_to_matrix(subject_id, hemisphere, modeldir):
    """
    Convert aggregated model coefficients to matrix representation.
    Rows of matrix represent source.  Columns of matric represent target.
    
    Parameters:
    - - - - -
    subject_id: string
        Subject name
    hemisphere: string
        L/R
    modeldir: string
        Path where model coefficients are saved
    """

    subj_dir = '%s%s/' % (modeldir, subject_id)
    coef_file = '%s%s.%s.Fit.Coefficients.csv' % (
        subj_dir, subject_id, hemisphere)
    coefs = pd.read_csv(coef_file, index_col=False)
    regions = list(coefs['target_region'].unique())
    regions.sort()

    nr = len(regions)

    reg_map = dict(zip(regions, np.arange(nr)))

    amplitude = np.zeros((nr, nr))
    exponent = np.zeros((nr, nr))

    n, p = coefs.shape

    for j in np.arange(n):

        temp = coefs.iloc[j]
        amplitude[reg_map[temp['source_region']],
                  reg_map[temp['target_region']]] = temp['amplitude']
        exponent[reg_map[temp['source_region']],
                 reg_map[temp['target_region']]] = temp['exponent']


def plot_regional(subject_id, hemisphere, modeldir):
    """
    For each region in cortical atlas, plot the model coefficients, 
    using that region as either the source or target.  OLS line is 
    fit to each data type and plotted
    
    Example usage:
        
    Using 'bankssts' as source region
    df.head()
    'source_region' 'target_region' 'exponent' 'amplitude'
     bankssts       frontalpole      1          2
     bankssts       temporalpole     1          3
     
    Using 'bankssts' as target region
    df.head()
    'source_region' 'target_region' 'exponent' 'amplitude'
     frontalpole    bankssts         1          2
     temporalpole   bankssts         1          3
    
    """

    subj_dir = '%s%s/' % (modeldir, subject_id)
    coef_file = '%s%s.%s.Fit.Coefficients.csv' % (
        subj_dir, subject_id, hemisphere)
    coefs = pd.read_csv(coef_file, index_col=False)

    [xmin, xmax] = coefs.amplitude.min(), coefs.amplitude.max()
    [ymin, ymax] = coefs.exponent.min(), coefs.exponent.max()

    regions = list(coefs['target_region'].unique())

    targets = {reg: None for reg in regions}
    sources = {reg: None for reg in regions}

    for reg in regions:

        targets[reg] = coefs[coefs['target_region'] == reg]
        sources[reg] = coefs[coefs['source_region'] == reg]

    L = lm.LinearRegression(fit_intercept=True)
    for reg in regions:

        xpred = np.linspace(xmin, xmax, 100)

        xtarg = targets[reg]['amplitude']
        ytarg = targets[reg]['exponent']
        L.fit(X=xtarg[:, None], y=ytarg)
        ytarg_hat = L.predict(xpred[:, None])

        xsour = sources[reg]['amplitude']
        ysour = sources[reg]['exponent']
        L.fit(X=xsour[:, None], y=ysour)
        ysour_hat = L.predict(xpred[:, None])

        fig = plt.figure(figsize=(12, 8))
        plt.scatter(xtarg, ytarg, marker='.', c='b', label='Target')
        plt.plot(xpred, ytarg_hat, c='b')

        plt.scatter(xsour, ysour, marker='.', c='r', label='Source')
        plt.plot(xpred, ysour_hat, c='r')

        plt.legend(fontsize=15)
        plt.xlabel('Amplitude', fontsize=15)
        plt.ylabel('Exponent', fontsize=15)
        plt.title(reg, fontsize=15)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        plt.savefig('%s%s.%s.%s.OLS.jpg' %
                    (subj_dir, subject_id, hemisphere, reg))
        plt.close()


def aggregate_model_coefficients(subject_id, hemisphere, modeldir):
    """
    Aggregate the coefficients across all region pairs for a single subject.
    
    Parameters:
    - - - - -
    subject_id: string
        Subject name
    hemisphere: string
        L/R
    modeldir: string
        Oath where model coefficients are saved
    """

    subj_dir = '%s%s/' % (modeldir, subject_id)
    model_extension = '%s.%s.Fit.Coefficients.*.2.*.csv' % (
        subject_id, hemisphere)
    model_files = glob('%s%s' % (subj_dir, model_extension))

    m = [None]*len(model_files)
    for i, model in enumerate(model_files):
        temp = pd.read_csv(model, index_col=False)
        m[i] = temp

    models = pd.concat(m, sort=False)
    models.index = np.arange(models.shape[0])
    out_coefs = '%s%s.%s.Fit.Coefficients.csv' % (
        subj_dir, subject_id, hemisphere)
    models.to_csv(out_coefs)

    return models


def plot_dispersion(subject_id, region_1, region_2, dir_label, dir_func, dir_dist, hemisphere, outdir,
                    nsize=5):
    
    """
    Method to generate functional connectivity dispersion plots.
    
    Two plots are generated:
        1. Density plots of dispersion (x) vs. correlation (y)
        2. Fitted exponential regression of dispersion (x) vs. correlation (y)
    
    Parameters:
    - - - - -
    subject_id: string
        Subject name
    region_1, region_2: strings
        Names of regions for which to analyze functional dispersion
    dir_label: string
        Directory where label files are located
    dir_func: string
        Directory where neighborhood functional maps are located
    dir_dist: string
        Directory where neighborhood distance maps are located
    outdir: string
        Output directory
    hemisphere: string
        L/R
    nsize: int
        Size of neighborhood mapping to consider
        Default = 5
    """

    sreg = []
    treg = []
    exponents = []
    amplitude = []

    subj_outdir = '%s%s/' % (outdir, subject_id)
    if not os.path.exists(subj_outdir):
        os.mkdir(subj_outdir)

    [g, x, y] = pair(subject_id, region_1, region_2, dir_label,
                     dir_func, dir_dist, hemisphere, nsize)
    print('X: {:}'.format(x.shape))
    print('Y: {:}'.format(y.shape))

    out_density = '%s%s.%s.Density.%s.2.%s.jpg' % (subj_outdir, subject_id,
                                                      hemisphere, region_1, region_2)

    model = fit(x, y)
    sreg.append(region_1)
    treg.append(region_2)
    exponents.append(model.best_values['exponent'])
    amplitude.append(model.best_values['amplitude'])

    p = plot_model_fit(model)
    out_fit = '%s%s.%s.Fit.%s.2.%s.jpg' % (subj_outdir, subject_id,
                                              hemisphere, region_1, region_2)
    p.savefig(out_fit)
    plt.close()

    coefs = {'source_region': [region_1],
             'target_region': [region_2],
             'exponent': [model.best_values['exponent']],
             'amplitude': [model.best_values['amplitude']]}
    coefs = pd.DataFrame(coefs)
    out_df = '%s%s.%s.Fit.Coefficients.%s.2.%s.csv' % (subj_outdir, subject_id,
                                                          hemisphere, region_1, region_2)
    coefs.to_csv(out_df, index=False)

    [g, x, y] = pair(subject_id, region_2, region_1, dir_label,
                     dir_func, dir_dist, hemisphere, nsize)
    print('X: {:}'.format(x.shape))
    print('Y: {:}'.format(y.shape))

    out_density = '%s%s.%s.Density.%s.2.%s.jpg' % (subj_outdir, subject_id,
                                                      hemisphere, region_2, region_1)

    model = fit(x, y)
    sreg.append(region_2)
    treg.append(region_1)
    exponents.append(model.best_values['exponent'])
    amplitude.append(model.best_values['amplitude'])
    p = plot_model_fit(model)
    out_fit = '%s%s.%s.Fit.%s.2.%s.jpg' % (subj_outdir, subject_id,
                                              hemisphere, region_2, region_1)
    p.savefig(out_fit)
    plt.close()
    coefs = {'source_region': [region_2],
             'target_region': [region_1],
             'exponent': [model.best_values['exponent']],
             'amplitude': [model.best_values['amplitude']]}
    coefs = pd.DataFrame(coefs)
    out_df = '%s%s.%s.Fit.Coefficients.%s.2.%s.csv' % (subj_outdir, subject_id,
                                                          hemisphere, region_2, region_1)
    coefs.to_csv(out_df, index=False)

def pair(subject_id, sreg, treg, dir_label, dir_func, dir_dist, hemisphere, nsize):
    
    """
    Sub method for a single direction analysis (source-to-target).
    """

    print('Source Region: {:}'.format(sreg))
    print('Target Region: {:}'.format(treg))

    base_knn = '%s.%s.knn_mean.2.%s.func.gii' % (subject_id, hemisphere, treg)
    in_knn = '%s%s/%s' % (dir_func, subject_id, base_knn)
    knn = loaded.load(in_knn)
    print('KNN: {:}'.format(knn.shape))

    base_dist = '%s.%s.Distance.2.%s.func.gii' % (subject_id, hemisphere, treg)
    in_dist = '%s%s/%s' % (dir_dist, subject_id, base_dist)
    dist = loaded.load(in_dist)
    print('DIST: {:}'.format(dist.shape))

    in_label = '%s%s.%s.aparc.32k_fs_LR.label.gii' % (
        dir_label, subject_id, hemisphere)
    R = re.Extractor(in_label)
    region_map = R.map_regions()

    source_indices = region_map[sreg]
    print('Source Inds: {:}'.format(source_indices.shape))
    nsize = nsize-1
    print('NSize: {:}'.format(nsize))

    x = dist[source_indices, nsize]
    print(x)
    y = knn[source_indices, nsize]
    print(y)

    print('X pair: {:}'.format(x.shape))
    print('Y pair: {:}'.format(y.shape))

    inds = np.arange(len(source_indices))[~np.isnan(y)]
    inds = inds[y[inds] != 0]
    print('Inds: {:}'.format(inds.shape))

    x = x[inds]
    y = y[inds]

    print('X pair inds: {:}'.format(x.shape))
    print('Y pair inds: {:}'.format(y.shape))

    df = pd.DataFrame({'Size': x[np.argsort(x)],
                       'Correlation': y[np.argsort(x)]
                       })

    g = (ggplot(df, aes('Size', 'Correlation'))
         + geom_point(alpha=0.5, size=0.25)
         + geom_density_2d(size=1, color='r')
         + plotnine.ggtitle('Dispersion Correlations\n{:} --> {:}'.format(sreg, treg)))

    return [g, x, y]


def fit(X, y):
    
    """
    Sub-method to fit the correlation vs dispersion data.
    
    Parameters:
    - - - - - 
    model_function: function
        The function you wish to fit to the data.
        In the dispersion case, we are using an exponential decay model.
        
    x: float, array
        independent variable i.e. spatial dispersion
    y: float, array
        dependent variable i.e. spatial correlation
    """

    inds = np.argsort(X)
    model = models.PowerLawModel(
        independent_vars=['x'], nan_policy='propagate')
    fitted = model.fit(y[inds], x=X[inds])

    return fitted


def plot_model_fit(fitted_model):
    
    """
    Sub-method to plot the fitted models.
    """

    [fig, gridspec] = fitted_model.plot(datafmt='o',
                                        data_kws={'alpha': 0.35},
                                        fit_kws={'c': 'r', 'linewidth': 3},
                                        fig_kws={'figsize': (12, 8)})

    return fig
