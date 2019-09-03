from glob import glob
import os

import numpy as np
import scipy.io as sio

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

csv2matrix:
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


def csv2matrix(subject_id, hemisphere, modeldir, mtype):

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
    mtype: string
        type of model (Power, Exponent)
    """

    subj_dir = '%s%s/' % (modeldir, subject_id)
    coef_file = '%s%s.%s.%s.Fit.Coefficients.csv' % (
        subj_dir, subject_id, hemisphere, mtype)
    coefs = pd.read_csv(coef_file, index_col=False)
    regions = list(coefs['target_region'].unique())
    regions.sort()

    nr = len(regions)

    reg_map = dict(zip(regions, np.arange(nr)))

    params = list(set(list(coefs.columns)).difference(
        ['source_region', 'target_region', 'aic', 'bic', 'name']))

    n, p = coefs.shape

    for param in params:
        for j in np.arange(n):

            temp_array = np.zeros((nr, nr))
            temp_data = coefs.iloc[j]

            temp_source = temp_data['source_region']
            temp_target = temp_data['target_region']

            temp_array[reg_map[temp_source], reg_map[temp_target]] = temp_data[param]
        
        temp = {param: temp_array}
        out_matrix = '%s%s.%s' % (subj_dir, subject_id, hemisphere, mtype, param)
        sio.savemat(file_name=out_matrix, mdict=temp)


def aggregate_model_fits(subject_id, hemisphere, modeldir, mtype):

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
    mtype: string
        Model type, in ['Power', 'Exponential', 'Linear']
    """

    subj_dir = '%s%s/' % (modeldir, subject_id)
    model_extension = '%s.%s.%s.Fit.Coefficients.*.2.*.csv' % (
        subject_id, hemisphere, mtype)
    model_files = glob('%s%s' % (subj_dir, model_extension))

    m = [None]*len(model_files)
    for i, model in enumerate(model_files):
        temp = pd.read_csv(model, index_col=False)
        m[i] = temp

    models = pd.concat(m, sort=False)
    models.index = np.arange(models.shape[0])
    out_coefs = '%s%s.%s.%s.Fit.Coefficients.csv' % (
        subj_dir, subject_id, hemisphere, mtype)
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

    subj_outdir = '%s%s/' % (outdir, subject_id)
    if not os.path.exists(subj_outdir):
        os.mkdir(subj_outdir)

    [_, X, y] = prep_data(subject_id, region_1, region_2, dir_label,
              dir_func, dir_dist, hemisphere, nsize)

    # g = density(X, y, region_1, region_2)
    M = fit(X, y)

    for model, mtype in zip(M, ['Exponential', 'Power', 'Linear']):
        save_model(outdir, mtype, subject_id, 
                    hemisphere, model, region_1, region_2)
    
    [_, X, y] = prep_data(subject_id, region_2, region_1, dir_label,
                       dir_func, dir_dist, hemisphere, nsize)

    # g = density(X, y, region_2, region_1)
    M = fit(X, y)

    for model, mtype in zip(M, ['Exponential', 'Power', 'Linear']):
        save_model(outdir, mtype, subject_id,
                   hemisphere, model, region_2, region_1)


def density(X, y, sreg, treg):

    """
    Plot the 2d-density of the size vs correlation data.

    Parameters:
    - - - - -
    X: float, array
        independent variable
    y: float, array
        dependent variable
    
    Returns:
    - - - -
    g: figure
        density plot
    """

    df = pd.DataFrame({'Size': X,
                       'Correlation': y})
    
    g = (ggplot(df, aes('Size', 'Correlation'))
         + geom_point(alpha=0.5, size=0.25)
         + geom_density_2d(size=1, color='r')
         + plotnine.ggtitle('Dispersion Correlations\n{:} --> {:}'.format(sreg, treg)))

    return g


def prep_data(subject_id, sreg, treg, dir_label, dir_func, dir_dist, hemisphere, nsize):

    """
    Source and target distance and correlation data for modeling.

    Parameters:
    - - - - -
    subject_id: string
        Subject name
    sreg, treg: string
        source, target region pair
    dir_label: string
        Directory where labels exist
    dir_func: string
        Directory where Nearest Neighbor correlation maps exist
    dir_dist: string:
        Directory where Nearest Neighbor distance maps exist
    hemisphere: string
        Hemisphere to process, in ['L', 'R']
    nsize: int
        neighborhood size

    Returns:
    - - - -
    inds: int, array
        indices of source voxels
    x: float, array
        dispersion vector
    y: float, array
        correlation vector
    """

    base_knn = '%s.%s.knn_mean.2.%s.func.gii' % (subject_id, hemisphere, treg)
    in_knn = '%s%s/%s' % (dir_func, subject_id, base_knn)
    knn = loaded.load(in_knn)

    base_dist = '%s.%s.Distance.2.%s.func.gii' % (subject_id, hemisphere, treg)
    in_dist = '%s%s/%s' % (dir_dist, subject_id, base_dist)
    dist = loaded.load(in_dist)

    in_label = '%s%s.%s.aparc.32k_fs_LR.label.gii' % (
        dir_label, subject_id, hemisphere)
    R = re.Extractor(in_label)
    region_map = R.map_regions()

    source_indices = region_map[sreg]
    nsize = nsize-1

    x = dist[source_indices, nsize]
    y = knn[source_indices, nsize]

    inds = np.arange(len(source_indices))[~np.isnan(y)]
    inds = inds[y[inds] != 0]

    x = x[inds]
    y = y[inds]

    return [inds, x, y]


def save_model(dir_out, mtype, subject_id, hemisphere, model, sreg, treg):

    """
    Method to save a model and plot the fit and residuals.

    Parameters:
    - - - - -
    dir_out: string
        Directory where model coefficients and plots are saved
    mtype: string
        Model type, in ['Power', 'Exponential', 'Linear']
    subject_id: string
        Subject name
    hemisphere: string
        Hemisphere to process, in ['L', 'R']
    model: lmfit model object
        fitted model
    sreg, treg: string
        source, target region pair
    """

    [F, gridspec] = model.plot()
    ax0 = F.axes[0]
    ax1 = F.axes[1]

    curr_title = ax0.get_title()
    ax0_title = curr_title + ' Residuals'
    ax1_tight = curr_title + ' Fit'
    ext = '%s to %s\n' % (sreg, treg)
    ax0.set_title(ext + ax0_title)
    ax1.set_title(ext + ax1_tight)
    F.tight_layout()

    data_dict = {'source_region': [sreg],
                 'target_region': [treg]}

    for k, v in model.params.valuesdict().items():
        data_dict[k] = v
    
    data_dict['aic'] = [model.aic]
    data_dict['bic'] = [model.bic]
    data_dict['name'] = [model.model.name]

    df = pd.DataFrame(data_dict)

    subj_ext = '%s%s/%s.%s.' % (dir_out, subject_id, subject_id, hemisphere)
    
    out_df = '%s%s.Fit.Coefficients.%s.2.%s.csv' % (subj_ext, mtype, sreg, treg)
    df.to_csv(out_df, index=False)

    out_fig = '%s%s.Fit.%s.2.%s.jpg' % (subj_ext, mtype, sreg, treg)
    F.savefig(out_fig)

def power(X, y):

    """
    Sub-method to fit a Power regression model.

    Returns:
    - - - -
    fitted: lmtfit model object
        fitted model
    """

    inds = np.argsort(X)
    X = X[inds]
    y = y[inds]

    model = models.PowerLawModel(
        independent_vars=['x'], nan_policy='propagate')
    fitted = model.fit(y, x=X)

    return fitted

def exponential(X, y):

    """
    Sub-method to fit an Exponential regression model.

    Returns:
    - - - -
    fitted: lmtfit model object
        fitted model
    """

    inds = np.argsort(X)
    X = X[inds]
    y = y[inds]

    model = models.ExponentialModel(
        independent_vars=['x'], nan_policy='propagate')
    fitted = model.fit(y, x=X)

    return fitted

def linear(X, y):

    """
    Sub-method to fit a Linear regression model.
    """

    inds = np.argsort(X)
    X = X[inds]
    y = y[inds]

    model = models.LinearModel(
        independent_vars=['x'], nan_policy='propagate')
    fitted = model.fit(y, x=X)

    return fitted


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

    Returns:
    - - - -
    e, p: lmfit model objects
        fitted exponential (e) and power (p) models
    """

    E = exponential(X, y)
    P = power(X, y)
    L = linear(X, y)

    return [E, P, L]


def plot_model_fit(fitted_model):
    
    """
    Sub-method to plot the fitted models.

    Returns:
    - - - -
    fig: matplotlib Figure object
    """

    [fig, gridspec] = fitted_model.plot(datafmt='o',
                                        data_kws={'alpha': 0.35},
                                        fit_kws={'c': 'r', 'linewidth': 3},
                                        fig_kws={'figsize': (12, 8)})

    return fig
