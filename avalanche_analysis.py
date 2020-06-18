# -*- coding: utf-8 -*-
# @Author: joaopn
# @Date:   2020-02-09 13:45:12
# @Last Modified by:   joaopn
# @Last Modified time: 2020-02-10 14:41:38

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import powerlaw
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import InterpolatedUnivariateSpline

'''
TODO
- Parse both single timeseries and lists of avalanches
- Substitute RMSE fitting by MLE
'''

def full_analysis(data, newFig=True, label='Data', color='k', comparison = 'fit', collapse_subplots = False, collapse_extrapolate=False, savefig=False):
    """Runs the complete avalanche analysis.
    
    Args:
        data (float): Input data, either an ndarray with a thresholded (1D) or trial-based structure, or a dict.
        newFig (bool, optional): whether to use a new figure for plotting
        label (str, optional): label to plot
        color (str, optional): color to plot.
        comparison (str, optional): which line to compare to. Values: 'fit', 'BP', None
        collapse_subplots (bool, optional): whether to show the individual shapes
        collapse_extrapolate (bool, optional): whether to extrapolate shape collapse beyond the data limit
    """
    #Sets up figure
    if newFig is True:
        fig = plt.figure(figsize=(8,6))
        gs = fig.add_gridspec(2,2)
        ax_pS = fig.add_subplot(gs[0,0])
        ax_pD = fig.add_subplot(gs[0,1])
        ax_avgS = fig.add_subplot(gs[1,0])
        ax_shape = fig.add_subplot(gs[1,1])
    else:
        fig = plt.gcf()
        if len(fig.get_axes()) != 4:
            ValueError('Current figure does not have a 2x2 layout.')
        ax_pS,ax_pD,ax_avgS,ax_shape = fig.get_axes()

    #Analyzes avalanches
    if type(data) == np.ndarray:
        avalanches = get_avalanches(data)
    elif type(data) == dict:
        avalanches = data
    else:
        ValueError('Invalid data type')

    #Builds individual lists
    S_list = [avalanches[i]['S'] for i in avalanches.keys()]
    D_list = [avalanches[i]['D'] for i in avalanches.keys()]
    shape_list = [avalanches[i]['shape'] for i in avalanches.keys()]

    #Calculates S_avg
    S_avg = np.zeros((np.max(D_list),3))
    for i in range(np.max(D_list)):
        S_avg[i,0] = i+1
        S_D = [avalanches[j]['S'] for j in avalanches.keys() if avalanches[j]['D'] == i+1]
        S_avg[i,1] = np.mean(S_D)
        S_avg[i,2] = np.std(S_D)

    #Fits distributions
    fit_pS = powerlaw.Fit(S_list,xmin=1)
    fit_pD = powerlaw.Fit(D_list,xmin=1)
    fit_gamma,_,_ = fit_powerlaw(S_avg[:,0],S_avg[:,1],S_avg[:,2], loglog=True)
    fit_gamma_shape = fit_collapse(shape_list, 4, 20, extrapolate=True)

    #Plots comparison lines
    if comparison == 'fit':
        fit_pS.power_law.plot_pdf(ax=ax_pS,color='k', linestyle='--')
        fit_pD.power_law.plot_pdf(ax=ax_pD, color='k', linestyle='--')
        ax_avgS.plot(S_avg[:,0], np.power(S_avg[:,0], fit_gamma), color='k', linestyle='--')

        str_label_S = label + r': $\alpha$ = {:0.3f}'.format(fit_pS.power_law.alpha)
        str_label_D = label + r': $\beta$ = {:0.3f}'.format(fit_pD.power_law.alpha)
        str_label_AVG = label + r': $\gamma$ = {:0.3f}'.format(fit_gamma)
        str_label_shape = label + r': $\gamma_s$ = {:0.2f}'.format(fit_gamma_shape)

    elif comparison == 'BP':
        BP_S = [3,1e4]
        BP_D = [3,1e2]
        BP_AVG = [1,1e2]

        ax_pS.plot(BP_S, 5*np.power(BP_S, -1.5), color='k', linestyle='--')
        ax_pD.plot(BP_D, 5*np.power(BP_D, -2), color='k', linestyle='--')
        ax_avgS.plot(BP_AVG, 1 + np.power(BP_AVG, 2), color='k', linestyle='--')

        str_label_S = label
        str_label_D = label
        str_label_AVG = label    
        str_label_shape = label

    #Plots distributions
    fit_pS.plot_pdf(ax=ax_pS, color=color, **{'label':str_label_S, 'lw':2})
    fit_pD.plot_pdf(ax=ax_pD, color= color, **{'label':str_label_D, 'lw':2})
    ax_avgS.plot(S_avg[:,0],S_avg[:,1], label=str_label_AVG, color=color, lw=2)
    
    #Plots the average avalanche shape
    str_leg_shape = label + r': $\gamma_s$ = {:0.2f}'.format(fit_gamma_shape)
    plot_collapse(shape_list,fit_gamma_shape,4,20,ax_shape, str_label_shape, extrapolate=collapse_extrapolate, color=color, show_subplots=collapse_subplots)

    #Prints results
    print('== Exponents for {:s} =='.format(label))
    print('alpha = {:0.3f}'.format(fit_pS.power_law.alpha))
    print('beta = {:0.3f}'.format(fit_pD.power_law.alpha))
    print('gamma_scaling = {:0.3f}'.format((fit_pD.power_law.alpha-1)/(fit_pS.power_law.alpha-1)))
    print('gamma = {:0.3f}'.format(fit_gamma))
    print('gamma_shape = {:0.3f}'.format(fit_gamma_shape))

    #Beautifies plots
    plt.sca(ax_pS)
    plt.legend(loc='upper right')
    plt.xlabel('Avalanche size S')
    plt.ylabel('p(S)')
    plt.xlim(left=1)

    plt.sca(ax_pD)
    plt.legend(loc='upper right')
    plt.xlabel('Avalanche duration D')
    plt.ylabel('p(D)')
    plt.xlim(left=1)

    plt.sca(ax_avgS)
    plt.legend(loc='upper left')
    plt.xlabel('Avalanche duration D')
    plt.ylabel(r'Average avalanche size $\langle S \rangle$ (D)')
    plt.xlim(left=1)
    ax_avgS.set_xscale('log')
    ax_avgS.set_yscale('log')

    # #Fixes minor ticks
    # locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    # locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),numticks=12)
    # for axis in [ax_pS, ax_pD]:
    #     x_current = axis.get_xlim()
    #     axis.set_xlim([1,1e10])

    #     #axis.xaxis.set_major_locator(locmaj)
    #     axis.xaxis.set_minor_locator(locmin)
    #     axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    #     axis.yaxis.set_major_locator(locmaj)
    #     axis.yaxis.set_minor_locator(locmin)
    #     axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    #     axis.set_xlim(x_current)

    plt.tight_layout()

    if savefig is not False:
        
        plt.savefig(savefig, dpi=200)

def fit_collapse(flat_list, min_d, min_rep, extrapolate=False):

    #Definitions
    interp_points = 1000
    gamma_x0 = 0.5
    opt_bounds = (-1,5)

    #Flattens list of reps
    #flat_list = np.array([item for sublist in shape_list for item in sublist])
    flat_list = np.array(flat_list)

    #List of avalanche sizes
    shape_size = np.zeros(len(flat_list))
    for i in range(len(flat_list)):
        shape_size[i] = flat_list[i].size

    max_size = shape_size.max()

    #Avalanche size count
    shape_count,_ = np.histogram(shape_size,bins=np.arange(0,max_size+2))

    #Censors data by size
    censor_d_keep = np.arange(0,max_size+1) >= min_d
    censor_rep_keep = shape_count >= min_rep
    censor_index =  np.where([a and b for a, b in zip(censor_d_keep, censor_rep_keep)])[0]

    #Defines average size matrix
    average_shape = np.zeros((censor_index.size, interp_points))

    #Defines bottom interpolation range from data, to prevent extrapolation bias
    if extrapolate is True:
        x_min = 0
    elif extrapolate is False:
        x_min = 1/censor_index[0]
    else:
        error('extrapolate is not binary.')
    x_range = np.linspace(x_min,1,num=interp_points)

    #Averages shape for each duration and interpolates results
    for i in range(len(censor_index)):

        #Calculates average shape
        size_i = censor_index[i]
        avg_shape_i_y = np.mean(flat_list[shape_size==size_i])
        avg_shape_i_x = np.arange(1,size_i+1)/size_i

        #Interpolates results
        fx = InterpolatedUnivariateSpline(avg_shape_i_x,avg_shape_i_y)
        average_shape[i,:] = fx(x_range)

    #Error function for optimization    
    def _error(gamma_shape, *params):
        average_shape, censor_index = params
        shape_scaled = np.zeros((censor_index.size, interp_points))
        for i in range(censor_index.size):
            shape_scaled[i,:] = average_shape[i,:]/np.power(censor_index[i],gamma_shape)

        err = np.mean(np.var(shape_scaled, axis=0))/np.power((np.max(np.max(shape_scaled))-np.min(np.min(shape_scaled))),2)
        return err

    #Minimizes error
    minimize_obj = minimize(_error, x0=[gamma_x0], args=(average_shape,censor_index), bounds=[opt_bounds])

    return minimize_obj.x[0] + 1

def fit_powerlaw(X,Y,Yerr, loglog=False):
    #Parameters
    kwargs = {'maxfev': 100000}

    if loglog:
        def lin(x,a,b):
            return a*x+b

        bool_keep = np.less(0,Y)
        X = X[bool_keep]
        Y = Y[bool_keep]
        Yerr = Yerr[bool_keep]

        X = np.log10(X)
        Y = np.log10(Y)
        Yerr = np.divide(Yerr,Y)

        results, pcov = curve_fit(lin,X,Y, **kwargs)
        #results, pcov = curve_fit(lin,X,Y, method='lm', p0=[0,2], **kwargs)
        lin_coef = 10**results[1]
        fit_exp = results[0]

    else:
        def pl(x,a,b):
            return a*np.power(x,b)

        #Fits courve with a LM algorithm
        #results, pcov = curve_fit(pl,X,Y,sigma=Yerr, method='lm', p0=[2,0.1], **kwargs)
        results, pcov = curve_fit(pl,X,Y, **kwargs)
        lin_coef = results[0]
        fit_exp = results[1]

    #Gets error
    fit_err_all = np.sqrt(np.diag(pcov))

    #Gets relevant variables

    fit_err = fit_err_all[1]

    return fit_exp, fit_err, lin_coef

def plot_collapse(flat_list, gamma_shape, min_rep=10,min_d = 4, ax=None, str_leg = None, extrapolate=False, color='r', show_subplots=True):

    #Definitions
    interp_points = 1000

    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)

    #Flattens list of reps
    #flat_list = np.array([item for sublist in shape_list for item in sublist])
    flat_list = np.array(flat_list)

    #List of avalanche sizes
    shape_size = np.zeros(len(flat_list))
    for i in range(len(flat_list)):
        shape_size[i] = flat_list[i].size

    max_size = shape_size.max()

    #Avalanche size count
    shape_count,_ = np.histogram(shape_size,bins=np.arange(0,max_size+2))

    #Censors data by size
    censor_d_keep = np.arange(0,max_size+1) >= min_d
    censor_rep_keep = shape_count >= min_rep
    censor_index =  np.where([a and b for a, b in zip(censor_d_keep, censor_rep_keep)])[0]

    #Defines average size matrix
    average_shape = np.zeros((censor_index.size, interp_points))

    #Defines bottom interpolation range from data, to prevent extrapolation bias
    #x_min = 1/censor_index[0]
    if extrapolate is True:
        x_min = 0
    elif extrapolate is False:
        x_min = 1/censor_index[0]
    else:
        error('extrapolate is not binary.')
    x_range = np.linspace(x_min,1,num=interp_points)

    #Averages shape for each duration and interpolates results
    y_min = 100
    for i in range(len(censor_index)):

        #Calculates average shape
        size_i = censor_index[i]
        avg_shape_i_y = np.mean(flat_list[shape_size==size_i])/np.power(size_i,gamma_shape-1)
        avg_shape_i_x = np.arange(1,size_i+1)/size_i

        if np.min(avg_shape_i_y) < y_min:
            y_min = np.min(avg_shape_i_y)

        #Interpolates results
        fx = InterpolatedUnivariateSpline(avg_shape_i_x,avg_shape_i_y)
        average_shape[i,:] = fx(x_range)

        #Plots transparent subplots
        if show_subplots:
            ax.plot(avg_shape_i_x, avg_shape_i_y, alpha=0.2, color=color)

    #Plots interpolated average curve
    if show_subplots:
        color_collapse = 'k'
    else:
        color_collapse = color
    plot_line, = ax.plot(x_range, np.mean(average_shape, axis=0), color=color_collapse, linewidth=2, label=str_leg)
    ax.legend([plot_line], [str_leg])
    plt.legend()

    #Beautifies plot
    ax.set_xlabel('Scaled time')
    ax.set_ylabel('Scaled activity')
    plt.xlim([0,1])
    # if extrapolate is True:
    #   _,ylim_1 = ax.get_ylim()
    #   ax.set_ylim([y_min, ylim_1])
    # if str_leg is not None:
    #   #plt.text(0.95, 0.95, str_leg, horizontalalignment='right',verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='none', alpha=0.5))
    #   plt.text(0.95, 0.95, str_leg, horizontalalignment='right',verticalalignment='top', transform=ax.transAxes)

def get_avalanches(data):
    """Returns a dict of avalanche details with shape, size and duration of all avalanches. 
    Selects function based on the data structure: 
    - if 1D, assumes a single timeseries with timescale separation. 
    - if 2D, assumes a trial structure where the first index is the trial number.
    
    Args:
        data (ndarray): Input data
    """

    data = np.array(data)

    if data.ndim == 1:
        return _get_avalanches_single(data)
    elif data.ndim == 2:
        return _get_avalanches_trial(data)
    else:
        ValueError('data.ndim > 2')

def _get_avalanches_single(data):
    """Returns a dict with shape, size and duration of all avalanches
    
    Args:
        data (float): single timeseries with events
    """

    #Dict fields
    observables = ['shape', 'S', 'D']
    avalanches = {}

    #Finds crossings of the signal to positive
    id_cross = np.where(np.sign(data[:-1]) != np.sign(data[1:]))[0] + 1
    id_cross_plus = id_cross[data[id_cross]>0]

    #Obtain avalanche properties
    n_avalanches = id_cross_plus.size

    #avalanches = dict.fromkeys(range(0,n_avalanches))

    for i in range(n_avalanches-1):
        avalanches[i] = dict.fromkeys(observables)
        avalanches[i]['shape'] = np.trim_zeros(data[id_cross_plus[i]:id_cross_plus[i+1]],'b')
        avalanches[i]['S'] = int(np.sum(data[id_cross_plus[i]:id_cross_plus[i+1]]))
        avalanches[i]['D'] = int(np.sum(data[id_cross_plus[i]:id_cross_plus[i+1]]!=0))

    return avalanches

def _get_avalanches_trial(data):
    """Returns a dict with shape, size and duration for all avalanches, from a trial structure.
    
    Args:
        data (ndarray): avalanche timeseries ndarray [num_trials, len_trials]
    """

    #Dict fields
    observables = ['shape', 'S', 'D']
    avalanches = {}

    #Gets non-empty trials
    trial_index = np.where(np.sum(data, axis=1) > 0)[0]
    n_avalanches = len(trial_index)

    for i in trial_index:
        avalanches[i] = dict.fromkeys(observables)
        avalanches[i]['shape'] = np.trim_zeros(data[i])
        avalanches[i]['S'] = int(np.sum(data[i]))
        avalanches[i]['D'] = len(avalanches[i]['shape'])

    return avalanches

def simulate_bp(m, trials, timesteps = 10000, A0=1):
    """Simulates a simple branching process without drive.
    
    Args:
        m (float): branching parameter
        trials (int): number of trials to return
        timesteps (int, optional): maximum number of timesteps in each trial
        A0 (int, optional): number of units to activate in the beginning
    
    Returns:
        ndarray: array (trials, timesteps) with activity from the bp.
    """
    At = np.zeros((trials, timesteps), dtype=int)
    At[:,0] = int(A0)

    for i in range(trials):
        for j in range(1,timesteps):
            At[i,j] = np.sum(np.random.poisson(m,At[i,j-1]))
            if At[i,j] == 0:
                break

    return At