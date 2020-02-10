# -*- coding: utf-8 -*-
# @Author: joaopn
# @Date:   2020-02-09 13:45:12
# @Last Modified by:   joaopn
# @Last Modified time: 2020-02-10 14:41:38

import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import InterpolatedUnivariateSpline

'''
TODO
- Parse both single timeseries and lists of avalanches
- Substitute RMSE fitting by MLE
'''

def run_analysis(data, newFig=True, label='Data', color='k'):


	#Sets up figure
	if newFig is True:
		fig = plt.figure(figsize=(18,12))
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
	
	avalanches = get_avalanches(data)
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

	#Plots p(S)
	fit_pS = powerlaw.Fit(S_list,xmin=1)
	str_label = label + r': $\alpha$ = {:0.3f}'.format(fit_pS.power_law.alpha)
	fit_pS.plot_pdf(ax=ax_pS, color=color, **{'label':str_label})
	fit_pS.power_law.plot_pdf(ax=ax_pS,color='k', linestyle='--')

	#Plots p(D)
	fit_pD = powerlaw.Fit(D_list,xmin=1)
	str_label = label + r': $\beta$ = {:0.3f}'.format(fit_pD.power_law.alpha)
	fit_pD.plot_pdf(ax=ax_pD, color= color, **{'label':str_label})
	fit_pD.power_law.plot_pdf(ax=ax_pD, color='k', linestyle='--')

	#Plots <S>(D)
	fit_gamma,_,_ = fit_powerlaw(S_avg[:,0],S_avg[:,1],S_avg[:,2], loglog=True)
	str_label = label + r': $\gamma$ = {:0.3f}'.format(fit_gamma)
	ax_avgS.plot(S_avg[:,0],S_avg[:,1], label=str_label, color=color)

	#Fits and plots the average avalanche shape
	fit_gamma_shape = fit_collapse(shape_list, 4, 20, extrapolate=True)
	print(label + 'gamma_shape = {:0.2f}'.format(fit_gamma_shape))
	#fit_gamma_shape = 2.0
	str_leg = label + r': $\gamma_s$ = {:0.2f}'.format(fit_gamma_shape)
	plot_collapse(shape_list,fit_gamma_shape,4,20,ax_shape, None, True, color, show_subplots=False)

	#Beautifies plots
	plt.sca(ax_pS)
	plt.legend(loc='upper right')
	plt.xlabel('S')
	plt.ylabel('p(S)')
	plt.sca(ax_pD)
	plt.legend(loc='upper right')
	plt.xlabel('D')
	plt.ylabel('p(D)')
	plt.sca(ax_avgS)
	plt.legend(loc='upper left')
	plt.xlabel('D')
	plt.ylabel(r'$\langle S \rangle$ (D)')
	plt.xlim([1,1e3])
	plt.ylim([1,1e5])
	ax_avgS.set_xscale('log')
	ax_avgS.set_yscale('log')
	plt.sca(ax_shape)
	plt.legend()

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
			plt.plot(avg_shape_i_x, avg_shape_i_y, alpha=0.2, color=color)

	#Plots interpolated average curve
	if show_subplots:
		color_collapse = 'k'
	else:
		color_collapse = color
	plt.plot(x_range, np.mean(average_shape, axis=0), color=color_collapse, linewidth=2)

	#Beautifies plot
	ax.set_xlabel('Scaled time')
	ax.set_ylabel('Scaled activity')
	plt.xlim([0,1])
	if extrapolate is True:
		_,ylim_1 = ax.get_ylim()
		ax.set_ylim([y_min, ylim_1])
	if str_leg is not None:
		#plt.text(0.95, 0.95, str_leg, horizontalalignment='right',verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='none', alpha=0.5))
		plt.text(0.95, 0.95, str_leg, horizontalalignment='right',verticalalignment='top', transform=ax.transAxes)

def get_avalanches(data):
	"""Returns a dict with shape, size and duration of all avalanches
	
	Args:
		data (float): thresholded timeseries with events
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