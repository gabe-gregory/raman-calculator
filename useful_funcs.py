from sys import path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# import oitg.fitting
# import oitg.errorbars
# import oitg.results as results
import os.path
# from statsmodels.stats.proportion import proportion_confint
from matplotlib.pyplot import figure
# import math
# from scipy.optimize import minimize
# from scipy.stats import binom
from qutip import *

# reading in data

def get_path(rid, day=None):
    location = 'Z:\\Users\\Gregory\\Ca40_multi_photon\\experiment_data\\data\\'  
    for root, dirs, files in os.walk(location):
        for name in files:
            if rid in name:
                path = os.path.join(root, name)
    return path

def get_data(path):
    data = results.load_hdf5_file(path)
    pop = data['datasets']['ndscan.points.channel_ion_bright']
    x = data['datasets']['ndscan.points.axis_0']
    return x, pop

def get_data_qudit_time(path):
    data = results.load_hdf5_file(path)
    x = data['datasets']['ndscan.points.axis_0']
    m52 = data['datasets']['ndscan.points.channel_ion_tag_m52']
    m32 = data['datasets']['ndscan.points.channel_ion_tag_m32']
    m12 = data['datasets']['ndscan.points.channel_ion_tag_m12']
    p12 = data['datasets']['ndscan.points.channel_ion_tag_p12']
    p32 = data['datasets']['ndscan.points.channel_ion_tag_p32']
    p52 = data['datasets']['ndscan.points.channel_ion_tag_p52']
    x = data['datasets']['ndscan.points.axis_0']
    rejected_shots = data['datasets']['ndscan.points.channel_rejected_shots']
    return x, m52, m32,m12,p12,p32,p52,rejected_shots 


# fitting functions

def Sinc(x,A,x0,w):
    return A*((np.sin(w*(x-x0)))/(w*(x-x0)))**2

def fitSinc(x_data, y_data, p0_guess):
    popt, pcov = curve_fit(Sinc, x_data, y_data, p0 = p0_guess)
    x_fit = np.linspace(min(x_data), max(x_data),600)
    y_fit = Sinc(x_fit, *popt)
    error = np.sqrt(np.diag(pcov))
    return popt, [x_fit,y_fit], error


def SinFull(x,tpi,phi, A):
    return A*np.sin(np.pi*x/(2*tpi) + phi)**2

def fitSinFull(x_data, y_data, p0_guess, bounds):
    popt, pcov = curve_fit(SinFull, x_data, y_data, p0 = p0_guess, bounds = bounds)
    x_fit = np.linspace(min(x_data), max(x_data),600)
    y_fit = SinFull(x_fit, *popt)
    error = np.sqrt(np.diag(pcov))
    return popt, [x_fit,y_fit], error


def quad(t, t0,m, b):
    return m*((t-t0)**2) + b

def fitQuad(x_data, y_data, p0_guess):
    popt, pcov = curve_fit(quad, x_data, y_data, p0 = p0_guess)
    x_fit = np.linspace(min(x_data), max(x_data),600)
    y_fit = quad(x_fit, *popt)
    error = np.sqrt(np.diag(pcov))
    return popt, [x_fit,y_fit], error

def Sin(x,A,phi,y0):
    return A*np.sin(2*np.pi*x + phi) + y0

def fitSin(x_data, y_data, p0_guess):
    popt, pcov = curve_fit(Sin, x_data, y_data, p0 = p0_guess)
    x_fit = np.linspace(min(x_data), max(x_data),600)
    y_fit = Sin(x_fit, *popt)
    error = np.sqrt(np.diag(pcov))
    return popt, [x_fit,y_fit], error

    return A*np.sin(w*x + phi) + y0

def Gaus(x,σ):
    return np.exp(-((x)**2)/(2*(σ**2)))

def fitGaus(x_data, y_data, p0_guess):
    bounds = ((-np.inf, 0))
    popt, pcov = curve_fit(Gaus, x_data, y_data, p0 = p0_guess)
    x_fit = np.linspace(min(x_data), max(x_data),600)
    y_fit = Gaus(x_fit, *popt)
    error = np.sqrt(np.diag(pcov))
    return popt, [x_fit,y_fit], error