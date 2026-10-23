import numpy as np
from qutip import *
from qutip.ui.progressbar import BaseProgressBar
# from PhotonScatteringFuncs import *
from scipy.signal import find_peaks
from tqdm.notebook import tqdm
import sympy as sp
from scipy.signal import find_peaks
import useful_funcs
import matplotlib.pyplot as plt
from pathlib import Path


from .PhotonScatteringFuncs import *
# from . import useful_funcs

class StitchedProgressBar(BaseProgressBar):
    def __init__(self, callback, solve_index=0, n_solves=1):
        self.callback = callback
        self.solve_index = solve_index
        self.n_solves = n_solves

    def start(self, iterations, **kwargs):
        self.iterations = iterations

    def update(self, n):
        local = n / max(self.iterations - 1, 1)
        self.callback((self.solve_index + local) / self.n_solves)

    def finished(self):
        self.callback((self.solve_index + 1) / self.n_solves)

class sim:
    def __init__(self,
                 specie = '40Ca', 
                 transition = '+5/2<->-1/2',
                 Pr = 0.195,   # power in Rpi
                 w0r = 30e-6,   # Rpi waist
                 r_sp = 0,   # fraction of rpi power in sigma + 
                 r_pi = 0,   # fraction pf rpi power in pi
                 Pb = 0.18,   # power in Rsig
                 w0b = 30e-6,   # Rsig waist
                 b_sp = 1/3,   # fraction of rsig power in sigma + 
                 b_pi = 1/3,   # fraction of rsig power in pi
                 ω = 2*np.pi*3.71e6,   # raman beam detuning
                 ω0 = 2*np.pi*2.6338*1e6,   # D5/2 zeeman splitting
                 Δ = -2*np.pi*44e12,   # P3/2 detuning
                 Δc = -2*np.pi*658e12,   # P3/2 detuning (counter rotating paths)
                 ΔF = -2*np.pi*1322e12,   # F7/2 detuning,
                 Δls = -2*np.pi*1e12,   # lightshift beam detuning
                 Pls = 0,   # power of lightshift beam
                 w0ls = 30e-6,   # lightshift beam waist
                 ls_sp = 0,   # fraction of lightshift beam power in sigma + 
                 ls_pi = 0,   # fraction pf lightshift beam power in pi
                 carrier = False,   # True if driving +5/2<->+3/2 two-photon
                 pred_eighth_order = False,   # predict eighth order shift to resonance when calculating rabi frequencies?
                 ls_854 = 0,   # Lightshift on +1/2 from 854 beam (0 if carrier = False)
                 PS_power = 0,   # sin ramp exponent in pulse shaping
                 counter_rot = False,   # True if include counter rotating terms in sim / analytics
                 F_states = False,
                 ϕ = 0,   # phase difference between Raman beams
                 ls_order = 2,   # order to which a.c. stark shifts are calculated
                 rabi_order = 4):   # order to which four-photon analytic rabi freq is calculated
        
        self.specie = specie
        self.transition = transition
        self.Pr = Pr
        self.w0r = w0r
        self.r_sp = r_sp
        self.r_pi = r_pi
        self.Pls = Pls
        self.w0ls = w0ls
        self.ls_sp = ls_sp
        self.ls_pi = ls_pi
        self.Pb = Pb
        self.w0b = w0b
        self.b_sp = b_sp
        self.b_pi = b_pi
        self.ω = ω
        self.ω0 = ω0
        self.ω01 = self.ω0
        self.ω12 = self.ω0
        self.ω23 = self.ω0
        self.ω34 = self.ω0
        self.ω45 = self.ω0
        self.Δ = Δ
        self.Δc = Δc
        self.ΔF = ΔF
        self.Δls = Δls
        self.carrier = carrier
        self.ls_854 = ls_854
        self.PS_power = PS_power
        self.counter_rot = counter_rot
        self.F_states = F_states
        self.pred_eighth_order = pred_eighth_order
        self.ϕ = ϕ
        self.ls_order = ls_order
        self.PS_power = PS_power
        self.rabi_order = rabi_order
        
        # Experiment data, taken from Ca40_multiphoton / experiment data / data_analysis.ipynb
        
        self.RID_rpi = np.array([180.0, 152.0, 130.0, 112.0, 98.0, 86.0, 75.0, 67.0, 60.0, 54.5, 50.0, 45.0, 41.5, 38.0, 34.5, 32.0, 30.0, 27.8, 26.0, 24.3, 22.8, 21.4, 19.9, 18.8])
        self.tpi_2 = np.array([1.31e-06, 1.41e-06, 1.52e-06, 1.64e-06, 1.73e-06, 1.81e-06, 1.95e-06, 2.05e-06, 2.17e-06, 2.28e-06, 2.35e-06, 2.46e-06, 2.6e-06, 2.71e-06, 2.81e-06, 2.93e-06, 2.97e-06, 3.16e-06, 3.26e-06, 3.34e-06, 3.5e-06, 3.56e-06, 3.69e-06, 3.8e-06])
        self.δtpi_2 = np.array([1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 2e-08, 1e-08, 2e-08, 2e-08, 2e-08, 3e-08, 2e-08, 3e-08, 3e-08, 2e-08, 3e-08, 4e-08, 4e-08, 4e-08, 5e-08, 5e-08])
        self.tpi_4 = np.array([2.329e-05, 2.729e-05, 3.21e-05, 3.72e-05, 4.199e-05, 4.863e-05, 5.5e-05, 6.106e-05, 6.764e-05, 7.375e-05, 8.147e-05, 8.712e-05, 9.771e-05, 0.00010537, 0.0001168, 0.00012667, 0.00013316, 0.00013962, 0.00015292, 0.00016413, 0.00017433, 0.00018492, 0.00019531, 0.00020794])
        self.δtpi_4 = np.array([1e-07, 1.3e-07, 2.2e-07, 1.6e-07, 2.4e-07, 2.6e-07, 2.3e-07, 3.9e-07, 4.6e-07, 5.5e-07, 5.5e-07, 1.01e-06, 6.3e-07, 8.7e-07, 7.4e-07, 7.1e-07, 8.9e-07, 8e-07, 9.2e-07, 2.41e-06, 1.51e-06, 3.75e-06, 5.9e-06, 3.95e-06])
        self.tpi_6 = np.array([0.00013909, 0.00017321, 0.00020856, 0.00026433, 0.00035042])
        self.δtpi_6 = np.array([1.4e-06, 2.14e-06, 9.77e-06, 1.218e-05, 2.767e-05])
        self.res_2 = np.array([2402465.0, 2403519.0, 2403638.0, 2402803.0, 2402483.0, 2403075.0, 2402438.0, 2403046.0, 2402666.0, 2402969.0, 2402522.0, 2403180.0, 2402591.0, 2402758.0, 2403014.0, 2403068.0, 2402934.0, 2402555.0, 2402928.0, 2402492.0, 2402747.0, 2402853.0, 2402693.0, 2402784.0])
        self.δres_2 = np.array([72.0, 59.0, 138.0, 80.0, 58.0, 103.0, 84.0, 63.0, 118.0, 91.0, 88.0, 110.0, 115.0, 111.0, 86.0, 100.0, 76.0, 84.0, 106.0, 105.0, 94.0, 107.0, 93.0, 70.0])
        self.res_4 = np.array([3693086.0, 3694211.0, 3695506.0, 3697549.0, 3698045.0, 3698729.0, 3699900.0, 3700067.0, 3700831.0, 3701458.0, 3701211.0, 3702202.0, 3701963.0, 3702430.0, 3703267.0, 3703982.0, 3703617.0, 3703494.0, 3703809.0, 3703724.0, 3704198.0, 3704598.0, 3704159.0, 3704378.0])
        self.δres_4 = np.array([302.0, 284.0, 385.0, 315.0, 274.0, 224.0, 99.0, 118.0, 104.0, 278.0, 156.0, 68.0, 48.0, 59.0, 50.0, 55.0, 70.0, 53.0, 66.0, 42.0, 38.0, 41.0, 24.0, 56.0])
        self.res_6 = np.array([3328099.0, 3329200.0, 3331130.0, 3331773.0, 3332437.0])
        self.δres_6 = np.array([47.0, 3.0, 40.0, 28.0, 28.0])
        self.p32_mean_4 = np.array([0.0448, 0.0357, 0.03, 0.031, 0.0238, 0.0176, 0.0195, 0.0224, 0.0162, 0.0124, 0.0119, 0.0119, 0.0124, 0.0105, 0.011, 0.009, 0.01, 0.0071, 0.0057, 0.0052, 0.0043, 0.0029, 0.0048, 0.0076])
        self.δp32_mean_4 = np.array([0.007, 0.0074, 0.0053, 0.0043, 0.0043, 0.0034, 0.0036, 0.0041, 0.0042, 0.0028, 0.0031, 0.002, 0.0023, 0.0026, 0.0028, 0.0025, 0.0022, 0.0025, 0.0017, 0.0016, 0.0011, 0.0012, 0.002, 0.0021])
        self.p12_mean_4 = np.array([0.0181, 0.0152, 0.0129, 0.0186, 0.011, 0.0138, 0.0076, 0.0105, 0.0043, 0.0071, 0.0057, 0.0057, 0.0048, 0.0048, 0.0038, 0.0067, 0.0043, 0.0067, 0.0048, 0.0029, 0.0038, 0.0033, 0.0057, 0.0029])
        self.δp12_mean_4 = np.array([0.0037, 0.0033, 0.0029, 0.0039, 0.0027, 0.003, 0.0019, 0.0018, 0.0014, 0.0018, 0.0014, 0.0019, 0.0013, 0.0013, 0.0014, 0.0019, 0.0013, 0.0016, 0.0016, 0.0012, 0.0018, 0.001, 0.0019, 0.0012])
        self.m32_mean_4 = np.array([0.001, 0.001, 0.0024, 0.0014, 0.001, 0.0005, 0.0014, 0.0014, 0.001, 0.001, 0.0014, 0.0014, 0.0019, 0.0014, 0.0019, 0.0005, 0.0, 0.0005, 0.0014, 0.001, 0.0019, 0.001, 0.001, 0.0])
        self.δm32_mean_4 = np.array([0.0006, 0.0006, 0.0009, 0.0008, 0.0006, 0.0005, 0.001, 0.0008, 0.0006, 0.0006, 0.0008, 0.0008, 0.0009, 0.0008, 0.0013, 0.0005, 0.0, 0.0005, 0.0008, 0.0006, 0.0009, 0.0006, 0.0009, 0.0])
        self.m52_mean_4 = np.array([0.0038, 0.0029, 0.0062, 0.0019, 0.0019, 0.001, 0.0029, 0.0014, 0.0024, 0.0033, 0.0019, 0.0014, 0.0019, 0.001, 0.0019, 0.0014, 0.0033, 0.0019, 0.001, 0.0033, 0.0014, 0.0014, 0.0024, 0.0014])
        self.δm52_mean_4 = np.array([0.0013, 0.0012, 0.0018, 0.0009, 0.0009, 0.0006, 0.0012, 0.0008, 0.0013, 0.0014, 0.0009, 0.0008, 0.0011, 0.0006, 0.0009, 0.001, 0.0012, 0.0009, 0.0006, 0.001, 0.0008, 0.0008, 0.0011, 0.0008])
        self.p32_max_4 = np.array([0.11, 0.14, 0.08, 0.07, 0.06, 0.05, 0.06, 0.09, 0.07, 0.05, 0.06, 0.03, 0.03, 0.05, 0.05, 0.04, 0.03, 0.05, 0.03, 0.02, 0.01, 0.02, 0.04, 0.04])
        self.δp32_max_4 = np.array([0.0313, 0.0347, 0.0271, 0.0255, 0.0237, 0.0218, 0.0237, 0.0286, 0.0255, 0.0218, 0.0237, 0.0171, 0.0171, 0.0218, 0.0218, 0.0196, 0.0171, 0.0218, 0.0171, 0.014, 0.0099, 0.014, 0.0196, 0.0196])
        self.p12_max_4 = np.array([0.05, 0.06, 0.05, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02, 0.03, 0.02, 0.03, 0.02, 0.0202, 0.02, 0.03, 0.02, 0.0202, 0.02, 0.02, 0.03, 0.01, 0.03, 0.02])
        self.δp12_max_4 = np.array([0.0218, 0.0237, 0.0218, 0.0237, 0.0196, 0.0196, 0.0171, 0.0171, 0.014, 0.0171, 0.014, 0.0171, 0.014, 0.0141, 0.014, 0.0171, 0.014, 0.0141, 0.014, 0.014, 0.0171, 0.0099, 0.0171, 0.014])
        self.m32_max_4 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.0101, 0.02, 0.0101, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0101, 0.02, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.0])
        self.δm32_max_4 = np.array([0.0099, 0.0099, 0.0099, 0.0099, 0.0099, 0.01, 0.014, 0.01, 0.0099, 0.0099, 0.0099, 0.0099, 0.0099, 0.01, 0.014, 0.0099, 0.0, 0.0099, 0.0099, 0.0099, 0.0099, 0.0099, 0.014, 0.0])
        self.m52_max_4 = np.array([0.02, 0.02, 0.03, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.0101, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01])
        self.δm52_max_4 = np.array([0.014, 0.014, 0.0171, 0.0099, 0.0099, 0.0099, 0.014, 0.0099, 0.014, 0.014, 0.0099, 0.0099, 0.014, 0.0099, 0.0099, 0.014, 0.014, 0.01, 0.0099, 0.0099, 0.0099, 0.0099, 0.014, 0.0099])
        self.p32_mean_6 = np.array([0.041, 0.0467, 0.0252, 0.0362, 0.0324])
        self.δp32_mean_6 = np.array([0.0073, 0.0085, 0.0048, 0.0061, 0.0046])
        self.p12_mean_6 = np.array([0.011, 0.0133, 0.0086, 0.0086, 0.0067])
        self.δp12_mean_6 = np.array([0.0018, 0.0022, 0.0021, 0.0014, 0.0018])
        self.m12_mean_6 = np.array([0.0295, 0.0219, 0.0152, 0.0119, 0.0124])
        self.δm12_mean_6 = np.array([0.0047, 0.0035, 0.0032, 0.0021, 0.0027])
        self.m52_mean_6 = np.array([0.0005, 0.0014, 0.0014, 0.0014, 0.001])
        self.δm52_mean_6 = np.array([0.0005, 0.0008, 0.0008, 0.0008, 0.0006])
        self.fid_4 = np.array([0.941, 0.9075, 0.9305, 0.9315, 0.9375, 0.9596, 0.9459, 0.9571, 0.9529, 0.9242, 0.939, 0.9008, 0.9163, 0.8852, 0.9022, 0.8819, 0.8847, 0.9005, 0.8864, 0.8739, 0.8355, 0.8029, 0.8333, 0.7962])
        self.δfid_4 = np.array([0.0124, 0.0148, 0.0127, 0.0136, 0.0094, 0.0081, 0.0141, 0.0107, 0.0093, 0.0103, 0.0154, 0.0122, 0.0128, 0.0216, 0.018, 0.0119, 0.0135, 0.0117, 0.0165, 0.0145, 0.0262, 0.0287, 0.0115, 0.0239])
        self.d52_splittings = np.array([2.51487242, 2.40204469, 2.54989117, 2.4705837 , 2.5844747 ,2.53840501, 2.61883663, 2.60509844, 2.65297022, 2.67113929])
        self.δd52_splittings = np.array([3.81506537e-05, 6.68857779e-05, 3.55727141e-05, 7.62923892e-05, 6.38425333e-05, 5.22479098e-05, 5.82881747e-05, 7.64665084e-05, 5.10474521e-05, 5.32115225e-05])
        self.rpi_splittings = np.array([100, 195, 100, 195, 100, 195, 100, 195, 100, 195])
        
        return
    
    # function to calculate state probabilities from d = 6 state vector
    def get_populations(self):
        
        self.p52 = np.array(())
        self.p32 = np.array(())
        self.p12 = np.array(())
        self.m12 = np.array(())
        self.m32 = np.array(())
        self.m52 = np.array(())
        
        for ψ in self.ψ:
            self.p52 = np.append(self.p52, np.abs(ψ[0][0][0])**2)
            self.p32 = np.append(self.p32, np.abs(ψ[1][0][0])**2)
            self.p12 = np.append(self.p12, np.abs(ψ[2][0][0])**2)
            self.m12 = np.append(self.m12, np.abs(ψ[3][0][0])**2)
            self.m32 = np.append(self.m32, np.abs(ψ[4][0][0])**2)
            self.m52 = np.append(self.m52, np.abs(ψ[5][0][0])**2)
            
        return
    
    def get_rabi_frequencies(self):
        # calculate single beam Rabi frequency from power, waists, and polarization vectors
        self.r_sm = 1 - self.r_sp - self.r_pi   # calculat ratio of power in sigma- 
        self.b_sm = 1 - self.b_sp - self.b_pi
        self.ls_sm = 1 - self.ls_sp - self.ls_pi
        
        me_dict = {'40Ca':3.283, '88Sr':4.187, '138Ba':4.103}
        
        def get_intensity(P, w):
            e = 1.6e-19   # fundamental unit charge
            ħ = 1.055e-34   # reduced plank constant
            c = 3e8   # speed of light
            a0 = 5.292e-11   # bohr radius
            ϵ0 = 8.85e-12   # permitivity of free space
            μ = me_dict[self.specie]
            # μ = 3.283   # P3/2 matrix element
            
            # constant to be multiplied by Clebsch-Gordan coefficient to get single beam rabi frequency
            ci = 2*(P**.5)*e*a0*μ/(w*ħ*((ϵ0*c*np.pi)**.5))
            return ci
        
        cr_sm = get_intensity(self.Pr*self.r_sm, self.w0r)
        cr_pi = get_intensity(self.Pr*self.r_pi, self.w0r)
        cr_sp = get_intensity(self.Pr*self.r_sp, self.w0r)
        
        cb_sm = get_intensity(self.Pb*self.b_sm, self.w0b)
        cb_pi = get_intensity(self.Pb*self.b_pi, self.w0b)
        cb_sp = get_intensity(self.Pb*self.b_sp, self.w0b)
        
        cls_sm = get_intensity(self.Pls*self.ls_sm, self.w0ls)
        cls_pi = get_intensity(self.Pls*self.ls_pi, self.w0ls)
        cls_sp = get_intensity(self.Pls*self.ls_sp, self.w0ls)
               
        # functions imported from PhotonScatteringFuncs.py to calculate CGs
        mu_D52 = mu(3/2,5/2,1,2)
        Ca40_mat_dict_D52 = construct_mat_dict_fine_structure(3/2,5/2,1,2,mu_D52);
        
        # calculating CG coefficients between D and P m states
        Ω06CG = Ca40_mat_dict_D52['5/2->3/2']   
        Ω16CG = Ca40_mat_dict_D52['3/2->3/2']
        Ω26CG = Ca40_mat_dict_D52['1/2->3/2']
        Ω17CG = Ca40_mat_dict_D52['3/2->1/2']
        Ω27CG = Ca40_mat_dict_D52['1/2->1/2']
        Ω37CG = Ca40_mat_dict_D52['-1/2->1/2']
        Ω28CG = Ca40_mat_dict_D52['1/2->-1/2']
        Ω38CG = Ca40_mat_dict_D52['-1/2->-1/2']
        Ω48CG = Ca40_mat_dict_D52['-3/2->-1/2']
        Ω39CG = Ca40_mat_dict_D52['-1/2->-3/2']
        Ω49CG = Ca40_mat_dict_D52['-3/2->-3/2']
        Ω59CG = Ca40_mat_dict_D52['-5/2->-3/2'] 
        
        # single beam rabi frequencies
        
        # red beam (rsig)
        self.Ω06r = cr_sm*Ω06CG
        self.Ω16r = cr_pi*Ω16CG
        self.Ω26r = cr_sp*Ω26CG
        self.Ω17r = cr_sm*Ω17CG
        self.Ω27r = cr_pi*Ω27CG
        self.Ω37r = cr_sp*Ω37CG 
        self.Ω28r = cr_sm*Ω28CG
        self.Ω38r = cr_pi*Ω38CG
        self.Ω48r = cr_sp*Ω48CG
        self.Ω39r = cr_sm*Ω39CG
        self.Ω49r = cr_pi*Ω49CG
        self.Ω59r = cr_sp*Ω59CG 
        
        # blue beam (rpi)
        self.Ω06b = cb_sm*Ω06CG
        self.Ω16b = cb_pi*Ω16CG
        self.Ω26b = cb_sp*Ω26CG
        self.Ω17b = cb_sm*Ω17CG
        self.Ω27b = cb_pi*Ω27CG
        self.Ω37b = cb_sp*Ω37CG 
        self.Ω28b = cb_sm*Ω28CG
        self.Ω38b = cb_pi*Ω38CG
        self.Ω48b = cb_sp*Ω48CG
        self.Ω39b = cb_sm*Ω39CG
        self.Ω49b = cb_pi*Ω49CG
        self.Ω59b = cb_sp*Ω59CG
        
        # lightshift beam
        self.Ω06ls = cls_sm*Ω06CG
        self.Ω16ls = cls_pi*Ω16CG
        self.Ω26ls = cls_sp*Ω26CG
        self.Ω17ls = cls_sm*Ω17CG
        self.Ω27ls = cls_pi*Ω27CG
        self.Ω37ls = cls_sp*Ω37CG 
        self.Ω28ls = cls_sm*Ω28CG
        self.Ω38ls = cls_pi*Ω38CG
        self.Ω48ls = cls_sp*Ω48CG
        self.Ω39ls = cls_sm*Ω39CG
        self.Ω49ls = cls_pi*Ω49CG
        self.Ω59ls = cls_sp*Ω59CG
        
        # counter rotating single beam rabi frequencies
        # polarizations are flipped as emission -> absorption and vice-versa
        self.Ω06rc = cr_sp*Ω06CG
        self.Ω16rc = cr_pi*Ω16CG
        self.Ω26rc = cr_sm*Ω26CG
        self.Ω17rc = cr_sp*Ω17CG
        self.Ω27rc = cr_pi*Ω27CG
        self.Ω37rc = cr_sm*Ω37CG 
        self.Ω28rc = cr_sp*Ω28CG
        self.Ω38rc = cr_pi*Ω38CG
        self.Ω48rc = cr_sm*Ω48CG
        self.Ω39rc = cr_sp*Ω39CG
        self.Ω49rc = cr_pi*Ω49CG
        self.Ω59rc = cr_sm*Ω59CG 
        
        self.Ω06bc = cb_sp*Ω06CG
        self.Ω16bc = cb_pi*Ω16CG
        self.Ω26bc = cb_sm*Ω26CG
        self.Ω17bc = cb_sp*Ω17CG
        self.Ω27bc = cb_pi*Ω27CG
        self.Ω37bc = cb_sm*Ω37CG 
        self.Ω28bc = cb_sp*Ω28CG
        self.Ω38bc = cb_pi*Ω38CG
        self.Ω48bc = cb_sm*Ω48CG
        self.Ω39bc = cb_sp*Ω39CG
        self.Ω49bc = cb_pi*Ω49CG
        self.Ω59bc = cb_sm*Ω59CG
        
        # 3d5/2 <-> F7/2 Clebsch Gordan coefficients
        mu_F = mu(7/2,5/2,3,2)
        Ca40_mat_dict_F = construct_mat_dict_fine_structure(7/2,5/2,3,2,mu_F)
        
        Ω010CG = Ca40_mat_dict_F['5/2->7/2']
        Ω011CG = Ca40_mat_dict_F['5/2->5/2']
        Ω012CG = Ca40_mat_dict_F['5/2->3/2']
        Ω111CG = Ca40_mat_dict_F['3/2->5/2']
        Ω112CG = Ca40_mat_dict_F['3/2->3/2']
        Ω113CG = Ca40_mat_dict_F['3/2->1/2']
        Ω212CG = Ca40_mat_dict_F['1/2->3/2']
        Ω213CG = Ca40_mat_dict_F['1/2->1/2']
        Ω214CG = Ca40_mat_dict_F['1/2->-1/2']
        Ω313CG = Ca40_mat_dict_F['-1/2->1/2']
        Ω314CG = Ca40_mat_dict_F['-1/2->-1/2']
        Ω315CG = Ca40_mat_dict_F['-1/2->-3/2']
        Ω414CG = Ca40_mat_dict_F['-3/2->-1/2']
        Ω415CG = Ca40_mat_dict_F['-3/2->-3/2']
        Ω416CG = Ca40_mat_dict_F['-3/2->-5/2']
        Ω515CG = Ca40_mat_dict_F['-5/2->-3/2']
        Ω516CG = Ca40_mat_dict_F['-5/2->-5/2']
        Ω517CG = Ca40_mat_dict_F['-5/2->-7/2']
        
        μF = 2.309   # matrix element to F manifolds
        μP = 3.283   # matrix element to P manifolds
        
        if self.F_states == False:
            μF = 0
        
        # Rsig beam F state single beam rabi frequencies
        self.Ω010r = cr_sp*Ω010CG*μF/μP   # divided by P3/2 matrix element, need relative strength
        self.Ω011r = cr_pi*Ω011CG*μF/μP
        self.Ω012r = cr_sm*Ω012CG*μF/μP
        self.Ω111r = cr_sp*Ω111CG*μF/μP
        self.Ω112r = cr_pi*Ω112CG*μF/μP
        self.Ω113r = cr_sm*Ω113CG*μF/μP
        self.Ω212r = cr_sp*Ω212CG*μF/μP
        self.Ω213r = cr_pi*Ω213CG*μF/μP
        self.Ω214r = cr_sm*Ω214CG*μF/μP
        self.Ω313r = cr_sp*Ω313CG*μF/μP
        self.Ω314r = cr_pi*Ω314CG*μF/μP
        self.Ω315r = cr_sm*Ω315CG*μF/μP
        self.Ω414r = cr_sp*Ω414CG*μF/μP
        self.Ω415r = cr_pi*Ω415CG*μF/μP
        self.Ω416r = cr_sm*Ω416CG*μF/μP
        self.Ω515r = cr_sp*Ω515CG*μF/μP
        self.Ω516r = cr_pi*Ω516CG*μF/μP
        self.Ω517r = cr_sm*Ω517CG*μF/μP
        
        # Rpi beam F state single beam rabi frequencies
        self.Ω010b = cb_sp*Ω010CG*μF/μP
        self.Ω011b = cb_pi*Ω011CG*μF/μP
        self.Ω012b = cb_sm*Ω012CG*μF/μP
        self.Ω111b = cb_sp*Ω111CG*μF/μP
        self.Ω112b = cb_pi*Ω112CG*μF/μP
        self.Ω113b = cb_sm*Ω113CG*μF/μP
        self.Ω212b = cb_sp*Ω212CG*μF/μP
        self.Ω213b = cb_pi*Ω213CG*μF/μP
        self.Ω214b = cb_sm*Ω214CG*μF/μP
        self.Ω313b = cb_sp*Ω313CG*μF/μP
        self.Ω314b = cb_pi*Ω314CG*μF/μP
        self.Ω315b = cb_sm*Ω315CG*μF/μP
        self.Ω414b = cb_sp*Ω414CG*μF/μP
        self.Ω415b = cb_pi*Ω415CG*μF/μP
        self.Ω416b = cb_sm*Ω416CG*μF/μP
        self.Ω515b = cb_sp*Ω515CG*μF/μP
        self.Ω516b = cb_pi*Ω516CG*μF/μP
        self.Ω517b = cb_sm*Ω517CG*μF/μP 
        
        # create sympy variables of single beam rabi frequencies
        Ω06r, Ω16r, Ω17r, Ω26r, Ω27r, Ω28r, Ω37r, Ω38r, Ω39r, Ω48r, Ω49r, Ω59r = sp.symbols('Ω06r Ω16r Ω17r Ω26r Ω27r Ω28r Ω37r Ω38r Ω39r Ω48r Ω49r Ω59r')
        Ω06rc, Ω16rc, Ω17rc, Ω26rc, Ω27rc, Ω28rc, Ω37rc, Ω38rc, Ω39rc, Ω48rc, Ω49rc, Ω59rc = sp.symbols('Ω06rc Ω16rc Ω17rc Ω26rc Ω27rc Ω28rc Ω37rc Ω38rc Ω39rc Ω48rc Ω49rc Ω59rc')
        Ω012r, Ω112r, Ω212r, Ω113r, Ω213r, Ω313r, Ω214r, Ω314r, Ω414r, Ω315r, Ω415r, Ω515r = sp.symbols('Ω012r Ω112r Ω212r Ω113r Ω213r Ω313r Ω214r Ω314r Ω414r Ω315r Ω415r Ω515r')
        Ω06b, Ω16b, Ω17b, Ω26b, Ω27b, Ω28b, Ω37b, Ω38b, Ω39b, Ω48b, Ω49b, Ω59b = sp.symbols('Ω06b Ω16b Ω17b Ω26b Ω27b Ω28b Ω37b Ω38b Ω39b Ω48b Ω49b Ω59b')
        Ω06bc, Ω16bc, Ω17bc, Ω26bc, Ω27bc, Ω28bc, Ω37bc, Ω38bc, Ω39bc, Ω48bc, Ω49bc, Ω59bc = sp.symbols('Ω06bc Ω16bc Ω17bc Ω26bc Ω27bc Ω28bc Ω37bc Ω38bc Ω39bc Ω48bc Ω49bc Ω59bc')
        Ω012b, Ω112b, Ω212b, Ω113b, Ω213b, Ω313b, Ω214b, Ω314b, Ω414b, Ω315b, Ω415b, Ω515b = sp.symbols('Ω012b Ω112b Ω212b Ω113b Ω213b Ω313b Ω214b Ω314b Ω414b Ω315b Ω415b Ω515b')
        Δ = sp.symbols('Δ')
        Δc = sp.symbols('Δc')
        Δf = sp.symbols('Δf')
        ωr = sp.symbols('ωr')
        ω0 = sp.symbols('ω0')
        ω01, ω12, ω23, ω34, ω45 = sp.symbols('ω01 ω12 ω23 ω34 ω45')

        # list of single beam rabi frequencies as sympy variables
        self.symbols = (Ω06r, Ω16r, Ω26r, Ω17r, Ω27r, Ω37r, Ω28r, Ω38r, Ω48r, Ω39r, Ω49r, Ω59r, 
                        Ω06b, Ω16b, Ω26b, Ω17b, Ω27b, Ω37b, Ω28b, Ω38b, Ω48b, Ω39b, Ω49b, Ω59b, 
                        Ω06rc, Ω16rc, Ω26rc, Ω17rc, Ω27rc, Ω37rc, Ω28rc, Ω38rc, Ω48rc, Ω39rc, Ω49rc, Ω59rc, 
                        Ω06bc, Ω16bc, Ω26bc, Ω17bc, Ω27bc, Ω37bc, Ω28bc, Ω38bc, Ω48bc, Ω39bc, Ω49bc, Ω59bc, 
                        Ω012r, Ω112r, Ω212r, Ω113r, Ω213r, Ω313r, Ω214r, Ω314r, Ω414r, Ω315r, Ω415r, Ω515r, 
                        Ω012b, Ω112b, Ω212b, Ω113b, Ω213b, Ω313b, Ω214b, Ω314b, Ω414b, Ω315b, Ω415b, Ω515b, 
                        Δ, Δc, Δf,ωr, ω0, ω01, ω12, ω23, ω34, ω45)
        
        # list of the values of each single beam Rabi frequencies
        # will later map the values in nums to the sympy variables in symbols to convert analytic expressions to actual numbers
        self.nums= (self.Ω06r, self.Ω16r, self.Ω26r, self.Ω17r, self.Ω27r, self.Ω37r, self.Ω28r, self.Ω38r,  self.Ω48r, self.Ω39r, self.Ω49r, self.Ω59r, 
                    self.Ω06b, self.Ω16b, self.Ω26b, self.Ω17b, self.Ω27b, self.Ω37b, self.Ω28b, self.Ω38b, self.Ω48b, self.Ω39b, self.Ω49b, self.Ω59b, 
                    self.Ω06rc, self.Ω16rc, self.Ω26rc, self.Ω17rc, self.Ω27rc, self.Ω37rc, self.Ω28rc, self.Ω38rc,  self.Ω48rc, self.Ω39rc, self.Ω49rc, self.Ω59rc, 
                    self.Ω06bc, self.Ω16bc, self.Ω26bc, self.Ω17bc, self.Ω27bc, self.Ω37bc, self.Ω28bc, self.Ω38bc, self.Ω48bc, self.Ω39bc, self.Ω49bc, self.Ω59bc, 
                    self.Ω012r, self.Ω112r, self.Ω212r, self.Ω113r, self.Ω213r, self.Ω313r, self.Ω214r, self.Ω314r,  self.Ω414r, self.Ω315r, self.Ω415r, self.Ω515r, 
                    self.Ω012b, self.Ω112b, self.Ω212b, self.Ω113b, self.Ω213b, self.Ω313b, self.Ω214b, self.Ω314b,  self.Ω414b, self.Ω315b, self.Ω415b, self.Ω515b,
                    self.Δ, self.Δc, self.ΔF, self.ω, self.ω0, self.ω01, self.ω12, self.ω23, self.ω34, self.ω45)          
        
        # numbers to put in for lightshift beam stark shift calculations
        self.nums_ls= (self.Ω06ls, self.Ω16ls, self.Ω26ls, self.Ω17ls, self.Ω27ls, self.Ω37ls, self.Ω28ls, self.Ω38ls,  self.Ω48ls, self.Ω39ls, self.Ω49ls, self.Ω59ls, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    self.Δls, self.Δc, self.ΔF, self.ω, self.ω0, self.ω01, self.ω12, self.ω23, self.ω34, self.ω45)          
        return
    
    def simulate(self, t, s0, progress_bar=None):
        # simulation of 6 level system in D5/2
        # P states have been adiabatically eliminated, each coupling between D states is a two-photon coupling
        
        self.ψ = basis(6,s0)   # intialize state vector
        
        self.get_rabi_frequencies()   # refresh values of single beam rabi frequencies, incase power has been updated
        
        # basic terms
        Ω06r = self.Ω06r
        Ω16r = self.Ω16r
        Ω26r = self.Ω26r
        Ω17r = self.Ω17r
        Ω27r = self.Ω27r
        Ω37r = self.Ω37r
        Ω28r = self.Ω28r
        Ω38r = self.Ω38r
        Ω48r = self.Ω48r 
        Ω39r = self.Ω39r
        Ω49r = self.Ω49r
        Ω59r = self.Ω59r 
        Ω06b = self.Ω06b
        Ω16b = self.Ω16b
        Ω26b = self.Ω26b
        Ω17b = self.Ω17b
        Ω27b = self.Ω27b
        Ω37b = self.Ω37b
        Ω28b = self.Ω28b
        Ω38b = self.Ω38b
        Ω48b = self.Ω48b 
        Ω39b = self.Ω39b
        Ω49b = self.Ω49b
        Ω59b = self.Ω59b 
        Ω06ls = self.Ω06ls
        Ω16ls = self.Ω16ls
        Ω26ls = self.Ω26ls
        Ω17ls = self.Ω17ls
        Ω27ls = self.Ω27ls
        Ω37ls = self.Ω37ls
        Ω28ls = self.Ω28ls
        Ω38ls = self.Ω38ls
        Ω48ls = self.Ω48ls 
        Ω39ls = self.Ω39ls
        Ω49ls = self.Ω49ls
        Ω59ls = self.Ω59ls
        
        
        # counter rotating terms
        Ω06rc = self.Ω06rc
        Ω16rc = self.Ω16rc
        Ω26rc = self.Ω26rc
        Ω17rc = self.Ω17rc
        Ω27rc = self.Ω27rc
        Ω37rc = self.Ω37rc
        Ω28rc = self.Ω28rc
        Ω38rc = self.Ω38rc
        Ω48rc = self.Ω48rc 
        Ω39rc = self.Ω39rc
        Ω49rc = self.Ω49rc
        Ω59rc = self.Ω59rc 
        Ω06bc = self.Ω06bc
        Ω16bc = self.Ω16bc
        Ω26bc = self.Ω26bc
        Ω17bc = self.Ω17bc
        Ω27bc = self.Ω27bc
        Ω37bc = self.Ω37bc
        Ω28bc = self.Ω28bc
        Ω38bc = self.Ω38bc
        Ω48bc = self.Ω48bc 
        Ω39bc = self.Ω39bc
        Ω49bc = self.Ω49bc
        Ω59bc = self.Ω59bc 
        
        # F states
        Ω010r = self.Ω010r
        Ω011r = self.Ω011r
        Ω012r = self.Ω012r
        Ω111r = self.Ω111r 
        Ω112r = self.Ω112r 
        Ω113r = self.Ω113r
        Ω212r = self.Ω212r 
        Ω213r = self.Ω213r
        Ω214r = self.Ω214r 
        Ω313r = self.Ω313r
        Ω314r = self.Ω314r
        Ω315r = self.Ω315r
        Ω414r = self.Ω414r
        Ω415r = self.Ω415r
        Ω416r = self.Ω416r 
        Ω515r = self.Ω515r 
        Ω516r = self.Ω516r
        Ω517r = self.Ω517r
        
        Ω010b = self.Ω010b
        Ω011b = self.Ω011b
        Ω012b = self.Ω012b
        Ω111b = self.Ω111b 
        Ω112b = self.Ω112b 
        Ω113b = self.Ω113b
        Ω212b = self.Ω212b 
        Ω213b = self.Ω213b
        Ω214b = self.Ω214b 
        Ω313b = self.Ω313b
        Ω314b = self.Ω314b
        Ω315b = self.Ω315b
        Ω414b = self.Ω414b
        Ω415b = self.Ω415b
        Ω416b = self.Ω416b 
        Ω515b = self.Ω515b 
        Ω516b = self.Ω516b
        Ω517b = self.Ω517b

        ϕ = self.ϕ   # raman beam relative phase
        
        # basic terms
        def H00_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b**2 + Ω06b*Ω06r*np.exp(-1j*ω*t + 1j*ϕ)+ Ω06b*Ω06r*np.exp(+1j*ω*t - 1j*ϕ) + Ω06r**2)/(-4*self.Δ) + (Ω06ls**2)/(4*self.Δls)
        def H01_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*Ω16b*np.exp(-1j*ω0*t) + Ω06r*Ω16b*np.exp(+1j*(ω-ω0)*t -1j*ϕ) + Ω06b*Ω16r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω06r*Ω16r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H01d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*Ω16b*np.exp(+1j*ω0*t) + Ω06r*Ω16b*np.exp(-1j*(ω-ω0)*t +1j*ϕ) + Ω06b*Ω16r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω06r*Ω16r*np.exp(+1j*(ω0)*t) )/(-4*self.Δ)
        def H02_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*Ω26b*np.exp(-2j*ω0*t) + Ω06r*Ω26b*np.exp(+1j*(ω-2*ω0)*t -1j*ϕ) + Ω06b*Ω26r*np.exp(-1j*(ω+2*ω0)*t +1j*ϕ) + Ω06r*Ω26r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H02d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*Ω26b*np.exp(+2j*ω0*t) + Ω06r*Ω26b*np.exp(-1j*(ω-2*ω0)*t +1j*ϕ) + Ω06b*Ω26r*np.exp(+1j*(ω+2*ω0)*t -1j*ϕ) + Ω06r*Ω26r*np.exp(+2j*ω0*t))/(-4*self.Δ)
        def H11_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16b**2 + Ω16r**2 + Ω17b**2 + Ω17b*Ω17r*np.exp(-1j*ω*t +1j*ϕ)+ Ω17b*Ω17r*np.exp(+1j*ω*t -1j*ϕ) + Ω17r**2 + Ω16b*Ω16r*np.exp(-1j*ω*t +1j*ϕ)+ Ω16b*Ω16r*np.exp(+1j*ω*t -1j*ϕ))/(-4*self.Δ)   + (Ω16ls**2 + Ω17ls**2)/(4*self.Δls)
        def H12_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16b*Ω26b*np.exp(-1j*ω0*t) + Ω16r*Ω26b*np.exp(+1j*(ω-ω0)*t-1j*ϕ)+ Ω16b*Ω26r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω16r*Ω26r*np.exp(-1j*(ω0)*t) + Ω27b*Ω17b*np.exp(-1j*ω0*t) + Ω17r*Ω27b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω17b*Ω27r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω17r*Ω27r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H12d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16b*Ω26b*np.exp(+1j*ω0*t) + Ω16r*Ω26b*np.exp(-1j*(ω-ω0)*t+1j*ϕ)+ Ω16b*Ω26r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω16r*Ω26r*np.exp(+1j*(ω0)*t) + Ω27b*Ω17b*np.exp(+1j*ω0*t) + Ω17r*Ω27b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω17b*Ω27r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω17r*Ω27r*np.exp(+1j*(ω0)*t))/(-4*self.Δ)
        def H13_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17b*Ω37b*np.exp(-2j*ω0*t) +Ω17r*Ω37b*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) +Ω17b*Ω37r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ) +Ω17r*Ω37r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H13d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17b*Ω37b*np.exp(+2j*ω0*t) +Ω17r*Ω37b*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) +Ω17b*Ω37r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ) +Ω17r*Ω37r*np.exp(+2j*ω0*t) )/(-4*self.Δ)
        # 854 lightshift is included in H22(t). Also included in H33, H44, H55, with increasing magnitude as governed by CG coefficients
        def H22_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω26b**2 + Ω26b*Ω26r*np.exp(-1j*ω*t+1j*ϕ) + Ω26b*Ω26r*np.exp(1j*ω*t-1j*ϕ) +Ω26r**2 +Ω27b**2 + Ω27b*Ω27r*np.exp(-1j*ω*t+1j*ϕ) + Ω27b*Ω27r*np.exp(1j*ω*t-1j*ϕ) + Ω27r**2 +Ω28b**2 + Ω28b*Ω28r*np.exp(-1j*ω*t+1j*ϕ) + Ω28b*Ω28r*np.exp(1j*ω*t-1j*ϕ) +Ω28r**2)/(-4*self.Δ)  - self.ls_854    + (Ω26ls**2 + Ω27ls**2 + Ω28ls**2)/(4*self.Δls)
        def H23_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27b*Ω37b*np.exp(-1j*ω0*t) + Ω27r*Ω37b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω27b*Ω37r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω27r*Ω37r*np.exp(-1j*(ω0)*t) +Ω28b*Ω38b*np.exp(-1j*ω0*t) + Ω28r*Ω38b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω28b*Ω38r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω28r*Ω38r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H23d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27b*Ω37b*np.exp(+1j*ω0*t) + Ω27r*Ω37b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω27b*Ω37r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω27r*Ω37r*np.exp(+1j*(ω0)*t) +Ω28b*Ω38b*np.exp(+1j*ω0*t) + Ω28r*Ω38b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω28b*Ω38r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω28r*Ω38r*np.exp(+1j*(ω0)*t))/(-4*self.Δ)
        def H24_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28b*Ω48b*np.exp(-2j*ω0*t) + Ω28r*Ω48b*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) + Ω28b*Ω48r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)+ Ω28r*Ω48r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H24d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28b*Ω48b*np.exp(+2j*ω0*t) + Ω28r*Ω48b*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) + Ω28b*Ω48r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)+ Ω28r*Ω48r*np.exp(+2j*ω0*t))/(-4*self.Δ)
        def H33_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω37b**2 +Ω37b*Ω37r*np.exp(-1j*ω*t+1j*ϕ) +Ω37b*Ω37r*np.exp(+1j*ω*t-1j*ϕ) +Ω37r**2 +Ω38b**2 +Ω38b*Ω38r*np.exp(+1j*ω*t-1j*ϕ) +Ω38b*Ω38r*np.exp(-1j*ω*t+1j*ϕ) +Ω38r**2 +Ω39b**2 +Ω39b*Ω39r*np.exp(-1j*ω*t+1j*ϕ) +Ω39b*Ω39r*np.exp(+1j*ω*t-1j*ϕ) +Ω39r**2)/(-4*self.Δ) - 3*self.ls_854    + (Ω37ls**2 + Ω38ls**2 + Ω39ls**2)/(4*self.Δls)
        def H34_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38b*Ω48b*np.exp(-1j*ω0*t) +Ω38r*Ω48b*np.exp(1j*(ω-ω0)*t-1j*ϕ) +Ω38b*Ω48r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω38r*Ω48r*np.exp(-1j*(ω0)*t) +Ω39b*Ω49b*np.exp(-1j*ω0*t) +Ω39r*Ω49b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω39b*Ω49r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω39r*Ω49r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H34d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38b*Ω48b*np.exp(+1j*ω0*t) +Ω38r*Ω48b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω38b*Ω48r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω38r*Ω48r*np.exp(+1j*(ω0)*t) +Ω39b*Ω49b*np.exp(+1j*ω0*t) +Ω39r*Ω49b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω39b*Ω49r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω39r*Ω49r*np.exp(+1j*(ω0)*t))/(-4*self.Δ)
        def H35_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39b*Ω59b*np.exp(-2j*ω0*t) +Ω39r*Ω59b*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) +Ω39b*Ω59r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)  +Ω39r*Ω59r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H35d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39b*Ω59b*np.exp(+2j*ω0*t) +Ω39r*Ω59b*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) +Ω39b*Ω59r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)  +Ω39r*Ω59r*np.exp(+2j*ω0*t))/(-4*self.Δ)
        def H44_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω48b**2 +Ω48b*Ω48r*np.exp(-1j*ω*t+1j*ϕ) +Ω48b*Ω48r*np.exp(+1j*ω*t-1j*ϕ)+Ω48r**2 +Ω49b**2 +Ω49b*Ω49r*np.exp(+1j*ω*t-1j*ϕ) +Ω49b*Ω49r*np.exp(-1j*ω*t+1j*ϕ) + Ω49r**2)/(-4*self.Δ) - 6*self.ls_854   + (Ω48ls**2 + Ω49ls**2)/(4*self.Δls)
        def H45_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49b*Ω59b*np.exp(-1j*ω0*t) + Ω49r*Ω59b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω49r*Ω59r*np.exp(-1j*(ω0)*t) + Ω49b*Ω59r*np.exp(-1j*(ω+ω0)*t+1j*ϕ))/(-4*self.Δ)
        def H45d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49b*Ω59b*np.exp(+1j*ω0*t) + Ω49r*Ω59b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω49r*Ω59r*np.exp(+1j*(ω0)*t) + Ω49b*Ω59r*np.exp(+1j*(ω+ω0)*t-1j*ϕ))/(-4*self.Δ)
        def H55_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω59b**2 + Ω59b*Ω59r*np.exp(-1j*ω*t+1j*ϕ) + Ω59b*Ω59r*np.exp(+1j*ω*t-1j*ϕ) +Ω59r**2)/(-4*self.Δ) - 10*self.ls_854  + (Ω06ls**2)/(4*self.Δls) + (Ω59ls**2)/(4*self.Δls)
        
        # F states
        def H00_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b**2 + Ω012b*Ω012r*np.exp(-1j*ω*t + 1j*ϕ)+ Ω012b*Ω012r*np.exp(+1j*ω*t - 1j*ϕ) + Ω012r**2)/(-4*self.ΔF)\
                   -(Ω010b**2 + Ω010r**2 + Ω010r*Ω010b*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)) + Ω011b**2 + Ω011r**2 + Ω011r*Ω011b*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)))/(-4*self.ΔF)
        def H01_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*Ω112b*np.exp(-1j*ω0*t) + Ω012r*Ω112b*np.exp(+1j*(ω-ω0)*t -1j*ϕ) + Ω012b*Ω112r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω012r*Ω112r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)\
                   -(Ω011b*Ω111b*np.exp(-1j*ω0*t) + Ω011r*Ω111b*np.exp(+1j*(ω-ω0)*t -1j*ϕ) + Ω011b*Ω111r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω011r*Ω111r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H01d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*Ω112b*np.exp(+1j*ω0*t) + Ω012r*Ω112b*np.exp(-1j*(ω-ω0)*t +1j*ϕ) + Ω012b*Ω112r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω012r*Ω112r*np.exp(+1j*(ω0)*t) )/(-4*self.ΔF)\
                   -(Ω011b*Ω111b*np.exp(+1j*ω0*t) + Ω011r*Ω111b*np.exp(-1j*(ω-ω0)*t +1j*ϕ) + Ω011b*Ω111r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω011r*Ω111r*np.exp(+1j*(ω0)*t) )/(-4*self.ΔF)
        def H02_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*Ω212b*np.exp(-2j*ω0*t) + Ω012r*Ω212b*np.exp(+1j*(ω-2*ω0)*t -1j*ϕ) + Ω012b*Ω212r*np.exp(-1j*(ω+2*ω0)*t +1j*ϕ) + Ω012r*Ω212r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H02d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*Ω212b*np.exp(+2j*ω0*t) + Ω012r*Ω212b*np.exp(-1j*(ω-2*ω0)*t +1j*ϕ) + Ω012b*Ω212r*np.exp(+1j*(ω+2*ω0)*t -1j*ϕ) + Ω012r*Ω212r*np.exp(+2j*ω0*t))/(-4*self.ΔF)
        def H11_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω112b**2 + Ω112r**2 + Ω113b**2 + Ω113b*Ω113r*np.exp(-1j*ω*t +1j*ϕ)+ Ω113b*Ω113r*np.exp(+1j*ω*t -1j*ϕ) + Ω113r**2 + Ω112b*Ω112r*np.exp(-1j*ω*t +1j*ϕ)+ Ω112b*Ω112r*np.exp(+1j*ω*t -1j*ϕ))/(-4*self.ΔF) \
                   -(Ω111b**2 + Ω111r**2)/(-4*self.ΔF) 
        def H12_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω112b*Ω212b*np.exp(-1j*ω0*t) + Ω112r*Ω212b*np.exp(+1j*(ω-ω0)*t-1j*ϕ)+ Ω112b*Ω212r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω112r*Ω212r*np.exp(-1j*(ω0)*t) + Ω213b*Ω113b*np.exp(-1j*ω0*t) + Ω113r*Ω213b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω113b*Ω213r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω113r*Ω213r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H12d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω112b*Ω212b*np.exp(+1j*ω0*t) + Ω112r*Ω212b*np.exp(-1j*(ω-ω0)*t+1j*ϕ)+ Ω112b*Ω212r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω112r*Ω212r*np.exp(+1j*(ω0)*t) + Ω213b*Ω113b*np.exp(+1j*ω0*t) + Ω113r*Ω213b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω113b*Ω213r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω113r*Ω213r*np.exp(+1j*(ω0)*t))/(-4*self.ΔF)
        def H13_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω113b*Ω313b*np.exp(-2j*ω0*t) +Ω113r*Ω313b*np.exp(+1j*(ω-2*ω0)*t+1j*ϕ) +Ω113b*Ω313r*np.exp(-1j*(ω+2*ω0)*t-1j*ϕ) +Ω113r*Ω313r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H13d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω113b*Ω313b*np.exp(+2j*ω0*t) +Ω113r*Ω313b*np.exp(-1j*(ω-2*ω0)*t-1j*ϕ) +Ω113b*Ω313r*np.exp(+1j*(ω+2*ω0)*t+1j*ϕ) +Ω113r*Ω313r*np.exp(+2j*ω0*t) )/(-4*self.ΔF)
        def H22_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω212b**2 + Ω212b*Ω212r*np.exp(-1j*ω*t+1j*ϕ) + Ω212b*Ω212r*np.exp(1j*ω*t-1j*ϕ) +Ω212r**2 +Ω213b**2 + Ω213b*Ω213r*np.exp(-1j*ω*t+1j*ϕ) + Ω213b*Ω213r*np.exp(1j*ω*t-1j*ϕ) + Ω213r**2 +Ω214b**2 + Ω214b*Ω214r*np.exp(-1j*ω*t+1j*ϕ) + Ω214b*Ω214r*np.exp(1j*ω*t-1j*ϕ) +Ω214r**2)/(-4*self.ΔF)
        def H23_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω213b*Ω313b*np.exp(-1j*ω0*t) + Ω213r*Ω313b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω213b*Ω313r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω213r*Ω313r*np.exp(-1j*(ω0)*t) +Ω214b*Ω314b*np.exp(-1j*ω0*t) + Ω214r*Ω314b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω214b*Ω314r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω214r*Ω314r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H23d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω213b*Ω313b*np.exp(+1j*ω0*t) + Ω213r*Ω313b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω213b*Ω313r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω213r*Ω313r*np.exp(+1j*(ω0)*t) +Ω214b*Ω314b*np.exp(+1j*ω0*t) + Ω214r*Ω314b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω214b*Ω314r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω214r*Ω314r*np.exp(+1j*(ω0)*t))/(-4*self.ΔF)
        def H24_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω214b*Ω414b*np.exp(-2j*ω0*t) + Ω214r*Ω414b*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) + Ω214b*Ω414r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)+ Ω214r*Ω414r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H24d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω214b*Ω414b*np.exp(+2j*ω0*t) + Ω214r*Ω414b*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) + Ω214b*Ω414r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)+ Ω214r*Ω414r*np.exp(+2j*ω0*t))/(-4*self.ΔF)
        def H33_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω313b**2 +Ω313b*Ω313r*np.exp(-1j*ω*t+1j*ϕ) +Ω313b*Ω313r*np.exp(+1j*ω*t-1j*ϕ) +Ω313r**2 +Ω314b**2 +Ω314b*Ω314r*np.exp(+1j*ω*t-1j*ϕ) +Ω314b*Ω314r*np.exp(-1j*ω*t+1j*ϕ) +Ω314r**2 +Ω315b**2 +Ω315b*Ω315r*np.exp(-1j*ω*t+1j*ϕ) +Ω315b*Ω315r*np.exp(+1j*ω*t-1j*ϕ) +Ω315r**2)/(-4*self.ΔF)
        def H34_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω314b*Ω414b*np.exp(-1j*ω0*t) +Ω314r*Ω414b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω314b*Ω414r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω314r*Ω414r*np.exp(-1j*(ω0)*t) +Ω315b*Ω415b*np.exp(-1j*ω0*t) +Ω315r*Ω415b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω315b*Ω415r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω315r*Ω415r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H34d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω314b*Ω414b*np.exp(+1j*ω0*t) +Ω314r*Ω414b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω314b*Ω414r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω314r*Ω414r*np.exp(+1j*(ω0)*t) +Ω315b*Ω415b*np.exp(+1j*ω0*t) +Ω315r*Ω415b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω315b*Ω415r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω315r*Ω415r*np.exp(+1j*(ω0)*t))/(-4*self.ΔF)
        def H35_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω315b*Ω515b*np.exp(-2j*ω0*t) +Ω315r*Ω515b*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) +Ω315b*Ω515r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)  +Ω315r*Ω515r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H35d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω315b*Ω515b*np.exp(+2j*ω0*t) +Ω315r*Ω515b*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) +Ω315b*Ω515r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)  +Ω315r*Ω515r*np.exp(+2j*ω0*t))/(-4*self.ΔF)
        def H44_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω414b**2 +Ω414b*Ω414r*np.exp(-1j*ω*t+1j*ϕ) +Ω414b*Ω414r*np.exp(+1j*ω*t-1j*ϕ)+Ω414r**2 +Ω415b**2 +Ω415b*Ω415r*np.exp(+1j*ω*t-1j*ϕ) +Ω415b*Ω415r*np.exp(-1j*ω*t+1j*ϕ) + Ω415r**2)/(-4*self.ΔF)\
                   -(Ω416b**2 + Ω416r**2)/(-4*self.ΔF) 
        def H45_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω415b*Ω515b*np.exp(-1j*ω0*t) + Ω415r*Ω515b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω415r*Ω515r*np.exp(-1j*(ω0)*t) + Ω415b*Ω515r*np.exp(-1j*(ω+ω0)*t+1j*ϕ))/(-4*self.ΔF)\
                   -(Ω416b*Ω516b*np.exp(-1j*ω0*t) + Ω416r*Ω516b*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω416r*Ω516r*np.exp(-1j*(ω0)*t) + Ω416b*Ω516r*np.exp(-1j*(ω+ω0)*t+1j*ϕ))/(-4*self.ΔF)
        def H45d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω415b*Ω515b*np.exp(+1j*ω0*t) + Ω415r*Ω515b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω415r*Ω515r*np.exp(+1j*(ω0)*t) + Ω415b*Ω515r*np.exp(+1j*(ω+ω0)*t-1j*ϕ))/(-4*self.ΔF)\
                   -(Ω416b*Ω516b*np.exp(+1j*ω0*t) + Ω416r*Ω516b*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω416r*Ω516r*np.exp(+1j*(ω0)*t) + Ω416b*Ω516r*np.exp(+1j*(ω+ω0)*t-1j*ϕ))/(-4*self.ΔF)
        def H55_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω515b**2 + Ω515b*Ω515r*np.exp(-1j*ω*t+1j*ϕ) + Ω515b*Ω515r*np.exp(+1j*ω*t-1j*ϕ) +Ω515r**2)/(-4*self.ΔF) \
                   -(Ω516b**2 + Ω516r**2 + Ω516r*Ω516b*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)) + Ω517b**2 + Ω517r**2 + Ω517r*Ω517b*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)))/(-4*self.ΔF)
        
        #  counter rotating terms
        ϕ = -ϕ   # relative phase switches signs for these terms    
        def H00_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc**2 + Ω06bc*Ω06rc*np.exp(1j*ω*t + 1j*ϕ)+ Ω06bc*Ω06rc*np.exp(-1j*ω*t - 1j*ϕ) + Ω06rc**2)/(-4*self.Δc)
        def H01_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*Ω16bc*np.exp(-1j*ω0*t) + Ω06rc*Ω16bc*np.exp(+1j*(-ω-ω0)*t -1j*ϕ) + Ω06bc*Ω16rc*np.exp(-1j*(-ω+ω0)*t +1j*ϕ) + Ω06rc*Ω16rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H01d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*Ω16bc*np.exp(+1j*ω0*t) + Ω06rc*Ω16bc*np.exp(-1j*(-ω-ω0)*t +1j*ϕ) + Ω06bc*Ω16rc*np.exp(+1j*(-ω+ω0)*t -1j*ϕ) + Ω06rc*Ω16rc*np.exp(+1j*(ω0)*t) )/(-4*self.Δc)
        def H02_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*Ω26bc*np.exp(-2j*ω0*t) + Ω06rc*Ω26bc*np.exp(+1j*(-ω-2*ω0)*t -1j*ϕ) + Ω06bc*Ω26rc*np.exp(-1j*(-ω+2*ω0)*t +1j*ϕ) + Ω06rc*Ω26rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H02d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*Ω26bc*np.exp(+2j*ω0*t) + Ω06rc*Ω26bc*np.exp(-1j*(-ω-2*ω0)*t +1j*ϕ) + Ω06bc*Ω26rc*np.exp(+1j*(-ω+2*ω0)*t -1j*ϕ) + Ω06rc*Ω26rc*np.exp(+2j*ω0*t))/(-4*self.Δc)
        def H11_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16bc**2 + Ω16rc**2 + Ω17bc**2 + Ω17bc*Ω17rc*np.exp(+1j*ω*t +1j*ϕ)+ Ω17bc*Ω17rc*np.exp(-1j*ω*t -1j*ϕ) + Ω17rc**2 + Ω16bc*Ω16rc*np.exp(1j*ω*t +1j*ϕ)+ Ω16bc*Ω16rc*np.exp(-1j*ω*t -1j*ϕ))/(-4*self.Δc)
        def H12_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16bc*Ω26bc*np.exp(-1j*ω0*t) + Ω16rc*Ω26bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ)+ Ω16bc*Ω26rc*np.exp(-1j*(-ω+ω0)*t +1j*ϕ) + Ω16rc*Ω26rc*np.exp(-1j*(ω0)*t) + Ω27bc*Ω17bc*np.exp(-1j*ω0*t) + Ω17rc*Ω27bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) + Ω17bc*Ω27rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) + Ω17rc*Ω27rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H12d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16bc*Ω26bc*np.exp(+1j*ω0*t) + Ω16rc*Ω26bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ)+ Ω16bc*Ω26rc*np.exp(+1j*(-ω+ω0)*t -1j*ϕ) + Ω16rc*Ω26rc*np.exp(+1j*(ω0)*t) + Ω27bc*Ω17bc*np.exp(+1j*ω0*t) + Ω17rc*Ω27bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) + Ω17bc*Ω27rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) + Ω17rc*Ω27rc*np.exp(+1j*(ω0)*t))/(-4*self.Δc)
        def H13_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17bc*Ω37bc*np.exp(-2j*ω0*t) +Ω17rc*Ω37bc*np.exp(+1j*(-ω-2*ω0)*t+1j*ϕ) +Ω17bc*Ω37rc*np.exp(-1j*(-ω+2*ω0)*t-1j*ϕ) +Ω17rc*Ω37rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H13d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17bc*Ω37bc*np.exp(+2j*ω0*t) +Ω17rc*Ω37bc*np.exp(-1j*(-ω-2*ω0)*t-1j*ϕ) +Ω17bc*Ω37rc*np.exp(+1j*(-ω+2*ω0)*t+1j*ϕ) +Ω17rc*Ω37rc*np.exp(+2j*ω0*t) )/(-4*self.Δc)
        def H22_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω26bc**2 + Ω26bc*Ω26rc*np.exp(1j*ω*t+1j*ϕ) + Ω26bc*Ω26rc*np.exp(-1j*ω*t-1j*ϕ) +Ω26rc**2 +Ω27bc**2 + Ω27bc*Ω27rc*np.exp(1j*ω*t+1j*ϕ) + Ω27bc*Ω27rc*np.exp(-1j*ω*t-1j*ϕ) + Ω27rc**2 +Ω28bc**2 + Ω28bc*Ω28rc*np.exp(1j*ω*t+1j*ϕ) + Ω28bc*Ω28rc*np.exp(-1j*ω*t-1j*ϕ) +Ω28rc**2)/(-4*self.Δc)
        def H23_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27bc*Ω37bc*np.exp(-1j*ω0*t) + Ω27rc*Ω37bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) + Ω27bc*Ω37rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) + Ω27rc*Ω37rc*np.exp(-1j*(ω0)*t) +Ω28bc*Ω38bc*np.exp(-1j*ω0*t) + Ω28rc*Ω38bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) + Ω28bc*Ω38rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) + Ω28rc*Ω38rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H23d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27bc*Ω37bc*np.exp(+1j*ω0*t) + Ω27rc*Ω37bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) + Ω27bc*Ω37rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) + Ω27rc*Ω37rc*np.exp(+1j*(ω0)*t) +Ω28bc*Ω38bc*np.exp(+1j*ω0*t) + Ω28rc*Ω38bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) + Ω28bc*Ω38rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) + Ω28rc*Ω38rc*np.exp(+1j*(ω0)*t))/(-4*self.Δc)
        def H24_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28bc*Ω48bc*np.exp(-2j*ω0*t) + Ω28rc*Ω48bc*np.exp(+1j*(-ω-2*ω0)*t-1j*ϕ) + Ω28bc*Ω48rc*np.exp(-1j*(-ω+2*ω0)*t+1j*ϕ)+ Ω28rc*Ω48rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H24d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28bc*Ω48bc*np.exp(+2j*ω0*t) + Ω28rc*Ω48bc*np.exp(-1j*(-ω-2*ω0)*t+1j*ϕ) + Ω28bc*Ω48rc*np.exp(+1j*(-ω+2*ω0)*t-1j*ϕ)+ Ω28rc*Ω48rc*np.exp(+2j*ω0*t))/(-4*self.Δc)
        def H33_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω37bc**2 +Ω37bc*Ω37rc*np.exp(1j*ω*t+1j*ϕ) +Ω37bc*Ω37rc*np.exp(-1j*ω*t-1j*ϕ) +Ω37rc**2 +Ω38bc**2 +Ω38bc*Ω38rc*np.exp(-1j*ω*t-1j*ϕ) +Ω38bc*Ω38rc*np.exp(1j*ω*t+1j*ϕ) +Ω38rc**2 +Ω39bc**2 +Ω39bc*Ω39rc*np.exp(1j*ω*t+1j*ϕ) +Ω39bc*Ω39rc*np.exp(-1j*ω*t-1j*ϕ) +Ω39rc**2)/(-4*self.Δc)
        def H34_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38bc*Ω48bc*np.exp(-1j*ω0*t) +Ω38rc*Ω48bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) +Ω38bc*Ω48rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) +Ω38rc*Ω48rc*np.exp(-1j*(ω0)*t) +Ω39bc*Ω49bc*np.exp(-1j*ω0*t) +Ω39rc*Ω49bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) +Ω39bc*Ω49rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) +Ω39rc*Ω49rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H34d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38bc*Ω48bc*np.exp(+1j*ω0*t) +Ω38rc*Ω48bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) +Ω38bc*Ω48rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) +Ω38rc*Ω48rc*np.exp(+1j*(ω0)*t) +Ω39bc*Ω49bc*np.exp(+1j*ω0*t) +Ω39rc*Ω49bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) +Ω39bc*Ω49rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) +Ω39rc*Ω49rc*np.exp(+1j*(ω0)*t))/(-4*self.Δc)
        def H35_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39bc*Ω59bc*np.exp(-2j*ω0*t) +Ω39rc*Ω59bc*np.exp(+1j*(-ω-2*ω0)*t-1j*ϕ) +Ω39bc*Ω59rc*np.exp(-1j*(-ω+2*ω0)*t+1j*ϕ)  +Ω39rc*Ω59rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H35d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39bc*Ω59bc*np.exp(+2j*ω0*t) +Ω39rc*Ω59bc*np.exp(-1j*(-ω-2*ω0)*t+1j*ϕ) +Ω39bc*Ω59rc*np.exp(+1j*(-ω+2*ω0)*t-1j*ϕ)  +Ω39rc*Ω59rc*np.exp(+2j*ω0*t))/(-4*self.Δc)
        def H44_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω48bc**2 +Ω48bc*Ω48rc*np.exp(1j*ω*t+1j*ϕ) +Ω48bc*Ω48rc*np.exp(-1j*ω*t-1j*ϕ)+Ω48rc**2 +Ω49bc**2 +Ω49bc*Ω49rc*np.exp(-1j*ω*t-1j*ϕ) +Ω49bc*Ω49rc*np.exp(+1j*ω*t+1j*ϕ) + Ω49rc**2)/(-4*self.Δc)
        def H45_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49bc*Ω59bc*np.exp(-1j*ω0*t) + Ω49rc*Ω59bc*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) +Ω49rc*Ω59rc*np.exp(-1j*(ω0)*t) + Ω49bc*Ω59rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ))/(-4*self.Δc)
        def H45d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49bc*Ω59bc*np.exp(+1j*ω0*t) + Ω49rc*Ω59bc*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) +Ω49rc*Ω59rc*np.exp(+1j*(ω0)*t) + Ω49bc*Ω59rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ))/(-4*self.Δc)
        def H55_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω59bc**2 + Ω59bc*Ω59rc*np.exp(1j*ω*t+1j*ϕ) + Ω59bc*Ω59rc*np.exp(-1j*ω*t-1j*ϕ) +Ω59rc**2)/(-4*self.Δc) 

        # matrices to be multiplied by time dependant functions above to construct hamiltonian
        H00 = basis(6,0)*basis(6,0).dag()
        H01 = basis(6,1)*basis(6,0).dag()
        H02 = basis(6,2)*basis(6,0).dag()
        H03 = basis(6,3)*basis(6,0).dag()
        H04 = basis(6,4)*basis(6,0).dag()
        H05 = basis(6,5)*basis(6,0).dag()
        H11 = basis(6,1)*basis(6,1).dag()
        H12 = basis(6,2)*basis(6,1).dag()
        H13 = basis(6,3)*basis(6,1).dag()
        H14 = basis(6,4)*basis(6,1).dag()
        H15 = basis(6,5)*basis(6,1).dag()
        H22 = basis(6,2)*basis(6,2).dag()
        H23 = basis(6,3)*basis(6,2).dag()
        H24 = basis(6,4)*basis(6,2).dag()
        H25 = basis(6,5)*basis(6,2).dag()
        H33 = basis(6,3)*basis(6,3).dag()
        H34 = basis(6,4)*basis(6,3).dag()
        H35 = basis(6,5)*basis(6,3).dag()
        H44 = basis(6,4)*basis(6,4).dag()
        H45 = basis(6,5)*basis(6,4).dag()
        H55 = basis(6,5)*basis(6,5).dag()
        
        # full constructed hamiltonian, not including counter rotating terms, or F states
        if not self.counter_rot and not self.F_states:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t]]
        
        # full hamiltonian, including counter rotating terms, not including F states
        if self.counter_rot and not self.F_states:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t],
                      [H00,H00_tc],[H01,H01_tc],[H01.dag(),H01d_tc],[H02,H02_tc],[H02.dag(),H02d_tc],
                      [H11,H11_tc],[H12,H12_tc],[H12.dag(),H12d_tc],[H13,H13_tc],[H13.dag(),H13d_tc],
                      [H22,H22_tc],[H23,H23_tc],[H23.dag(),H23d_tc],[H24,H24_tc],[H24.dag(),H24d_tc],
                      [H33,H33_tc],[H34,H34_tc],[H34.dag(),H34d_tc],[H35,H35_tc],[H35.dag(),H35d_tc],
                      [H44,H44_tc],[H45,H45_tc],[H45.dag(),H45d_tc],
                      [H55,H55_tc]]
            
        # full hamiltonian, including F states, not including counter rotating terms
        if self.F_states and not self.counter_rot:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t],
                      [H00,H00_tf],[H01,H01_tf],[H01.dag(),H01d_tf],[H02,H02_tf],[H02.dag(),H02d_tf],
                      [H11,H11_tf],[H12,H12_tf],[H12.dag(),H12d_tf],[H13,H13_tf],[H13.dag(),H13d_tf],
                      [H22,H22_tf],[H23,H23_tf],[H23.dag(),H23d_tf],[H24,H24_tf],[H24.dag(),H24d_tf],
                      [H33,H33_tf],[H34,H34_tf],[H34.dag(),H34d_tf],[H35,H35_tf],[H35.dag(),H35d_tf],
                      [H44,H44_tf],[H45,H45_tf],[H45.dag(),H45d_tf],
                      [H55,H55_tf]]
            
        # full hamiltonian, including F states, including counter rotating terms
        if self.counter_rot == True:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t],
                      [H00,H00_tf],[H01,H01_tf],[H01.dag(),H01d_tf],[H02,H02_tf],[H02.dag(),H02d_tf],
                      [H11,H11_tf],[H12,H12_tf],[H12.dag(),H12d_tf],[H13,H13_tf],[H13.dag(),H13d_tf],
                      [H22,H22_tf],[H23,H23_tf],[H23.dag(),H23d_tf],[H24,H24_tf],[H24.dag(),H24d_tf],
                      [H33,H33_tf],[H34,H34_tf],[H34.dag(),H34d_tf],[H35,H35_tf],[H35.dag(),H35d_tf],
                      [H44,H44_tf],[H45,H45_tf],[H45.dag(),H45d_tf],
                      [H55,H55_tf],
                      [H00,H00_tc],[H01,H01_tc],[H01.dag(),H01d_tc],[H02,H02_tc],[H02.dag(),H02d_tc],
                      [H11,H11_tc],[H12,H12_tc],[H12.dag(),H12d_tc],[H13,H13_tc],[H13.dag(),H13d_tc],
                      [H22,H22_tc],[H23,H23_tc],[H23.dag(),H23d_tc],[H24,H24_tc],[H24.dag(),H24d_tc],
                      [H33,H33_tc],[H34,H34_tc],[H34.dag(),H34d_tc],[H35,H35_tc],[H35.dag(),H35d_tc],
                      [H44,H44_tc],[H45,H45_tc],[H45.dag(),H45d_tc],
                      [H55,H55_tc]]
            
        # arguments that appear in the time dependant components of the hamiltonian
        # ω: raman beam detuning, ω0: zeeman splitting frequency
        args = {'ω':self.ω, 'ω0':self.ω0}
                      
        # solve for bank of state vectors at each point specified in t, driven by the hamiltonian above
        self.ψ = sesolve(self.H,self.ψ,t,args=args,options=Options(nsteps=1e9),progress_bar=progress_bar).states

        return    
    
    def simulate_PS(self, tpi, f, ϵ, s0):
        # simulation of n-photon pi pulse, ramped by sin^2(x) or sin^4(x)
        # f: relative length of ramp relative to pulse time
        # ϵ: relative error in length of pi pulse time
        
        self.ψ = basis(6,s0)   # intialize state vector
        self.get_rabi_frequencies()   # refresh values of single beam rabi frequencies, incase power has been updated
        
        # no pulse shaping (rectangular pulse)
        if self.PS_power == 0:
            self.t = np.linspace(1e-20, tpi, 1000)   # total time to simulate
            env = np.ones((len(self.t)))   # envelope is just the identity
        
        # sin^2 pulse parameters
        if self.PS_power == 2:
            t2 = tpi*(f-1)   # ramp on / off time
            t02 = tpi - t2   # DC pulse on time
            w = np.pi/(2*t2)   # frequency of sin^2 corresponding to ramp pulse time
            self.t = np.linspace(1e-20,(2*t2 + t02)*(1 + ϵ) , 1000)   # total time to simulate

            env = np.zeros((len(self.t)))   # intensity enveleope function
            
            # three different regions of the pulse: ramp on, DC, ramp off
            m1 = (self.t > 0 & (self.t < t2))   # indices of t corresponding to ramp on time
            m2 = ((self.t > t2) & (self.t < (t02 + t2)*(1 + ϵ)))   # indices of t corresponding to DC pulse time
            m3 = ((self.t > (t02 + t2)*(1 + ϵ)) & (self.t <= (t02 + 2*t2)*(1 + ϵ)))   # indices of time corresponding to ramp off time

            # draw the pulse envelope
            env[m1] = np.sin(w*(self.t)[m1] )**self.PS_power   
            env[m2] = (self.t/self.t)[m2]
            env[m3] = (np.sin(w*(self.t -  t2)*(1 + ϵ))[m3])**self.PS_power
            
        # sin^2 pulse parameters
        if self.PS_power == 2:
            t2 = tpi*(f-1)   # ramp on / off time
            t02 = (tpi - t2)*(1 + ϵ)    # DC pulse on time
            w = np.pi/(2*t2)   # frequency of sin^2 corresponding to ramp pulse time
            self.t = np.linspace(1e-20,(2*t2 + t02) , 1000)   # total time to simulate

            env = np.zeros((len(self.t)))   # intensity enveleope function
            
            # three different regions of the pulse: ramp on, DC, ramp off
            m1 = (self.t > 0 & (self.t < t2))   # indices of t corresponding to ramp on time
            m2 = ((self.t > t2) & (self.t < (t02 + t2)))   # indices of t corresponding to DC pulse time
            m3 = ((self.t > (t02 + t2)) & (self.t <= (t02 + 2*t2)))   # indices of time corresponding to ramp off time

            # draw the pulse envelope
            env[m1] = np.sin(w*(self.t)[m1] )**self.PS_power   
            env[m2] = (self.t/self.t)[m2]
            env[m3] = (np.sin(w*(self.t - t02 - 2*t2))[m3])**self.PS_power
            
        # sin^4 pulse parameter
        if self.PS_power == 4:
            t4 = 4*(f - 1)*tpi/5   # ramp on / off time
            t04 = (tpi - 3*t4/4)   # DC pulse on time
            w = np.pi/(2*t4)   # frequency of sin^4 corresponding to ramp pulse time
            self.t = np.linspace(1e-20,(2*t4 + t04)*(1 + ϵ) , 1000)   # total time to simulate
            
            env = np.zeros((len(self.t)))   # intensity pulse envelope function
            
            # three different regions of the pulse: ramp on, DC, ramp off
            m1 = (self.t > 0 & (self.t < t4))
            m2 = ((self.t > t4) & (self.t < (t04 + t4)*(1 + ϵ)))
            m3 = ((self.t > (t04 + t4)*(1 + ϵ)) & (self.t <= (t04 + 2*t4)*(1 + ϵ)))
            
            # draw the pulse envelope
            env[m1] = np.sin(w*(self.t)[m1] )**self.PS_power
            env[m2] = (self.t/self.t)[m2]
            env[m3] = (np.sin(w*(self.t - t04 - 2*t4)*(1 + ϵ))[m3])**self.PS_power  
            
        # sin^4 pulse parameter
        if self.PS_power == 4:
            t4 = 4*(f - 1)*tpi/5   # ramp on / off time
            t04 = (tpi - 3*t4/4)*(1 + ϵ)   # DC pulse on time
            w = np.pi/(2*t4)   # frequency of sin^4 corresponding to ramp pulse time
            self.t = np.linspace(1e-20, 2*t4 + t04, 1000)   # total time to simulate
            
            env = np.zeros((len(self.t)))   # intensity pulse envelope function
            
            # three different regions of the pulse: ramp on, DC, ramp off
            m1 = (self.t > 0 & (self.t < t4))
            m2 = ((self.t > t4) & (self.t < (t04 + t4)))
            m3 = ((self.t > (t04 + t4)) & (self.t <= (t04 + 2*t4)))
            
            # draw the pulse envelope
            env[m1] = np.sin(w*(self.t)[m1] )**self.PS_power
            env[m2] = (self.t/self.t)[m2]
            env[m3] = (np.sin(w*(self.t - t04 - 2*t4))[m3])**self.PS_power  
            
        self.env = env
        
        # enveleope function to multiply each single beam rabi frequency at any given point in time t_val
        # single beam rabi frequency goes like sqrt int. so multiply by sqrt(env)
        # 3very 
        def e(t_val):
            return np.interp(t_val, self.t, self.env)**.5
        
        # basic terms
        Ω06r = self.Ω06r
        Ω16r = self.Ω16r
        Ω26r = self.Ω26r
        Ω17r = self.Ω17r
        Ω27r = self.Ω27r
        Ω37r = self.Ω37r
        Ω28r = self.Ω28r
        Ω38r = self.Ω38r
        Ω48r = self.Ω48r 
        Ω39r = self.Ω39r
        Ω49r = self.Ω49r
        Ω59r = self.Ω59r 
        Ω06b = self.Ω06b
        Ω16b = self.Ω16b
        Ω26b = self.Ω26b
        Ω17b = self.Ω17b
        Ω27b = self.Ω27b
        Ω37b = self.Ω37b
        Ω28b = self.Ω28b
        Ω38b = self.Ω38b
        Ω48b = self.Ω48b 
        Ω39b = self.Ω39b
        Ω49b = self.Ω49b
        Ω59b = self.Ω59b 
        
        # counter rotating terms
        Ω06rc = self.Ω06rc
        Ω16rc = self.Ω16rc
        Ω26rc = self.Ω26rc
        Ω17rc = self.Ω17rc
        Ω27rc = self.Ω27rc
        Ω37rc = self.Ω37rc
        Ω28rc = self.Ω28rc
        Ω38rc = self.Ω38rc
        Ω48rc = self.Ω48rc 
        Ω39rc = self.Ω39rc
        Ω49rc = self.Ω49rc
        Ω59rc = self.Ω59rc 
        Ω06bc = self.Ω06bc
        Ω16bc = self.Ω16bc
        Ω26bc = self.Ω26bc
        Ω17bc = self.Ω17bc
        Ω27bc = self.Ω27bc
        Ω37bc = self.Ω37bc
        Ω28bc = self.Ω28bc
        Ω38bc = self.Ω38bc
        Ω48bc = self.Ω48bc 
        Ω39bc = self.Ω39bc
        Ω49bc = self.Ω49bc
        Ω59bc = self.Ω59bc 
        
        # F states
        Ω010r = self.Ω010r
        Ω011r = self.Ω011r
        Ω012r = self.Ω012r
        Ω111r = self.Ω111r 
        Ω112r = self.Ω112r 
        Ω113r = self.Ω113r
        Ω212r = self.Ω212r 
        Ω213r = self.Ω213r
        Ω214r = self.Ω214r 
        Ω313r = self.Ω313r
        Ω314r = self.Ω314r
        Ω315r = self.Ω315r
        Ω414r = self.Ω414r
        Ω415r = self.Ω415r
        Ω416r = self.Ω416r 
        Ω515r = self.Ω515r 
        Ω516r = self.Ω516r
        Ω517r = self.Ω517r
        
        Ω010b = self.Ω010b
        Ω011b = self.Ω011b
        Ω012b = self.Ω012b
        Ω111b = self.Ω111b 
        Ω112b = self.Ω112b 
        Ω113b = self.Ω113b
        Ω212b = self.Ω212b 
        Ω213b = self.Ω213b
        Ω214b = self.Ω214b 
        Ω313b = self.Ω313b
        Ω314b = self.Ω314b
        Ω315b = self.Ω315b
        Ω414b = self.Ω414b
        Ω415b = self.Ω415b
        Ω416b = self.Ω416b 
        Ω515b = self.Ω515b 
        Ω516b = self.Ω516b
        Ω517b = self.Ω517b

        ϕ = self.ϕ   # raman beam relative phase
        
        # basic terms
        def H00_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b**2*e(t)*e(t) + Ω06b*e(t)*Ω06r*np.exp(-1j*ω*t + 1j*ϕ)+ Ω06b*e(t)*Ω06r*np.exp(+1j*ω*t - 1j*ϕ) + Ω06r**2)/(-4*self.Δ)
        def H01_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*e(t)*Ω16b*e(t)*np.exp(-1j*ω0*t) + Ω06r*Ω16b*e(t)*np.exp(+1j*(ω-ω0)*t -1j*ϕ) + Ω06b*e(t)*Ω16r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω06r*Ω16r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H01d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*e(t)*Ω16b*e(t)*np.exp(+1j*ω0*t) + Ω06r*Ω16b*e(t)*np.exp(-1j*(ω-ω0)*t +1j*ϕ) + Ω06b*e(t)*Ω16r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω06r*Ω16r*np.exp(+1j*(ω0)*t) )/(-4*self.Δ)
        def H02_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*e(t)*Ω26b*e(t)*np.exp(-2j*ω0*t) + Ω06r*Ω26b*e(t)*np.exp(+1j*(ω-2*ω0)*t -1j*ϕ) + Ω06b*e(t)*Ω26r*np.exp(-1j*(ω+2*ω0)*t +1j*ϕ) + Ω06r*Ω26r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H02d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06b*e(t)*Ω26b*e(t)*np.exp(+2j*ω0*t) + Ω06r*Ω26b*e(t)*np.exp(-1j*(ω-2*ω0)*t +1j*ϕ) + Ω06b*e(t)*Ω26r*np.exp(+1j*(ω+2*ω0)*t -1j*ϕ) + Ω06r*Ω26r*np.exp(+2j*ω0*t))/(-4*self.Δ)
        def H11_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16b**2*e(t)*e(t) + Ω16r**2 + Ω17b**2*e(t)*e(t) + Ω17b*e(t)*Ω17r*np.exp(-1j*ω*t +1j*ϕ)+ Ω17b*e(t)*Ω17r*np.exp(+1j*ω*t -1j*ϕ) + Ω17r**2 + Ω16b*e(t)*Ω16r*np.exp(-1j*ω*t +1j*ϕ)+ Ω16b*e(t)*Ω16r*np.exp(+1j*ω*t -1j*ϕ))/(-4*self.Δ) 
        def H12_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16b*e(t)*Ω26b*e(t)*np.exp(-1j*ω0*t) + Ω16r*Ω26b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ)+ Ω16b*e(t)*Ω26r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω16r*Ω26r*np.exp(-1j*(ω0)*t) + Ω27b*e(t)*Ω17b*e(t)*np.exp(-1j*ω0*t) + Ω17r*Ω27b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω17b*e(t)*Ω27r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω17r*Ω27r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H12d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16b*e(t)*Ω26b*e(t)*np.exp(+1j*ω0*t) + Ω16r*Ω26b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ)+ Ω16b*e(t)*Ω26r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω16r*Ω26r*np.exp(+1j*(ω0)*t) + Ω27b*e(t)*Ω17b*e(t)*np.exp(+1j*ω0*t) + Ω17r*Ω27b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω17b*e(t)*Ω27r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω17r*Ω27r*np.exp(+1j*(ω0)*t))/(-4*self.Δ)
        def H13_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17b*e(t)*Ω37b*e(t)*np.exp(-2j*ω0*t) +Ω17r*Ω37b*e(t)*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) +Ω17b*e(t)*Ω37r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ) +Ω17r*Ω37r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H13d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17b*e(t)*Ω37b*e(t)*np.exp(+2j*ω0*t) +Ω17r*Ω37b*e(t)*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) +Ω17b*e(t)*Ω37r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ) +Ω17r*Ω37r*np.exp(+2j*ω0*t) )/(-4*self.Δ)
        # 854 lightshift is included in H22(t). Also included in H33, H44, H55, with increasing magnitude as governed by CG coefficients
        def H22_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω26b**2*e(t)*e(t) + Ω26b*e(t)*Ω26r*np.exp(-1j*ω*t+1j*ϕ) + Ω26b*e(t)*Ω26r*np.exp(1j*ω*t-1j*ϕ) +Ω26r**2 +Ω27b**2*e(t)*e(t) + Ω27b*e(t)*Ω27r*np.exp(-1j*ω*t+1j*ϕ) + Ω27b*e(t)*Ω27r*np.exp(1j*ω*t-1j*ϕ) + Ω27r**2 +Ω28b**2*e(t)*e(t) + Ω28b*e(t)*Ω28r*np.exp(-1j*ω*t+1j*ϕ) + Ω28b*e(t)*Ω28r*np.exp(1j*ω*t-1j*ϕ) +Ω28r**2)/(-4*self.Δ)  - self.ls_854
        def H23_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27b*e(t)*Ω37b*e(t)*np.exp(-1j*ω0*t) + Ω27r*Ω37b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω27b*e(t)*Ω37r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω27r*Ω37r*np.exp(-1j*(ω0)*t) +Ω28b*e(t)*Ω38b*e(t)*np.exp(-1j*ω0*t) + Ω28r*Ω38b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω28b*e(t)*Ω38r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω28r*Ω38r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H23d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27b*e(t)*Ω37b*e(t)*np.exp(+1j*ω0*t) + Ω27r*Ω37b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω27b*e(t)*Ω37r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω27r*Ω37r*np.exp(+1j*(ω0)*t) +Ω28b*e(t)*Ω38b*e(t)*np.exp(+1j*ω0*t) + Ω28r*Ω38b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω28b*e(t)*Ω38r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω28r*Ω38r*np.exp(+1j*(ω0)*t))/(-4*self.Δ)
        def H24_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28b*e(t)*Ω48b*e(t)*np.exp(-2j*ω0*t) + Ω28r*Ω48b*e(t)*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) + Ω28b*e(t)*Ω48r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)+ Ω28r*Ω48r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H24d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28b*e(t)*Ω48b*e(t)*np.exp(+2j*ω0*t) + Ω28r*Ω48b*e(t)*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) + Ω28b*e(t)*Ω48r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)+ Ω28r*Ω48r*np.exp(+2j*ω0*t))/(-4*self.Δ)
        def H33_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω37b**2*e(t)*e(t) +Ω37b*e(t)*Ω37r*np.exp(-1j*ω*t+1j*ϕ) +Ω37b*e(t)*Ω37r*np.exp(+1j*ω*t-1j*ϕ) +Ω37r**2 +Ω38b**2*e(t)*e(t) +Ω38b*e(t)*Ω38r*np.exp(+1j*ω*t-1j*ϕ) +Ω38b*e(t)*Ω38r*np.exp(-1j*ω*t+1j*ϕ) +Ω38r**2 +Ω39b**2*e(t)*e(t) +Ω39b*e(t)*Ω39r*np.exp(-1j*ω*t+1j*ϕ) +Ω39b*e(t)*Ω39r*np.exp(+1j*ω*t-1j*ϕ) +Ω39r**2)/(-4*self.Δ) - 3*self.ls_854
        def H34_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38b*e(t)*Ω48b*e(t)*np.exp(-1j*ω0*t) +Ω38r*Ω48b*e(t)*np.exp(1j*(ω-ω0)*t-1j*ϕ) +Ω38b*e(t)*Ω48r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω38r*Ω48r*np.exp(-1j*(ω0)*t) +Ω39b*e(t)*Ω49b*e(t)*np.exp(-1j*ω0*t) +Ω39r*Ω49b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω39b*e(t)*Ω49r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω39r*Ω49r*np.exp(-1j*(ω0)*t))/(-4*self.Δ)
        def H34d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38b*e(t)*Ω48b*e(t)*np.exp(+1j*ω0*t) +Ω38r*Ω48b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω38b*e(t)*Ω48r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω38r*Ω48r*np.exp(+1j*(ω0)*t) +Ω39b*e(t)*Ω49b*e(t)*np.exp(+1j*ω0*t) +Ω39r*Ω49b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω39b*e(t)*Ω49r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω39r*Ω49r*np.exp(+1j*(ω0)*t))/(-4*self.Δ)
        def H35_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39b*e(t)*Ω59b*e(t)*np.exp(-2j*ω0*t) +Ω39r*Ω59b*e(t)*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) +Ω39b*e(t)*Ω59r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)  +Ω39r*Ω59r*np.exp(-2j*ω0*t))/(-4*self.Δ)
        def H35d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39b*e(t)*Ω59b*e(t)*np.exp(+2j*ω0*t) +Ω39r*Ω59b*e(t)*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) +Ω39b*e(t)*Ω59r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)  +Ω39r*Ω59r*np.exp(+2j*ω0*t))/(-4*self.Δ)
        def H44_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω48b**2*e(t)*e(t) +Ω48b*e(t)*Ω48r*np.exp(-1j*ω*t+1j*ϕ) +Ω48b*e(t)*Ω48r*np.exp(+1j*ω*t-1j*ϕ)+Ω48r**2 +Ω49b**2*e(t)*e(t) +Ω49b*e(t)*Ω49r*np.exp(+1j*ω*t-1j*ϕ) +Ω49b*e(t)*Ω49r*np.exp(-1j*ω*t+1j*ϕ) + Ω49r**2)/(-4*self.Δ) - 6*self.ls_854 
        def H45_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49b*e(t)*Ω59b*e(t)*np.exp(-1j*ω0*t) + Ω49r*Ω59b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω49r*Ω59r*np.exp(-1j*(ω0)*t) + Ω49b*e(t)*Ω59r*np.exp(-1j*(ω+ω0)*t+1j*ϕ))/(-4*self.Δ)
        def H45d_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49b*e(t)*Ω59b*e(t)*np.exp(+1j*ω0*t) + Ω49r*Ω59b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω49r*Ω59r*np.exp(+1j*(ω0)*t) + Ω49b*e(t)*Ω59r*np.exp(+1j*(ω+ω0)*t-1j*ϕ))/(-4*self.Δ)
        def H55_t(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω59b**2*e(t)*e(t) + Ω59b*e(t)*Ω59r*np.exp(-1j*ω*t+1j*ϕ) + Ω59b*e(t)*Ω59r*np.exp(+1j*ω*t-1j*ϕ) +Ω59r**2)/(-4*self.Δ) - 10*self.ls_854
        
        # F states
        def H00_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b**2*e(t)*e(t) + Ω012b*e(t)*Ω012r*np.exp(-1j*ω*t + 1j*ϕ)+ Ω012b*e(t)*Ω012r*np.exp(+1j*ω*t - 1j*ϕ) + Ω012r**2)/(-4*self.ΔF)\
                   -(Ω010b**2*e(t)*e(t) + Ω010r**2 + Ω010r*Ω010b*e(t)*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)) + Ω011b**2*e(t)*e(t) + Ω011r**2 + Ω011r*Ω011b*e(t)*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)))/(-4*self.ΔF)
        def H01_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*e(t)*Ω112b*e(t)*np.exp(-1j*ω0*t) + Ω012r*Ω112b*e(t)*np.exp(+1j*(ω-ω0)*t -1j*ϕ) + Ω012b*e(t)*Ω112r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω012r*Ω112r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)\
                   -(Ω011b*e(t)*Ω111b*e(t)*np.exp(-1j*ω0*t) + Ω011r*Ω111b*e(t)*np.exp(+1j*(ω-ω0)*t -1j*ϕ) + Ω011b*e(t)*Ω111r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω011r*Ω111r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H01d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*e(t)*Ω112b*e(t)*np.exp(+1j*ω0*t) + Ω012r*Ω112b*e(t)*np.exp(-1j*(ω-ω0)*t +1j*ϕ) + Ω012b*e(t)*Ω112r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω012r*Ω112r*np.exp(+1j*(ω0)*t) )/(-4*self.ΔF)\
                   -(Ω011b*e(t)*Ω111b*e(t)*np.exp(+1j*ω0*t) + Ω011r*Ω111b*e(t)*np.exp(-1j*(ω-ω0)*t +1j*ϕ) + Ω011b*e(t)*Ω111r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω011r*Ω111r*np.exp(+1j*(ω0)*t) )/(-4*self.ΔF)
        def H02_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*e(t)*Ω212b*e(t)*np.exp(-2j*ω0*t) + Ω012r*Ω212b*e(t)*np.exp(+1j*(ω-2*ω0)*t -1j*ϕ) + Ω012b*e(t)*Ω212r*np.exp(-1j*(ω+2*ω0)*t +1j*ϕ) + Ω012r*Ω212r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H02d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω012b*e(t)*Ω212b*e(t)*np.exp(+2j*ω0*t) + Ω012r*Ω212b*e(t)*np.exp(-1j*(ω-2*ω0)*t +1j*ϕ) + Ω012b*e(t)*Ω212r*np.exp(+1j*(ω+2*ω0)*t -1j*ϕ) + Ω012r*Ω212r*np.exp(+2j*ω0*t))/(-4*self.ΔF)
        def H11_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω112b**2*e(t)*e(t) + Ω112r**2 + Ω113b**2*e(t)*e(t) + Ω113b*e(t)*Ω113r*np.exp(-1j*ω*t +1j*ϕ)+ Ω113b*e(t)*Ω113r*np.exp(+1j*ω*t -1j*ϕ) + Ω113r**2 + Ω112b*e(t)*Ω112r*np.exp(-1j*ω*t +1j*ϕ)+ Ω112b*e(t)*Ω112r*np.exp(+1j*ω*t -1j*ϕ))/(-4*self.ΔF) \
                   -(Ω111b**2*e(t)*e(t) + Ω111r**2)/(-4*self.ΔF) 
        def H12_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω112b*e(t)*Ω212b*e(t)*np.exp(-1j*ω0*t) + Ω112r*Ω212b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ)+ Ω112b*e(t)*Ω212r*np.exp(-1j*(ω+ω0)*t +1j*ϕ) + Ω112r*Ω212r*np.exp(-1j*(ω0)*t) + Ω213b*e(t)*Ω113b*e(t)*np.exp(-1j*ω0*t) + Ω113r*Ω213b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω113b*e(t)*Ω213r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω113r*Ω213r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H12d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω112b*e(t)*Ω212b*e(t)*np.exp(+1j*ω0*t) + Ω112r*Ω212b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ)+ Ω112b*e(t)*Ω212r*np.exp(+1j*(ω+ω0)*t -1j*ϕ) + Ω112r*Ω212r*np.exp(+1j*(ω0)*t) + Ω213b*e(t)*Ω113b*e(t)*np.exp(+1j*ω0*t) + Ω113r*Ω213b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω113b*e(t)*Ω213r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω113r*Ω213r*np.exp(+1j*(ω0)*t))/(-4*self.ΔF)
        def H13_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω113b*e(t)*Ω313b*e(t)*np.exp(-2j*ω0*t) +Ω113r*Ω313b*e(t)*np.exp(+1j*(ω-2*ω0)*t+1j*ϕ) +Ω113b*e(t)*Ω313r*np.exp(-1j*(ω+2*ω0)*t-1j*ϕ) +Ω113r*Ω313r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H13d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω113b*e(t)*Ω313b*e(t)*np.exp(+2j*ω0*t) +Ω113r*Ω313b*e(t)*np.exp(-1j*(ω-2*ω0)*t-1j*ϕ) +Ω113b*e(t)*Ω313r*np.exp(+1j*(ω+2*ω0)*t+1j*ϕ) +Ω113r*Ω313r*np.exp(+2j*ω0*t) )/(-4*self.ΔF)
        def H22_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω212b**2*e(t)*e(t) + Ω212b*e(t)*Ω212r*np.exp(-1j*ω*t+1j*ϕ) + Ω212b*e(t)*Ω212r*np.exp(1j*ω*t-1j*ϕ) +Ω212r**2 +Ω213b**2*e(t)*e(t) + Ω213b*e(t)*Ω213r*np.exp(-1j*ω*t+1j*ϕ) + Ω213b*e(t)*Ω213r*np.exp(1j*ω*t-1j*ϕ) + Ω213r**2 +Ω214b**2*e(t)*e(t) + Ω214b*e(t)*Ω214r*np.exp(-1j*ω*t+1j*ϕ) + Ω214b*e(t)*Ω214r*np.exp(1j*ω*t-1j*ϕ) +Ω214r**2)/(-4*self.ΔF)
        def H23_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω213b*e(t)*Ω313b*e(t)*np.exp(-1j*ω0*t) + Ω213r*Ω313b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω213b*e(t)*Ω313r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω213r*Ω313r*np.exp(-1j*(ω0)*t) +Ω214b*e(t)*Ω314b*e(t)*np.exp(-1j*ω0*t) + Ω214r*Ω314b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) + Ω214b*e(t)*Ω314r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) + Ω214r*Ω314r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H23d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω213b*e(t)*Ω313b*e(t)*np.exp(+1j*ω0*t) + Ω213r*Ω313b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω213b*e(t)*Ω313r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω213r*Ω313r*np.exp(+1j*(ω0)*t) +Ω214b*e(t)*Ω314b*e(t)*np.exp(+1j*ω0*t) + Ω214r*Ω314b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) + Ω214b*e(t)*Ω314r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) + Ω214r*Ω314r*np.exp(+1j*(ω0)*t))/(-4*self.ΔF)
        def H24_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω214b*e(t)*Ω414b*e(t)*np.exp(-2j*ω0*t) + Ω214r*Ω414b*e(t)*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) + Ω214b*e(t)*Ω414r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)+ Ω214r*Ω414r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H24d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω214b*e(t)*Ω414b*e(t)*np.exp(+2j*ω0*t) + Ω214r*Ω414b*e(t)*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) + Ω214b*e(t)*Ω414r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)+ Ω214r*Ω414r*np.exp(+2j*ω0*t))/(-4*self.ΔF)
        def H33_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω313b**2*e(t)*e(t) +Ω313b*e(t)*Ω313r*np.exp(-1j*ω*t+1j*ϕ) +Ω313b*e(t)*Ω313r*np.exp(+1j*ω*t-1j*ϕ) +Ω313r**2 +Ω314b**2*e(t)*e(t) +Ω314b*e(t)*Ω314r*np.exp(+1j*ω*t-1j*ϕ) +Ω314b*e(t)*Ω314r*np.exp(-1j*ω*t+1j*ϕ) +Ω314r**2 +Ω315b**2*e(t)*e(t) +Ω315b*e(t)*Ω315r*np.exp(-1j*ω*t+1j*ϕ) +Ω315b*e(t)*Ω315r*np.exp(+1j*ω*t-1j*ϕ) +Ω315r**2)/(-4*self.ΔF)
        def H34_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω314b*e(t)*Ω414b*e(t)*np.exp(-1j*ω0*t) +Ω314r*Ω414b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω314b*e(t)*Ω414r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω314r*Ω414r*np.exp(-1j*(ω0)*t) +Ω315b*e(t)*Ω415b*e(t)*np.exp(-1j*ω0*t) +Ω315r*Ω415b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω315b*e(t)*Ω415r*np.exp(-1j*(ω+ω0)*t+1j*ϕ) +Ω315r*Ω415r*np.exp(-1j*(ω0)*t))/(-4*self.ΔF)
        def H34d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω314b*e(t)*Ω414b*e(t)*np.exp(+1j*ω0*t) +Ω314r*Ω414b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω314b*e(t)*Ω414r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω314r*Ω414r*np.exp(+1j*(ω0)*t) +Ω315b*e(t)*Ω415b*e(t)*np.exp(+1j*ω0*t) +Ω315r*Ω415b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω315b*e(t)*Ω415r*np.exp(+1j*(ω+ω0)*t-1j*ϕ) +Ω315r*Ω415r*np.exp(+1j*(ω0)*t))/(-4*self.ΔF)
        def H35_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω315b*e(t)*Ω515b*e(t)*np.exp(-2j*ω0*t) +Ω315r*Ω515b*e(t)*np.exp(+1j*(ω-2*ω0)*t-1j*ϕ) +Ω315b*e(t)*Ω515r*np.exp(-1j*(ω+2*ω0)*t+1j*ϕ)  +Ω315r*Ω515r*np.exp(-2j*ω0*t))/(-4*self.ΔF)
        def H35d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω315b*e(t)*Ω515b*e(t)*np.exp(+2j*ω0*t) +Ω315r*Ω515b*e(t)*np.exp(-1j*(ω-2*ω0)*t+1j*ϕ) +Ω315b*e(t)*Ω515r*np.exp(+1j*(ω+2*ω0)*t-1j*ϕ)  +Ω315r*Ω515r*np.exp(+2j*ω0*t))/(-4*self.ΔF)
        def H44_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω414b**2*e(t)*e(t) +Ω414b*e(t)*Ω414r*np.exp(-1j*ω*t+1j*ϕ) +Ω414b*e(t)*Ω414r*np.exp(+1j*ω*t-1j*ϕ)+Ω414r**2 +Ω415b**2*e(t)*e(t) +Ω415b*e(t)*Ω415r*np.exp(+1j*ω*t-1j*ϕ) +Ω415b*e(t)*Ω415r*np.exp(-1j*ω*t+1j*ϕ) + Ω415r**2)/(-4*self.ΔF)\
                   -(Ω416b**2*e(t)*e(t) + Ω416r**2)/(-4*self.ΔF) 
        def H45_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω415b*e(t)*Ω515b*e(t)*np.exp(-1j*ω0*t) + Ω415r*Ω515b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω415r*Ω515r*np.exp(-1j*(ω0)*t) + Ω415b*e(t)*Ω515r*np.exp(-1j*(ω+ω0)*t+1j*ϕ))/(-4*self.ΔF)\
                   -(Ω416b*e(t)*Ω516b*e(t)*np.exp(-1j*ω0*t) + Ω416r*Ω516b*e(t)*np.exp(+1j*(ω-ω0)*t-1j*ϕ) +Ω416r*Ω516r*np.exp(-1j*(ω0)*t) + Ω416b*e(t)*Ω516r*np.exp(-1j*(ω+ω0)*t+1j*ϕ))/(-4*self.ΔF)
        def H45d_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω415b*e(t)*Ω515b*e(t)*np.exp(+1j*ω0*t) + Ω415r*Ω515b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω415r*Ω515r*np.exp(+1j*(ω0)*t) + Ω415b*e(t)*Ω515r*np.exp(+1j*(ω+ω0)*t-1j*ϕ))/(-4*self.ΔF)\
                   -(Ω416b*e(t)*Ω516b*e(t)*np.exp(+1j*ω0*t) + Ω416r*Ω516b*e(t)*np.exp(-1j*(ω-ω0)*t+1j*ϕ) +Ω416r*Ω516r*np.exp(+1j*(ω0)*t) + Ω416b*e(t)*Ω516r*np.exp(+1j*(ω+ω0)*t-1j*ϕ))/(-4*self.ΔF)
        def H55_tf(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω515b**2*e(t)*e(t) + Ω515b*e(t)*Ω515r*np.exp(-1j*ω*t+1j*ϕ) + Ω515b*e(t)*Ω515r*np.exp(+1j*ω*t-1j*ϕ) +Ω515r**2)/(-4*self.ΔF) \
                   -(Ω516b**2*e(t)*e(t) + Ω516r**2 + Ω516r*Ω516b*e(t)*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)) + Ω517b**2*e(t)*e(t) + Ω517r**2 + Ω517r*Ω517b*e(t)*(np.exp(-1j*ω*t) + np.exp(1j*ω*t)))/(-4*self.ΔF)
        
        #  counter rotating terms
        ϕ = -ϕ   # relative phase switches signs for these terms    
        def H00_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc**2*e(t)*e(t) + Ω06bc*e(t)*Ω06rc*np.exp(1j*ω*t + 1j*ϕ)+ Ω06bc*e(t)*Ω06rc*np.exp(-1j*ω*t - 1j*ϕ) + Ω06rc**2)/(-4*self.Δc)
        def H01_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*e(t)*Ω16bc*e(t)*np.exp(-1j*ω0*t) + Ω06rc*Ω16bc*e(t)*np.exp(+1j*(-ω-ω0)*t -1j*ϕ) + Ω06bc*e(t)*Ω16rc*np.exp(-1j*(-ω+ω0)*t +1j*ϕ) + Ω06rc*Ω16rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H01d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*e(t)*Ω16bc*e(t)*np.exp(+1j*ω0*t) + Ω06rc*Ω16bc*e(t)*np.exp(-1j*(-ω-ω0)*t +1j*ϕ) + Ω06bc*e(t)*Ω16rc*np.exp(+1j*(-ω+ω0)*t -1j*ϕ) + Ω06rc*Ω16rc*np.exp(+1j*(ω0)*t) )/(-4*self.Δc)
        def H02_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*e(t)*Ω26bc*e(t)*np.exp(-2j*ω0*t) + Ω06rc*Ω26bc*e(t)*np.exp(+1j*(-ω-2*ω0)*t -1j*ϕ) + Ω06bc*e(t)*Ω26rc*np.exp(-1j*(-ω+2*ω0)*t +1j*ϕ) + Ω06rc*Ω26rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H02d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω06bc*e(t)*Ω26bc*e(t)*np.exp(+2j*ω0*t) + Ω06rc*Ω26bc*e(t)*np.exp(-1j*(-ω-2*ω0)*t +1j*ϕ) + Ω06bc*e(t)*Ω26rc*np.exp(+1j*(-ω+2*ω0)*t -1j*ϕ) + Ω06rc*Ω26rc*np.exp(+2j*ω0*t))/(-4*self.Δc)
        def H11_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16bc**2*e(t)*e(t) + Ω16rc**2 + Ω17bc**2*e(t)*e(t) + Ω17bc*e(t)*Ω17rc*np.exp(+1j*ω*t +1j*ϕ)+ Ω17bc*e(t)*Ω17rc*np.exp(-1j*ω*t -1j*ϕ) + Ω17rc**2 + Ω16bc*e(t)*Ω16rc*np.exp(1j*ω*t +1j*ϕ)+ Ω16bc*e(t)*Ω16rc*np.exp(-1j*ω*t -1j*ϕ))/(-4*self.Δc)
        def H12_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16bc*e(t)*Ω26bc*e(t)*np.exp(-1j*ω0*t) + Ω16rc*Ω26bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ)+ Ω16bc*e(t)*Ω26rc*np.exp(-1j*(-ω+ω0)*t +1j*ϕ) + Ω16rc*Ω26rc*np.exp(-1j*(ω0)*t) + Ω27bc*e(t)*Ω17bc*e(t)*np.exp(-1j*ω0*t) + Ω17rc*Ω27bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) + Ω17bc*e(t)*Ω27rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) + Ω17rc*Ω27rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H12d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω16bc*e(t)*Ω26bc*e(t)*np.exp(+1j*ω0*t) + Ω16rc*Ω26bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ)+ Ω16bc*e(t)*Ω26rc*np.exp(+1j*(-ω+ω0)*t -1j*ϕ) + Ω16rc*Ω26rc*np.exp(+1j*(ω0)*t) + Ω27bc*e(t)*Ω17bc*e(t)*np.exp(+1j*ω0*t) + Ω17rc*Ω27bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) + Ω17bc*e(t)*Ω27rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) + Ω17rc*Ω27rc*np.exp(+1j*(ω0)*t))/(-4*self.Δc)
        def H13_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17bc*e(t)*Ω37bc*e(t)*np.exp(-2j*ω0*t) +Ω17rc*Ω37bc*e(t)*np.exp(+1j*(-ω-2*ω0)*t+1j*ϕ) +Ω17bc*e(t)*Ω37rc*np.exp(-1j*(-ω+2*ω0)*t-1j*ϕ) +Ω17rc*Ω37rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H13d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω17bc*e(t)*Ω37bc*e(t)*np.exp(+2j*ω0*t) +Ω17rc*Ω37bc*e(t)*np.exp(-1j*(-ω-2*ω0)*t-1j*ϕ) +Ω17bc*e(t)*Ω37rc*np.exp(+1j*(-ω+2*ω0)*t+1j*ϕ) +Ω17rc*Ω37rc*np.exp(+2j*ω0*t) )/(-4*self.Δc)
        def H22_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω26bc**2*e(t)*e(t) + Ω26bc*e(t)*Ω26rc*np.exp(1j*ω*t+1j*ϕ) + Ω26bc*e(t)*Ω26rc*np.exp(-1j*ω*t-1j*ϕ) +Ω26rc**2 +Ω27bc**2*e(t)*e(t) + Ω27bc*e(t)*Ω27rc*np.exp(1j*ω*t+1j*ϕ) + Ω27bc*e(t)*Ω27rc*np.exp(-1j*ω*t-1j*ϕ) + Ω27rc**2 +Ω28bc**2*e(t)*e(t) + Ω28bc*e(t)*Ω28rc*np.exp(1j*ω*t+1j*ϕ) + Ω28bc*e(t)*Ω28rc*np.exp(-1j*ω*t-1j*ϕ) +Ω28rc**2)/(-4*self.Δc)
        def H23_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27bc*e(t)*Ω37bc*e(t)*np.exp(-1j*ω0*t) + Ω27rc*Ω37bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) + Ω27bc*e(t)*Ω37rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) + Ω27rc*Ω37rc*np.exp(-1j*(ω0)*t) +Ω28bc*e(t)*Ω38bc*e(t)*np.exp(-1j*ω0*t) + Ω28rc*Ω38bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) + Ω28bc*e(t)*Ω38rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) + Ω28rc*Ω38rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H23d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω27bc*e(t)*Ω37bc*e(t)*np.exp(+1j*ω0*t) + Ω27rc*Ω37bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) + Ω27bc*e(t)*Ω37rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) + Ω27rc*Ω37rc*np.exp(+1j*(ω0)*t) +Ω28bc*e(t)*Ω38bc*e(t)*np.exp(+1j*ω0*t) + Ω28rc*Ω38bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) + Ω28bc*e(t)*Ω38rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) + Ω28rc*Ω38rc*np.exp(+1j*(ω0)*t))/(-4*self.Δc)
        def H24_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28bc*e(t)*Ω48bc*e(t)*np.exp(-2j*ω0*t) + Ω28rc*Ω48bc*e(t)*np.exp(+1j*(-ω-2*ω0)*t-1j*ϕ) + Ω28bc*e(t)*Ω48rc*np.exp(-1j*(-ω+2*ω0)*t+1j*ϕ)+ Ω28rc*Ω48rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H24d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω28bc*e(t)*Ω48bc*e(t)*np.exp(+2j*ω0*t) + Ω28rc*Ω48bc*e(t)*np.exp(-1j*(-ω-2*ω0)*t+1j*ϕ) + Ω28bc*e(t)*Ω48rc*np.exp(+1j*(-ω+2*ω0)*t-1j*ϕ)+ Ω28rc*Ω48rc*np.exp(+2j*ω0*t))/(-4*self.Δc)
        def H33_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω37bc**2*e(t)*e(t) +Ω37bc*e(t)*Ω37rc*np.exp(1j*ω*t+1j*ϕ) +Ω37bc*e(t)*Ω37rc*np.exp(-1j*ω*t-1j*ϕ) +Ω37rc**2 +Ω38bc**2*e(t)*e(t) +Ω38bc*e(t)*Ω38rc*np.exp(-1j*ω*t-1j*ϕ) +Ω38bc*e(t)*Ω38rc*np.exp(1j*ω*t+1j*ϕ) +Ω38rc**2 +Ω39bc**2*e(t)*e(t) +Ω39bc*e(t)*Ω39rc*np.exp(1j*ω*t+1j*ϕ) +Ω39bc*e(t)*Ω39rc*np.exp(-1j*ω*t-1j*ϕ) +Ω39rc**2)/(-4*self.Δc)
        def H34_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38bc*e(t)*Ω48bc*e(t)*np.exp(-1j*ω0*t) +Ω38rc*Ω48bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) +Ω38bc*e(t)*Ω48rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) +Ω38rc*Ω48rc*np.exp(-1j*(ω0)*t) +Ω39bc*e(t)*Ω49bc*e(t)*np.exp(-1j*ω0*t) +Ω39rc*Ω49bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) +Ω39bc*e(t)*Ω49rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ) +Ω39rc*Ω49rc*np.exp(-1j*(ω0)*t))/(-4*self.Δc)
        def H34d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω38bc*e(t)*Ω48bc*e(t)*np.exp(+1j*ω0*t) +Ω38rc*Ω48bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) +Ω38bc*e(t)*Ω48rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) +Ω38rc*Ω48rc*np.exp(+1j*(ω0)*t) +Ω39bc*e(t)*Ω49bc*e(t)*np.exp(+1j*ω0*t) +Ω39rc*Ω49bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) +Ω39bc*e(t)*Ω49rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ) +Ω39rc*Ω49rc*np.exp(+1j*(ω0)*t))/(-4*self.Δc)
        def H35_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39bc*e(t)*Ω59bc*e(t)*np.exp(-2j*ω0*t) +Ω39rc*Ω59bc*e(t)*np.exp(+1j*(-ω-2*ω0)*t-1j*ϕ) +Ω39bc*e(t)*Ω59rc*np.exp(-1j*(-ω+2*ω0)*t+1j*ϕ)  +Ω39rc*Ω59rc*np.exp(-2j*ω0*t))/(-4*self.Δc)
        def H35d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω39bc*e(t)*Ω59bc*e(t)*np.exp(+2j*ω0*t) +Ω39rc*Ω59bc*e(t)*np.exp(-1j*(-ω-2*ω0)*t+1j*ϕ) +Ω39bc*e(t)*Ω59rc*np.exp(+1j*(-ω+2*ω0)*t-1j*ϕ)  +Ω39rc*Ω59rc*np.exp(+2j*ω0*t))/(-4*self.Δc)
        def H44_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω48bc**2*e(t)*e(t) +Ω48bc*e(t)*Ω48rc*np.exp(1j*ω*t+1j*ϕ) +Ω48bc*e(t)*Ω48rc*np.exp(-1j*ω*t-1j*ϕ)+Ω48rc**2 +Ω49bc**2*e(t)*e(t) +Ω49bc*e(t)*Ω49rc*np.exp(-1j*ω*t-1j*ϕ) +Ω49bc*e(t)*Ω49rc*np.exp(+1j*ω*t+1j*ϕ) + Ω49rc**2)/(-4*self.Δc)
        def H45_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49bc*e(t)*Ω59bc*e(t)*np.exp(-1j*ω0*t) + Ω49rc*Ω59bc*e(t)*np.exp(+1j*(-ω-ω0)*t-1j*ϕ) +Ω49rc*Ω59rc*np.exp(-1j*(ω0)*t) + Ω49bc*e(t)*Ω59rc*np.exp(-1j*(-ω+ω0)*t+1j*ϕ))/(-4*self.Δc)
        def H45d_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω49bc*e(t)*Ω59bc*e(t)*np.exp(+1j*ω0*t) + Ω49rc*Ω59bc*e(t)*np.exp(-1j*(-ω-ω0)*t+1j*ϕ) +Ω49rc*Ω59rc*np.exp(+1j*(ω0)*t) + Ω49bc*e(t)*Ω59rc*np.exp(+1j*(-ω+ω0)*t-1j*ϕ))/(-4*self.Δc)
        def H55_tc(t,args):
            ω = args['ω']
            ω0 = args['ω0']
            return -(Ω59bc**2*e(t)*e(t) + Ω59bc*e(t)*Ω59rc*np.exp(1j*ω*t+1j*ϕ) + Ω59bc*e(t)*Ω59rc*np.exp(-1j*ω*t-1j*ϕ) +Ω59rc**2)/(-4*self.Δc) 

        # matrices to be multiplied by time dependant functions above to construct hamiltonian
        H00 = basis(6,0)*basis(6,0).dag()
        H01 = basis(6,1)*basis(6,0).dag()
        H02 = basis(6,2)*basis(6,0).dag()
        H03 = basis(6,3)*basis(6,0).dag()
        H04 = basis(6,4)*basis(6,0).dag()
        H05 = basis(6,5)*basis(6,0).dag()
        H11 = basis(6,1)*basis(6,1).dag()
        H12 = basis(6,2)*basis(6,1).dag()
        H13 = basis(6,3)*basis(6,1).dag()
        H14 = basis(6,4)*basis(6,1).dag()
        H15 = basis(6,5)*basis(6,1).dag()
        H22 = basis(6,2)*basis(6,2).dag()
        H23 = basis(6,3)*basis(6,2).dag()
        H24 = basis(6,4)*basis(6,2).dag()
        H25 = basis(6,5)*basis(6,2).dag()
        H33 = basis(6,3)*basis(6,3).dag()
        H34 = basis(6,4)*basis(6,3).dag()
        H35 = basis(6,5)*basis(6,3).dag()
        H44 = basis(6,4)*basis(6,4).dag()
        H45 = basis(6,5)*basis(6,4).dag()
        H55 = basis(6,5)*basis(6,5).dag()
        
        # full constructed hamiltonian, not including counter rotating terms, or F states
        if not self.counter_rot and not self.F_states:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t]]
        
        # full hamiltonian, including counter rotating terms, not including F states
        if self.counter_rot and not self.F_states:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t],
                      [H00,H00_tc],[H01,H01_tc],[H01.dag(),H01d_tc],[H02,H02_tc],[H02.dag(),H02d_tc],
                      [H11,H11_tc],[H12,H12_tc],[H12.dag(),H12d_tc],[H13,H13_tc],[H13.dag(),H13d_tc],
                      [H22,H22_tc],[H23,H23_tc],[H23.dag(),H23d_tc],[H24,H24_tc],[H24.dag(),H24d_tc],
                      [H33,H33_tc],[H34,H34_tc],[H34.dag(),H34d_tc],[H35,H35_tc],[H35.dag(),H35d_tc],
                      [H44,H44_tc],[H45,H45_tc],[H45.dag(),H45d_tc],
                      [H55,H55_tc]]
            
        # full hamiltonian, including F states, not including counter rotating terms
        if self.F_states and not self.counter_rot:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t],
                      [H00,H00_tf],[H01,H01_tf],[H01.dag(),H01d_tf],[H02,H02_tf],[H02.dag(),H02d_tf],
                      [H11,H11_tf],[H12,H12_tf],[H12.dag(),H12d_tf],[H13,H13_tf],[H13.dag(),H13d_tf],
                      [H22,H22_tf],[H23,H23_tf],[H23.dag(),H23d_tf],[H24,H24_tf],[H24.dag(),H24d_tf],
                      [H33,H33_tf],[H34,H34_tf],[H34.dag(),H34d_tf],[H35,H35_tf],[H35.dag(),H35d_tf],
                      [H44,H44_tf],[H45,H45_tf],[H45.dag(),H45d_tf],
                      [H55,H55_tf]]
            
        # full hamiltonian, including F states, including counter rotating terms
        if self.counter_rot == True:
            self.H = [[H00,H00_t],[H01,H01_t],[H01.dag(),H01d_t],[H02,H02_t],[H02.dag(),H02d_t],
                      [H11,H11_t],[H12,H12_t],[H12.dag(),H12d_t],[H13,H13_t],[H13.dag(),H13d_t],
                      [H22,H22_t],[H23,H23_t],[H23.dag(),H23d_t],[H24,H24_t],[H24.dag(),H24d_t],
                      [H33,H33_t],[H34,H34_t],[H34.dag(),H34d_t],[H35,H35_t],[H35.dag(),H35d_t],
                      [H44,H44_t],[H45,H45_t],[H45.dag(),H45d_t],
                      [H55,H55_t],
                      [H00,H00_tf],[H01,H01_tf],[H01.dag(),H01d_tf],[H02,H02_tf],[H02.dag(),H02d_tf],
                      [H11,H11_tf],[H12,H12_tf],[H12.dag(),H12d_tf],[H13,H13_tf],[H13.dag(),H13d_tf],
                      [H22,H22_tf],[H23,H23_tf],[H23.dag(),H23d_tf],[H24,H24_tf],[H24.dag(),H24d_tf],
                      [H33,H33_tf],[H34,H34_tf],[H34.dag(),H34d_tf],[H35,H35_tf],[H35.dag(),H35d_tf],
                      [H44,H44_tf],[H45,H45_tf],[H45.dag(),H45d_tf],
                      [H55,H55_tf],
                      [H00,H00_tc],[H01,H01_tc],[H01.dag(),H01d_tc],[H02,H02_tc],[H02.dag(),H02d_tc],
                      [H11,H11_tc],[H12,H12_tc],[H12.dag(),H12d_tc],[H13,H13_tc],[H13.dag(),H13d_tc],
                      [H22,H22_tc],[H23,H23_tc],[H23.dag(),H23d_tc],[H24,H24_tc],[H24.dag(),H24d_tc],
                      [H33,H33_tc],[H34,H34_tc],[H34.dag(),H34d_tc],[H35,H35_tc],[H35.dag(),H35d_tc],
                      [H44,H44_tc],[H45,H45_tc],[H45.dag(),H45d_tc],
                      [H55,H55_tc]]
            
        # arguments that appear in the time dependant components of the hamiltonian
        # ω: raman beam detuning, ω0: zeeman splitting frequency
        args = {'ω':self.ω, 'ω0':self.ω0}
                      
        # solve for bank of state vectors at each point specified in t, driven by the hamiltonian above
        self.ψ = sesolve(self.H,self.ψ, self.t, args = args, options=Options(nsteps=1e9)).states

        return    
    
    # simulates rabi spectroscopy of transition, scanning ramam beam detuning at fixed interrogation times
    # w0 and wf starting and stopping points in frequency scans
    # t_int = innterrogation time for each point. constant over scan
    def spectroscopy(self, w0, wf, t_int):
                      
        # bank of raman beam frequencies to scan over
        self.ws = np.linspace(w0, wf, self.nres)   # self.nres = number of points in spectroscopy scan
                      
        # bank of bright probabilties corresponding to self.ws in frequency scan
        self.y = np.array(())
                      
        t = np.linspace(0,t_int,2)   # time to simulate for spectroscopy scan
                      
        with tqdm(total= self.nres, desc="spectroscopy, Pb = " + str(round(self.Pb*1e3,2)) + ' mW') as pbar_spc:
            for w in self.ws:
                pbar_spc.update(1)
                self.ω = w   # update raman beam detuning
                self.simulate(t)
                self.get_populations()   # update state populations
                      
                if self.transition == '+5/2<->-1/2' and self.ls_854 == 0:   # conditions to fit to -1/2
                    self.y = np.append(self.y, self.m12[-1])
                      
                if self.transition == '+5/2<->-3/2':   # condition to fit to -3/2
                    self.y = np.append(self.y, self.m32[-1])
                      
                if self.ls_854 != 0:   # condition to fit to population not in +5/2 (mainly +3/2)
                    self.y = np.append(self.y, 1-self.p52[-1])
        
        # fit sinc(w) to simulated spectroscopy data to find numerically simulated resonance
        h = max(self.y)   # initial guess of height of sinc(w)
        x0 = self.ws[np.abs(self.y - h).argmin()] + 1e-6   # initial guess of resonance (offset from perfect to avoid errors), index of highest y
        hlfm = np.abs(self.ws[np.abs(self.y - h/2).argmin()] - x0)   # initial guess of half max of sinc(w)
        w0 = 1.5/hlfm   # convert half max to parameter used in fit
    
        # initial parameters
        p0 = [h,x0,w0]
        self.p, self.fit_res, self.err = useful_funcs.fitSinc(self.ws, self.y, p0)
        self.w_num = self.p[1]    
        
        return
                      
    # function that simulates resonant dynamics at different beam powers, to measure the rabi frequency as function of power
    # utility:                                   
    #   - calls analytics() to get analytic prediction of rabi frequency as resonances used for initial guesses
    #   - at each power set point, runs spectroscopy() to find resonance
    #   - drives the transtion resonantly out to 4x the pi time to fit to numerical rabi frequency
    def power_scaling(self, Pb_bank):
        self.m = len(Pb_bank)
        self.analytics(Pb_bank)
        
        self.w_num_bank = np.array(())   # fit resonance at each power set point
        self.δw_num_bank = np.array(())
        
        self.Ω_num_bank = np.array(())   # fit rabi frequency at each power set point
        self.δΩ_num_bank = np.array(())
        
        self.p52_bank = []
        self.p32_bank = []
        self.p12_bank = []
        self.m12_bank = []
        self.m32_bank = []
        self.m52_bank = []
        self.t_bank = []
        self.w_bank = []
        self.y_bank = []
        self.res_fits = []
        self.flop_fits = []
        
        # record the maximum population in intermediate states during resonant transitions
        self.p32_max = np.array(())
        self.p12_max = np.array(())
        self.m32_max = np.array(())
        self.m52_max = np.array(())
        
        self.nres = 40   # number of scan points in spectroscopy scan
        
        with tqdm(total= len(Pb_bank), desc="power scan progress") as pbar_pwr:
            for i in range(len(Pb_bank)):

                self.Pb = Pb_bank[i]
                rng = (2**5)*self.Ω_pred[i]/6
                
                # simulate rabi spectroscopy to find the resonance
                self.spectroscopy(self.ω_pred[i] - rng, self.ω_pred[i] + rng,.8*np.pi/self.Ω_pred[i])
                self.ω = self.w_num   # set raman beam detuning ω to numerically simulated resonance
                self.w_num_bank = np.append(self.w_num_bank, self.w_num)   # add numerically simulated resonance to bank
                self.δw_num_bank = np.append(self.δw_num_bank, self.err[1])
                self.t = np.linspace(0, np.abs(4*np.pi/self.Ω_pred[i]), 1000)   # time array used to simulate resonance dynamics
                self.simulate(self.t)
                self.get_populations()
                
                # initial parameters to fit rabi frequency
                p0 = [(np.abs(np.pi/self.Ω_pred[i])), 0,1]
                bnds = [(10e-9,-.1,.6),(100000e-6,.1,1)]

                # which populations to fit to based on transition we are driving
                if self.transition == '+5/2<->-1/2':
                    p,fit,error = useful_funcs.fitSinFull(self.t,self.m12,p0,bnds)
                
                if self.transition == '+5/2<->-3/2':
                    p,fit,error = useful_funcs.fitSinFull(self.t,self.m32,p0,bnds)

                if self.ls_854 != 0:
                    # in experiment, two-photon fits to bright population, all states except +5/2
                    p,fit,error = useful_funcs.fitSinFull(self.t,1 - self.p52,p0,bnds)
                    
                self.p52_bank.append(self.p52)
                self.p32_bank.append(self.p32)
                self.p12_bank.append(self.p12)
                self.m12_bank.append(self.m12)
                self.m32_bank.append(self.m32)
                self.m52_bank.append(self.m52)
                self.t_bank.append(self.t)
                self.w_bank.append(self.ws)
                self.y_bank.append(self.y)
                self.res_fits.append(self.fit_res)
                self.flop_fits.append(fit)
                self.Ω_num_bank = np.append(self.Ω_num_bank, np.pi/p[0])   # fit Ω converted from fit pi time
                self.δΩ_num_bank = np.append(self.δΩ_num_bank, np.pi*error[0]/(p[0]**2))
                self.p32_max = np.append(self.p32_max, max(self.p32))
                self.p12_max = np.append(self.p12_max, max(self.p12))
                self.m32_max = np.append(self.m32_max, max(self.m32))
                self.m52_max = np.append(self.m52_max, max(self.m52))
                
                pbar_pwr.update(1)
            
        return                      
    
    # function to perform ramped pi pulses, with sin^2 and sin^4 ramps on rpi intensity
    # for a fixed power set point, scans over relative lengths of ramp time, specified f_ps_bank
    # tpi is pi time at that power assuming no ramping of intensity
    def power_scaling_PS(self, f_ps_bank, tpi, scan = False):
        self.p52_bank = []
        self.p32_bank = []
        self.p12_bank = []
        self.m12_bank = []
        self.m32_bank = []
        self.m52_bank = []
        self.t_bank = []
        self.env_bank = []
        self.w_bank = []
        self.y_bank = []
        self.res_fits = []
        self.fid_bank = []
        self.t_final_bank = []
        
        self.nres = 40   # number of scan points in spectroscopy scan
    
        with tqdm(total= len(f_ps_bank), desc="PS scan progress") as pbar_pwr:
            for i in range(len(f_ps_bank)):

                rng = (2**5)*self.Ω_pred[0]/6
                
                # simulate rabi spectroscopy to find the resonance
                self.spectroscopy(self.ω_pred[0] - rng, self.ω_pred[0] + rng,.8*np.pi/self.Ω_pred[0])
                self.ω = self.w_num   # set raman beam detuning ω to numerically simulated resonance
                self.w_num_bank = np.append(self.w_num_bank, self.w_num)   # add numerically simulated resonance to bank
                self.δw_num_bank = np.append(self.δw_num_bank, self.err[1])
                
                ϵ = self.ϵ
                
                self.simulate_PS(tpi, f_ps_bank[i], ϵ)
                self.get_populations()
                   
                self.p52_bank.append(self.p52)
                self.p32_bank.append(self.p32)
                self.p12_bank.append(self.p12)
                self.m12_bank.append(self.m12)
                self.m32_bank.append(self.m32)
                self.m52_bank.append(self.m52)
                self.t_bank.append(self.t)
                self.env_bank.append(self.env)
                self.w_bank.append(self.ws)
                self.y_bank.append(self.y)
                self.res_fits.append(self.fit_res)
                
                if scan == True:
                    self.ϵ_bank = np.linspace(-.02, .01, 81)

                    fid_bank = np.array(())
                    t_final_bank = np.array(())

                    for ϵ in self.ϵ_bank:
                        self.simulate_PS(tpi, f_ps_bank[i], ϵ)
                        self.get_populations()
                        fid_bank = np.append(fid_bank, self.m12[-1])
                        t_final_bank = np.append(t_final_bank, self.t[-1])

                    self.fid_bank.append(fid_bank)
                    self.t_final_bank.append(t_final_bank)
                
                pbar_pwr.update(1)
            
        return       
    
    # get analytics for higher order rabi frequencies and higher order stark shifts
    def analytics(self, Pb_bank, sim_resonance = None):
        # sim resonance is optional array of raman beam detuning values to include into the analytics at each power set point
        
        # arrays of each stark shift at each value of Pb_bank
        # δij_bank is the j'th order stark shift on the i'th qudit state
        self.δ02_bank = np.array(())
        self.δ04_bank = np.array(())
        self.δ06_bank = np.array(())
        self.δ08_bank = np.array(())
        
        self.δ12_bank = np.array(())
        self.δ14_bank = np.array(())
        self.δ16_bank = np.array(())
        self.δ18_bank = np.array(())
        
        self.δ22_bank = np.array(())
        self.δ24_bank = np.array(())
        self.δ26_bank = np.array(())
        self.δ28_bank = np.array(())
        
        self.δ32_bank = np.array(())
        self.δ34_bank = np.array(())
        self.δ36_bank = np.array(())
        self.δ38_bank = np.array(())
        
        self.δ42_bank = np.array(())
        self.δ44_bank = np.array(())
        self.δ46_bank = np.array(())
        self.δ48_bank = np.array(())
        
        self.δ52_bank = np.array(())
        self.δ54_bank = np.array(())
        self.δ56_bank = np.array(())
        self.δ58_bank = np.array(())
        
        self.ω01_bank = np.array(())
        self.ω12_bank = np.array(())
        self.ω23_bank = np.array(())
        self.ω34_bank = np.array(())
        self.ω45_bank = np.array(())
        
        # arrays of higher order rabi frequencies at each value of Pb_bank
        # Ωij_k is the k'th order rabi frequency between qudit states i and j
        self.Ω01_2 = np.array(())
        self.Ω03_4 = np.array(())
        self.Ω03_6 = np.array(())
        self.Ω04_6 = np.array(())
        
        # analytic bounds on population in intermediate states for four-photon transition
        self.p32_bound = np.array(())
        self.p12_bound = np.array(())
        
        self.Ω_pred = np.array(())
        self.ω_pred = np.array(())
        
        for Pb in Pb_bank:
            self.Pb = Pb   # update self.Pb out of Pb_bank array
            self.get_rabi_frequencies()   # update beam powers
            # beam detuning is used in higher order (O>2) stark shift calculations. 
            # can plug in here what beam detuning is based on numerics or experiment
            # predicting the beam detuning just from analytics requires an iterative approach
            # here, we just use second order stark shifts for simplification later on
            if sim_resonance is not None: 
                ind = np.where(Pb_bank == Pb)[0][0]
                self.ω = sim_resonance[ind]
                
            # at zeroth order, lightshift on state i δi is zero   
            δ0 = 0   
            δ1 = 0
            δ2 = 0
            δ3 = 0
            δ4 = 0
            δ5 = 0
                   
            if self.ls_order >= 2:
                # add in second order stark shifts due to F states by hand. 
                # F state single beam rabi frequencies are all set to zero if self.F_states = False in get_rabi_frequencies()
                δ02 = np.sum(self.f02(*self.nums)) + np.sum(self.f02(*self.nums_ls)) +(self.Ω010b**2 + self.Ω010r**2 + self.Ω011b**2 + self.Ω011r**2 + self.Ω012b**2 + self.Ω012r**2)/(4*self.ΔF)
                δ12 = np.sum(self.f12(*self.nums)) + np.sum(self.f12(*self.nums_ls)) +(self.Ω111b**2 + self.Ω111r**2 + self.Ω112b**2 + self.Ω112r**2 + self.Ω113b**2 + self.Ω113r**2)/(4*self.ΔF)
                δ22 = np.sum(self.f22(*self.nums)) + np.sum(self.f22(*self.nums_ls)) +(self.Ω212b**2 + self.Ω212r**2 + self.Ω213b**2 + self.Ω213r**2 + self.Ω214b**2 + self.Ω214r**2)/(4*self.ΔF)
                δ32 = np.sum(self.f32(*self.nums)) + np.sum(self.f32(*self.nums_ls)) +(self.Ω313b**2 + self.Ω313r**2 + self.Ω314b**2 + self.Ω314r**2 + self.Ω315b**2 + self.Ω315r**2)/(4*self.ΔF)
                δ42 = np.sum(self.f42(*self.nums)) + np.sum(self.f42(*self.nums_ls)) +(self.Ω414b**2 + self.Ω414r**2 + self.Ω415b**2 + self.Ω415r**2 + self.Ω416b**2 + self.Ω416r**2)/(4*self.ΔF)
                δ52 = np.sum(self.f52(*self.nums)) + np.sum(self.f52(*self.nums_ls)) +(self.Ω515b**2 + self.Ω515r**2 + self.Ω516b**2 + self.Ω516r**2 + self.Ω517b**2 + self.Ω517r**2)/(4*self.ΔF)

                # update second order stark shifts on each qudit state arrays at each power set point
                self.δ02_bank = np.append(self.δ02_bank, δ02) 
                self.δ12_bank = np.append(self.δ12_bank, δ12) 
                self.δ22_bank = np.append(self.δ22_bank, δ22) 
                self.δ32_bank = np.append(self.δ32_bank, δ32) 
                self.δ42_bank = np.append(self.δ42_bank, δ42) 
                self.δ52_bank = np.append(self.δ52_bank, δ52) 
                
                # ls on state i δi is just due to second order shifts δi2
                δ0 += δ02   
                δ1 += δ12
                δ2 += δ22
                δ3 += δ32
                δ4 += δ42
                δ5 += δ52
                
                # bare unperturbed energies must be used for calculation of higher order stark shifts
                self.ω01 = self.ω0
                self.ω12 = self.ω0
                self.ω23 = self.ω0
                self.ω34 = self.ω0
                self.ω45 = self.ω0
                
                self.get_rabi_frequencies()   # update splittings to are energy splittings
                
                # if numerical beam detuning not provided, assume resonance according to second order stark shifts
                # if simulation has been carried out with fixed beam detuning prior, will use that number here instead
                # if sim_resonance is None:
                #     if self.transition == '+5/2<->-1/2':
                #         self.ω = (3*self.ω0 + δ0 - δ3)/2
                #     if self.transition == '+5/2<->-3/2':
                #         self.ω = (4*self.ω0 + δ0 - δ4)/3

                if self.ls_order >= 4:  # add in fourth order stark shifts δi4 on state i
                    with np.errstate(divide='ignore', invalid='ignore'):  
                        δ04 = np.nansum(self.f04(*self.nums)) + np.nansum(self.f04(*self.nums_ls))
                        δ14 = np.nansum(self.f14(*self.nums)) + np.nansum(self.f14(*self.nums_ls))
                        δ24 = np.nansum(self.f24(*self.nums)) + np.nansum(self.f24(*self.nums_ls))
                        δ34 = np.nansum(self.f34(*self.nums)) + np.nansum(self.f34(*self.nums_ls))
                        δ44 = np.nansum(self.f44(*self.nums)) + np.nansum(self.f44(*self.nums_ls))
                        δ54 = np.nansum(self.f54(*self.nums)) + np.nansum(self.f54(*self.nums_ls))
                    
                    # update fourth order stark shifts on each qudit state arrays at each power set point
                    self.δ04_bank = np.append(self.δ04_bank, δ04) 
                    self.δ14_bank = np.append(self.δ14_bank, δ14) 
                    self.δ24_bank = np.append(self.δ24_bank, δ24) 
                    self.δ34_bank = np.append(self.δ34_bank, δ34) 
                    self.δ44_bank = np.append(self.δ44_bank, δ44) 
                    self.δ54_bank = np.append(self.δ54_bank, δ54) 

                    δ0 += δ04
                    δ1 += δ14
                    δ2 += δ24
                    δ3 += δ34
                    δ4 += δ44
                    δ5 += δ54

                    if self.ls_order >= 6:   # add in sixth order shifts
                        with np.errstate(divide='ignore', invalid='ignore'):  
                            δ06 = np.nansum(self.f06(*self.nums))  + np.nansum(self.f06(*self.nums_ls))
                            δ16 = np.nansum(self.f16(*self.nums))  + np.nansum(self.f16(*self.nums_ls))
                            δ26 = np.nansum(self.f26(*self.nums))  + np.nansum(self.f26(*self.nums_ls))
                            δ36 = np.nansum(self.f36(*self.nums))  + np.nansum(self.f36(*self.nums_ls))
                            δ46 = np.nansum(self.f46(*self.nums))  + np.nansum(self.f46(*self.nums_ls))
                            δ56 = np.nansum(self.f56(*self.nums))  + np.nansum(self.f56(*self.nums_ls))
                        
                        # update sixth order stark shifts on each qudit state arrays at each power set point
                        self.δ06_bank = np.append(self.δ06_bank, δ06) 
                        self.δ16_bank = np.append(self.δ16_bank, δ16) 
                        self.δ26_bank = np.append(self.δ26_bank, δ26) 
                        self.δ36_bank = np.append(self.δ36_bank, δ36) 
                        self.δ46_bank = np.append(self.δ46_bank, δ46) 
                        self.δ56_bank = np.append(self.δ56_bank, δ56) 

                        δ0 += δ06
                        δ1 += δ16
                        δ2 += δ26
                        δ3 += δ36
                        δ4 += δ46
                        δ5 += δ56

                        if self.ls_order >= 8:   # add in eighth order shifts
                            with np.errstate(divide='ignore', invalid='ignore'):  
                                δ08 = np.nansum(self.f08(*self.nums))  + np.nansum(self.f08(*self.nums_ls))
                                δ18 = np.nansum(self.f18(*self.nums))  + np.nansum(self.f18(*self.nums_ls))
                                δ28 = np.nansum(self.f28(*self.nums))  + np.nansum(self.f28(*self.nums_ls))
                                δ38 = np.nansum(self.f38(*self.nums))  + np.nansum(self.f38(*self.nums_ls))
                                δ48 = np.nansum(self.f48(*self.nums))  + np.nansum(self.f48(*self.nums_ls))
                                δ58 = np.nansum(self.f58(*self.nums))  + np.nansum(self.f58(*self.nums_ls))
                            
                            # update eighth order stark shifts on each qudit state arrays at each power set point
                            self.δ08_bank = np.append(self.δ08_bank, δ08) 
                            self.δ18_bank = np.append(self.δ18_bank, δ18) 
                            self.δ28_bank = np.append(self.δ28_bank, δ28) 
                            self.δ38_bank = np.append(self.δ38_bank, δ38) 
                            self.δ48_bank = np.append(self.δ48_bank, δ48) 
                            self.δ58_bank = np.append(self.δ58_bank, δ58) 
                        
                            δ0 += δ08
                            δ1 += δ18
                            δ2 += δ28
                            δ3 += δ38
                            δ4 += δ48
                            δ5 += δ58
            
            # state splittings including lightshifts
            w01 = self.ω0 + δ0 - δ1
            w12 = self.ω0 + δ1 - δ2
            w23 = self.ω0 + δ2 - δ3
            w34 = self.ω0 + δ3 - δ4
            w45 = self.ω0 + δ4 - δ5
                
            # include ls from 854 if on. integer prefactors determined by clebsch-gordan coefficients
            self.ω01 = w01
            self.ω12 = w12 + 1*self.ls_854
            self.ω23 = w23 + 2*self.ls_854
            self.ω34 = w34 + 3*self.ls_854
            self.ω45 = w45 + 4*self.ls_854    
                
            # update numbers for each sympy variable
            self.get_rabi_frequencies()

            # rabi frequency analytics include stark shifts to intermediate state detunings
            if self.transition == '+5/2<->-1/2':
                Ω03_4 = np.sum(self.f03_4(*self.nums))
                Ω03_6 = np.sum(self.f03_6(*self.nums))
                self.Ω03_4 = np.append(self.Ω03_4, Ω03_4)
                self.Ω03_6 = np.append(self.Ω03_6, Ω03_6)
            if self.transition == '+5/2<->-3/2':    
                Ω04_6 = np.sum(self.f04_6(*self.nums))
                self.Ω04_6 = np.append(self.Ω04_6, Ω04_6)
            if self.ls_854 != 0:
                Ω01_2 = self.Ω06r*self.Ω16b/(2*self.Δ)   # simple single term two-photon Rabi frequency
                self.Ω01_2 = np.append(self.Ω01_2, abs(Ω01_2))
            
            # calculate predicted resonance from analytics
            if self.transition == '+5/2<->-1/2':
                wr = (3*self.ω0 + δ0 - δ3)/2
            if self.transition == '+5/2<->-3/2':
                wr = (4*self.ω0 + δ0 - δ4)/3
            if self.ls_854 != 0:
                wr = self.ω0 + δ0 - δ1
            
            # calculate predicted rabi frequencies to guide simulation
            if self.transition == '+5/2<->-1/2' and self.ls_854 == 0:
                self.Ω_pred = np.append(self.Ω_pred, np.abs(Ω03_4))
            if self.transition == '+5/2<->-3/2':
                self.Ω_pred = np.append(self.Ω_pred, np.abs(Ω04_6))
            if self.ls_854 != 0:
                self.Ω_pred = np.append(self.Ω_pred, np.abs(Ω01_2))
            
            self.ω01_bank = np.append(self.ω01_bank, self.ω01)
            self.ω12_bank = np.append(self.ω12_bank, self.ω12)
            self.ω23_bank = np.append(self.ω23_bank, self.ω23)
            self.ω34_bank = np.append(self.ω34_bank, self.ω34)
            self.ω45_bank = np.append(self.ω45_bank, self.ω45)
            
            # calculate predicted resonance to guide simulation
            self.ω_pred_full = {
                '+5/2<->+3/2': self.ω01,
                '+5/2<->+1/2': (self.ω01+self.ω12),
                '+3/2<->+1/2': self.ω12,
                '+3/2<->-1/2': (self.ω12+self.ω23),
                '+1/2<->-1/2': self.ω23,
                '+1/2<->-3/2': (self.ω23+self.ω34),
                '-1/2<->-3/2': self.ω34,
                '-1/2<->-5/2': (self.ω34+self.ω45),
                '-3/2<->-5/2': self.ω45,
                '+5/2<->-1/2': (self.ω01+self.ω12+self.ω23)/2,
                '+3/2<->-3/2': (self.ω12+self.ω23+self.ω34)/2,
                '+1/2<->-5/2': (self.ω23+self.ω34+self.ω45)/2,
                '+5/2<->-3/2': (self.ω01+self.ω12+self.ω23+self.ω34)/3,
                '+3/2<->-5/2': (self.ω12+self.ω23+self.ω34+self.ω45)/3,
                '+5/2<->-5/2': (self.ω01+self.ω12+self.ω23+self.ω34+self.ω45)/3
            }

            wr = self.ω_pred_full[self.transition]
            # calculate bounds on maximum population in intermediate states (only works for four-photon transition)
            # p32_bound = ((self.Ω06r*self.Ω16b/(2*self.Δ))**2)/(((self.Ω06r*self.Ω16b/(2*self.Δ))**2) + ((self.Ω17r*self.Ω37b/(2*self.Δ))**2) + (wr - w01)**2)
            # p12_bound = ((self.Ω06r*self.Ω26b/(2*self.Δ))**2)/(((self.Ω06r*self.Ω26b/(2*self.Δ))**2) + ((self.Ω28r*self.Ω38b/(2*self.Δ))**2) + (w01 + w01 - wr)**2)
            
            # self.p32_bound = np.append(self.p32_bound, p32_bound)
            # self.p12_bound = np.append(self.p12_bound, p12_bound)
            

            
        return
    
    # function that calculates rabi frequency and resonance of every transition that provides
    # all-to-all connectivity
    def full_analytics(self): 
        transition = self.transition

        self.transition = '+5/2<->-1/2'
        
        # calculate lightshifts at order = 2 (beam detuning is irrelevant at this order)
        self.ls_order = 2
        self.analytics([self.Pb])
        print(self.ω01)
        self.transition = transition
        
        self.ω_pred_2nd = {'+5/2<->+3/2':self.ω01,
                           '+5/2<->+1/2':(self.ω01+self.ω12),
                           '+3/2<->+1/2':self.ω12,
                           '+3/2<->-1/2':(self.ω12+self.ω23),
                           '+1/2<->-1/2':self.ω23,
                           '+1/2<->-3/2':(self.ω23+self.ω34),
                           '-1/2<->-3/2':self.ω34,
                           '-1/2<->-5/2':(self.ω34+self.ω45),
                           '-3/2<->-5/2':self.ω45,
                           '+5/2<->-1/2':(self.ω01+self.ω12+self.ω23)/2,
                           '+3/2<->-3/2':(self.ω12+self.ω23+self.ω34)/2,
                           '+1/2<->-5/2':(self.ω23+self.ω34+self.ω45)/2,
                           '+5/2<->-3/2':(self.ω01+self.ω12+self.ω23+self.ω34)/3,
                           '+3/2<->-5/2':(self.ω12+self.ω23+self.ω34+self.ω45)/3,
                           '+5/2<->-5/2':(self.ω01+self.ω12+self.ω23+self.ω34+self.ω45)/3}
        
        self.rabi_functions = {'+5/2<->+3/2':self.f01_2,
                               '+5/2<->+1/2':self.f02_2,
                               '+3/2<->+1/2':self.f12_2,
                               '+3/2<->-1/2':self.f13_2,
                               '+1/2<->-1/2':self.f23_2,
                               '+1/2<->-3/2':self.f24_2,
                               '-1/2<->-3/2':self.f34_2,
                               '-1/2<->-5/2':self.f35_2,
                               '-3/2<->-5/2':self.f45_2,
                               '+5/2<->-1/2':self.f03_4,
                               '+3/2<->-3/2':self.f14_4,
                               '+1/2<->-5/2':self.f25_4,
                               '+5/2<->-3/2':self.f04_6,
                               '+3/2<->-5/2':self.f15_6,
                               '+5/2<->-5/2':self.f05_6}
        self.rabi_freqs = {}
        self.rabi_terms = {}
        
        # for each transition, set beam detuning to prediction from second order stark shifts
        # calculate rabi frequency of that transition
        # calculate lightshifts up to eighth order (with beam detuning set to predicted resonance)
        # update predicted resonance (used for simulations)
        
        
        
        
        i = 0
        for key,value in self.ω_pred_2nd.items():
            self.ω = value
            
            self.ls_order = 2
            self.analytics([self.Pb])
            
            # print(key, np.sum(self.f14_4(*self.nums)))
            self.get_rabi_frequencies() # update nums with correct value of ω
            i+=1
            # if i == 1:
            #     print(self.nums)
            with np.errstate(divide='ignore', invalid='ignore'):    
                Ω = abs(np.sum(self.rabi_functions[key](*self.nums)))
            self.rabi_freqs[key] = Ω/(2*np.pi*1e3)
            with np.errstate(divide='ignore', invalid='ignore'):
                terms = self.rabi_functions[key](*self.nums)

            self.rabi_terms[key] = np.array(terms)/(2*np.pi*1e3)

            if self.pred_eighth_order == True:

                self.ls_order = 8
                self.analytics([self.Pb])
                
                self.ω_pred_8th = {'+5/2<->+3/2':self.ω01,
                                '+5/2<->+1/2':(self.ω01+self.ω12),
                                '+3/2<->+1/2':self.ω12,
                                '+3/2<->-1/2':(self.ω12+self.ω23),
                                '+1/2<->-1/2':self.ω23,
                                '+1/2<->-3/2':(self.ω23+self.ω34),
                                '-1/2<->-3/2':self.ω34,
                                '-1/2<->-5/2':(self.ω34+self.ω45),
                                '-3/2<->-5/2':self.ω45,
                                '+5/2<->-1/2':(self.ω01+self.ω12+self.ω23)/2,
                                '+3/2<->-3/2':(self.ω12+self.ω23+self.ω34)/2,
                                '+1/2<->-5/2':(self.ω23+self.ω34+self.ω45)/2,
                                '+5/2<->-3/2':(self.ω01+self.ω12+self.ω23+self.ω34)/3,
                                '+3/2<->-5/2':(self.ω12+self.ω23+self.ω34+self.ω45)/3,
                                '+5/2<->-5/2':(self.ω01+self.ω12+self.ω23+self.ω34+self.ω45)/3}
                
                self.ω_pred_full[key] = self.ω_pred_8th[key]
            
        return
        

    
    # # function that calculates rabi frequency and resonance of every transition that provides
    # # all-to-all connectivity
    # def full_analytics(self): 
    #     transition = self.transition

    #     self.transition = '+5/2<->-1/2'
    #     self.ls_order = 2
    #     self.analytics([self.Pb])
    #     self.transition = transition
        
    #     self.ω_pred_full = {'+5/2<->+3/2':self.ω01,
    #                        '+5/2<->+1/2':(self.ω01+self.ω12),
    #                        '+3/2<->+1/2':self.ω12,
    #                        '+3/2<->-1/2':(self.ω12+self.ω23),
    #                        '+1/2<->-1/2':self.ω23,
    #                        '+1/2<->-3/2':(self.ω23+self.ω34),
    #                        '-1/2<->-3/2':self.ω34,
    #                        '-1/2<->-5/2':(self.ω34+self.ω45),
    #                        '-3/2<->-5/2':self.ω45,
    #                        '+5/2<->-1/2':(self.ω01+self.ω12+self.ω23)/2,
    #                        '+3/2<->-3/2':(self.ω12+self.ω23+self.ω34)/2,
    #                        '+1/2<->-5/2':(self.ω23+self.ω34+self.ω45)/2,
    #                        '+5/2<->-3/2':(self.ω01+self.ω12+self.ω23+self.ω34)/3,
    #                        '+3/2<->-5/2':(self.ω12+self.ω23+self.ω34+self.ω45)/3,
    #                        '+5/2<->-5/2':(self.ω01+self.ω12+self.ω23+self.ω34+self.ω45)/3}
        
    #     self.rabi_functions = {'+5/2<->+3/2':self.f01_2,
    #                            '+5/2<->+1/2':self.f02_2,
    #                            '+3/2<->+1/2':self.f12_2,
    #                            '+3/2<->-1/2':self.f13_2,
    #                            '+1/2<->-1/2':self.f23_2,
    #                            '+1/2<->-3/2':self.f24_2,
    #                            '-1/2<->-3/2':self.f34_2,
    #                            '-1/2<->-5/2':self.f35_2,
    #                            '-3/2<->-5/2':self.f45_2,
    #                            '+5/2<->-1/2':self.f03_4,
    #                            '+3/2<->-3/2':self.f14_4,
    #                            '+1/2<->-5/2':self.f25_4,
    #                            '+5/2<->-3/2':self.f04_6,
    #                            '+3/2<->-5/2':self.f15_6,
    #                            '+5/2<->-5/2':self.f05_6}
        
    #     self.rabi_freqs = {}
    #     self.rabi_terms = {}
        
        
        
    #     # calculate 8th order stark shift contributions to the resonance
    #     self.ls_order = 8

    #     for key,value in self.ω_pred_full.items():
    #         self.ω = value
    #         # self.get_rabi_frequencies()
    #         self.analytics([self.Pb])
    #         Ω = abs(np.sum(self.rabi_functions[key](*self.nums)))
    #         self.rabi_freqs[key] = Ω/(2*np.pi*1e3)
    #         terms = self.rabi_functions[key](*self.nums)
    #         self.rabi_terms[key] = np.array(terms)/(2*np.pi*1e3)
    #         self.ω_temp = {'+5/2<->+3/2':self.ω01,
    #                        '+5/2<->+1/2':(self.ω01+self.ω12),
    #                        '+3/2<->+1/2':self.ω12,
    #                        '+3/2<->-1/2':(self.ω12+self.ω23),
    #                        '+1/2<->-1/2':self.ω23,
    #                        '+1/2<->-3/2':(self.ω23+self.ω34),
    #                        '-1/2<->-3/2':self.ω34,
    #                        '-1/2<->-5/2':(self.ω34+self.ω45),
    #                        '-3/2<->-5/2':self.ω45,
    #                        '+5/2<->-1/2':(self.ω01+self.ω12+self.ω23)/2,
    #                        '+3/2<->-3/2':(self.ω12+self.ω23+self.ω34)/2,
    #                        '+1/2<->-5/2':(self.ω23+self.ω34+self.ω45)/2,
    #                        '+5/2<->-3/2':(self.ω01+self.ω12+self.ω23+self.ω34)/3,
    #                        '+3/2<->-5/2':(self.ω12+self.ω23+self.ω34+self.ω45)/3,
    #                        '+5/2<->-5/2':(self.ω01+self.ω12+self.ω23+self.ω34+self.ω45)/3}
            
    #         self.ω_pred_full[key] = self.ω_temp[key]
            
    #     # calculate rabi frequencies only considering second order stark shifts
    #     self.ls_order = 2

    #     for key,value in self.ω_pred_full.items():
    #         self.ω = value
    #         # self.get_rabi_frequencies()
    #         self.analytics([self.Pb])
    #         Ω = abs(np.sum(self.rabi_functions[key](*self.nums)))
    #         self.rabi_freqs[key] = Ω/(2*np.pi*1e3)
    #         terms = self.rabi_functions[key](*self.nums)
    #         self.rabi_terms[key] = np.array(terms)/(2*np.pi*1e3)

    #     # calculate 8th order stark shift contributions to the resonance
    #     self.ls_order = 8

    #     for key,value in self.ω_pred_full.items():
    #         self.ω = value
    #         # self.get_rabi_frequencies()
    #         self.analytics([self.Pb])
    #         self.ω_temp = {'+5/2<->+3/2':self.ω01,
    #                        '+5/2<->+1/2':(self.ω01+self.ω12),
    #                        '+3/2<->+1/2':self.ω12,
    #                        '+3/2<->-1/2':(self.ω12+self.ω23),
    #                        '+1/2<->-1/2':self.ω23,
    #                        '+1/2<->-3/2':(self.ω23+self.ω34),
    #                        '-1/2<->-3/2':self.ω34,
    #                        '-1/2<->-5/2':(self.ω34+self.ω45),
    #                        '-3/2<->-5/2':self.ω45,
    #                        '+5/2<->-1/2':(self.ω01+self.ω12+self.ω23)/2,
    #                        '+3/2<->-3/2':(self.ω12+self.ω23+self.ω34)/2,
    #                        '+1/2<->-5/2':(self.ω23+self.ω34+self.ω45)/2,
    #                        '+5/2<->-3/2':(self.ω01+self.ω12+self.ω23+self.ω34)/3,
    #                        '+3/2<->-5/2':(self.ω12+self.ω23+self.ω34+self.ω45)/3,
    #                        '+5/2<->-5/2':(self.ω01+self.ω12+self.ω23+self.ω34+self.ω45)/3}
    #         self.ω_pred_full[key] = self.ω_temp[key]
            
    #     return
        
    def full_simulate(self, spectroscopy_progress=None, dynamics_progress=None):
        
        self.pred_eighth_order = True
        self.full_analytics()
        self.pred_eighth_order = False
        
        self.trans_states = {'+5/2<->+3/2':[0,1],
                             '+5/2<->+1/2':[0,2],
                             '+3/2<->+1/2':[1,2],
                             '+3/2<->-1/2':[1,3],
                             '+1/2<->-1/2':[2,3],
                             '+1/2<->-3/2':[2,4],
                             '-1/2<->-3/2':[3,4],
                             '-1/2<->-5/2':[3,5],
                             '-3/2<->-5/2':[4,5],
                             '+5/2<->-1/2':[0,3],
                             '+3/2<->-3/2':[1,4],
                             '+1/2<->-5/2':[2,5],
                             '+5/2<->-3/2':[0,4],
                             '+3/2<->-5/2':[1,5],
                             '+5/2<->-5/2':[0,5]}
        
        s0 = self.trans_states[self.transition][0]
        sf = self.trans_states[self.transition][1]
        self.ω_pred = self.ω_pred_full[self.transition]
        self.Ω_pred = self.rabi_freqs[self.transition]*2*np.pi*1e3
        t_int = 0.8*np.pi/self.Ω_pred
        t = np.linspace(0,t_int,100)
        # do spectroscopy
        rng = (2**5)*self.Ω_pred/4
        self.ω_bank = np.linspace(self.ω_pred - rng, self.ω_pred + rng, 50)
        self.p52_res = np.array(())
        self.p32_res = np.array(())
        self.p12_res = np.array(())
        self.m12_res = np.array(())
        self.m32_res = np.array(())
        self.m52_res = np.array(())
        
        for i in range(len(self.ω_bank)):
            self.ω = self.ω_bank[i]
            self.get_rabi_frequencies()
            progress = StitchedProgressBar(spectroscopy_progress, i, len(self.ω_bank)) if spectroscopy_progress else None
            self.simulate(t, s0, progress)
            self.get_populations()
            self.p52_res = np.append(self.p52_res, self.p52[-1])
            self.p32_res = np.append(self.p32_res, self.p32[-1])
            self.p12_res = np.append(self.p12_res, self.p12[-1])
            self.m12_res = np.append(self.m12_res, self.m12[-1])
            self.m32_res = np.append(self.m32_res, self.m32[-1])
            self.m52_res = np.append(self.m52_res, self.m52[-1])
            
        state_dict = {0: self.p52_res, 1: self.p32_res, 2: self.p12_res, 3: self.m12_res, 4: self.m32_res, 5: self.m52_res}
        arr_to_fit = state_dict[sf]
        
        # fit sinc(w) to simulated spectroscopy data to find numerically simulated resonance
        h = max(arr_to_fit)   # initial guess of height of sinc(w)
        x0 = self.ω_bank[np.abs(arr_to_fit - h).argmin()] + 1e-6   # initial guess of resonance (offset from perfect to avoid errors), index of highest y
        hlfm = np.abs(self.ω_bank[np.abs(arr_to_fit - h/2).argmin()] - x0)   # initial guess of half max of sinc(w)
        w0 = 1.5/hlfm   # convert half max to parameter used in fit
    
        # initial parameters
        p0 = [h,x0,w0]
        self.p, self.fit_res, self.err = useful_funcs.fitSinc(self.ω_bank, arr_to_fit, p0)
        self.w_num = self.p[1]
        
        # simulate rabi oscillations
        self.ω = self.w_num
        self.get_rabi_frequencies()
        self.t = np.linspace(0,4*np.pi/self.Ω_pred,1000)

        progress = StitchedProgressBar(dynamics_progress) if dynamics_progress else None
        self.simulate(self.t, s0, progress)
        self.get_populations()
        
        # initial parameters to fit rabi frequency
        p0 = [(np.abs(np.pi/self.Ω_pred)), 0,1]
        bnds = [(10e-9,-.1,.6),(100000e-6,.1,1)]

        flop_dict = {0: self.p52, 1: self.p32, 2: self.p12, 3: self.m12, 4: self.m32, 5: self.m52}
        arr_to_fit = flop_dict[sf]
        
        self.p,self.fit,self.error = useful_funcs.fitSinFull(self.t,arr_to_fit,p0,bnds)
        
        self.analytics([self.Pb]) 
        
        self.second_order_shift = {
            '+5/2<->+3/2': self.δ02_bank[0] - self.δ12_bank[0],
            '+5/2<->+1/2': self.δ02_bank[0] - self.δ22_bank[0],
            '+3/2<->+1/2': self.δ12_bank[0] - self.δ22_bank[0],
            '+3/2<->-1/2': self.δ12_bank[0] - self.δ32_bank[0],
            '+1/2<->-1/2': self.δ22_bank[0] - self.δ32_bank[0],
            '+1/2<->-3/2': self.δ22_bank[0] - self.δ42_bank[0],
            '-1/2<->-3/2': self.δ32_bank[0] - self.δ42_bank[0],
            '-1/2<->-5/2': self.δ32_bank[0] - self.δ52_bank[0],
            '-3/2<->-5/2': self.δ42_bank[0] - self.δ52_bank[0],
            '+5/2<->-1/2': (self.δ02_bank[0] - self.δ32_bank[0])/2,
            '+3/2<->-3/2': (self.δ12_bank[0] - self.δ42_bank[0])/2,
            '+1/2<->-5/2': (self.δ22_bank[0] - self.δ52_bank[0])/2,
            '+5/2<->-3/2': (self.δ02_bank[0] - self.δ42_bank[0])/3,
            '+3/2<->-5/2': (self.δ12_bank[0] - self.δ52_bank[0])/3,
            '+5/2<->-5/2': (self.δ02_bank[0] - self.δ52_bank[0])/3
        }
        
        self.fourth_order_shift = {
            '+5/2<->+3/2': self.δ04_bank[0] - self.δ14_bank[0],
            '+5/2<->+1/2': self.δ04_bank[0] - self.δ24_bank[0],
            '+3/2<->+1/2': self.δ14_bank[0] - self.δ24_bank[0],
            '+3/2<->-1/2': self.δ14_bank[0] - self.δ34_bank[0],
            '+1/2<->-1/2': self.δ24_bank[0] - self.δ34_bank[0],
            '+1/2<->-3/2': self.δ24_bank[0] - self.δ44_bank[0],
            '-1/2<->-3/2': self.δ34_bank[0] - self.δ44_bank[0],
            '-1/2<->-5/2': self.δ34_bank[0] - self.δ54_bank[0],
            '-3/2<->-5/2': self.δ44_bank[0] - self.δ54_bank[0],
            '+5/2<->-1/2': (self.δ04_bank[0] - self.δ34_bank[0])/2,
            '+3/2<->-3/2': (self.δ14_bank[0] - self.δ44_bank[0])/2,
            '+1/2<->-5/2': (self.δ24_bank[0] - self.δ54_bank[0])/2,
            '+5/2<->-3/2': (self.δ04_bank[0] - self.δ44_bank[0])/3,
            '+3/2<->-5/2': (self.δ14_bank[0] - self.δ54_bank[0])/3,
            '+5/2<->-5/2': (self.δ04_bank[0] - self.δ54_bank[0])/3
        }
        
        self.sixth_order_shift = {
            '+5/2<->+3/2': self.δ06_bank[0] - self.δ16_bank[0],
            '+5/2<->+1/2': self.δ06_bank[0] - self.δ26_bank[0],
            '+3/2<->+1/2': self.δ16_bank[0] - self.δ26_bank[0],
            '+3/2<->-1/2': self.δ16_bank[0] - self.δ36_bank[0],
            '+1/2<->-1/2': self.δ26_bank[0] - self.δ36_bank[0],
            '+1/2<->-3/2': self.δ26_bank[0] - self.δ46_bank[0],
            '-1/2<->-3/2': self.δ36_bank[0] - self.δ46_bank[0],
            '-1/2<->-5/2': self.δ36_bank[0] - self.δ56_bank[0],
            '-3/2<->-5/2': self.δ46_bank[0] - self.δ56_bank[0],
            '+5/2<->-1/2': (self.δ06_bank[0] - self.δ36_bank[0])/2,
            '+3/2<->-3/2': (self.δ16_bank[0] - self.δ46_bank[0])/2,
            '+1/2<->-5/2': (self.δ26_bank[0] - self.δ56_bank[0])/2,
            '+5/2<->-3/2': (self.δ06_bank[0] - self.δ46_bank[0])/3,
            '+3/2<->-5/2': (self.δ16_bank[0] - self.δ56_bank[0])/3,
            '+5/2<->-5/2': (self.δ06_bank[0] - self.δ56_bank[0])/3
        }
        
        self.eighth_order_shift = {
            '+5/2<->+3/2': self.δ08_bank[0] - self.δ18_bank[0],
            '+5/2<->+1/2': self.δ08_bank[0] - self.δ28_bank[0],
            '+3/2<->+1/2': self.δ18_bank[0] - self.δ28_bank[0],
            '+3/2<->-1/2': self.δ18_bank[0] - self.δ38_bank[0],
            '+1/2<->-1/2': self.δ28_bank[0] - self.δ38_bank[0],
            '+1/2<->-3/2': self.δ28_bank[0] - self.δ48_bank[0],
            '-1/2<->-3/2': self.δ38_bank[0] - self.δ48_bank[0],
            '-1/2<->-5/2': self.δ38_bank[0] - self.δ58_bank[0],
            '-3/2<->-5/2': self.δ48_bank[0] - self.δ58_bank[0],
            '+5/2<->-1/2': (self.δ08_bank[0] - self.δ38_bank[0])/2,
            '+3/2<->-3/2': (self.δ18_bank[0] - self.δ48_bank[0])/2,
            '+1/2<->-5/2': (self.δ28_bank[0] - self.δ58_bank[0])/2,
            '+5/2<->-3/2': (self.δ08_bank[0] - self.δ48_bank[0])/3,
            '+3/2<->-5/2': (self.δ18_bank[0] - self.δ58_bank[0])/3,
            '+5/2<->-5/2': (self.δ08_bank[0] - self.δ58_bank[0])/3
        }
        
        self.δ2 = self.second_order_shift[self.transition]
        # self.δ4 = 0
        # self.δ6 = 0
        # self.δ8 = 0
        
        self.δ4 = self.fourth_order_shift[self.transition]
        self.δ6 = self.sixth_order_shift[self.transition]
        self.δ8 = self.eighth_order_shift[self.transition]
        
    # function that calculates shifts throughout D5/2 manifold up to 8th order in stark shift and optionally counter rotating terms and F state couplings
    def rsig_splittings(self, Pr_bank):
    
        self.w01 = np.array(())
        self.w12 = np.array(())
        self.w23 = np.array(())
        self.w34 = np.array(())
        self.w45 = np.array(())
        
        for Pr in Pr_bank:
            self.Pr = Pr   # update Pr value
            self.get_rabi_frequencies()   # get new single beam rabi_frequencies as new power setpoint
            self.analytics([0])   # set rpi power to zero for each point in this scan
            
            self.w01 = np.append(self.w01, self.ω01)
            self.w12 = np.append(self.w12, self.ω12)
            self.w23 = np.append(self.w23, self.ω23)
            self.w34 = np.append(self.w34, self.ω34)
            self.w45 = np.append(self.w45, self.ω45)
            
        return
                   
    # reads in numpy arrays of higher order lightshifts and rabi frequencies. Reading in higher order terms takes a while, ideally do once per simulation instanciation
    def get_f(self):   # f is a function that has an array of sympy variables saved, along with an array of sympy expressions made up of the sympy variables.
                       # feeding in an array of numbers that map one-to-one to sympy variables, f computes the value of each sympy expression saved in the stored array
            
        self.get_rabi_frequencies()   # need to save symbols as global variable first 
        
        base_path = Path(__file__).resolve().parent.parent
        path_ls = str(base_path / "stark_shifts") + "/"
        path_rf = str(base_path / "rabi_frequencies") + "/"
        
        # path_ls = "\\Users\\Gregory\\hop_calculator\\stark_shifts\\"   # path pointing to where higher order stark shift arrays are saved
        # path_rf = "\\Users\\Gregory\\hop_calculator\\rabi_frequencies\\"   # path pointing to higher order rabi frequencies

        # read in rabi frequencies:
        # fij_k() computes the kth order rabi frequencies between qudit states labelled ij
        if not self.counter_rot and not self.F_states:
            # two-photon transitions
            self.f01_2 = sp.lambdify(self.symbols,np.load(path_rf + "01_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f02_2 = sp.lambdify(self.symbols,np.load(path_rf + "02_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f12_2 = sp.lambdify(self.symbols,np.load(path_rf + "12_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f13_2 = sp.lambdify(self.symbols,np.load(path_rf + "13_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f23_2 = sp.lambdify(self.symbols,np.load(path_rf + "23_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f24_2 = sp.lambdify(self.symbols,np.load(path_rf + "24_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f34_2 = sp.lambdify(self.symbols,np.load(path_rf + "34_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f35_2 = sp.lambdify(self.symbols,np.load(path_rf + "35_2.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f45_2 = sp.lambdify(self.symbols,np.load(path_rf + "45_2.npy", allow_pickle=True).tolist(),modules='numpy') 
            
            # four-photon transitions
            self.f03_4 = sp.lambdify(self.symbols,np.load(path_rf + "03_4.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f14_4 = sp.lambdify(self.symbols,np.load(path_rf + "14_4.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f25_4 = sp.lambdify(self.symbols,np.load(path_rf + "25_4.npy", allow_pickle=True).tolist(),modules='numpy')
            
            # six-photon transitions
            self.f04_6 = sp.lambdify(self.symbols,np.load(path_rf + "04_6.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f15_6 = sp.lambdify(self.symbols,np.load(path_rf + "15_6.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f05_6 = sp.lambdify(self.symbols,np.load(path_rf + "05_6.npy", allow_pickle=True).tolist(),modules='numpy')
            
            self.f03_6 = sp.lambdify(self.symbols,np.load(path_rf + "03_6.npy", allow_pickle=True).tolist(),modules='numpy')
            
        if self.counter_rot and not self.F_states:
            self.f03_4 = sp.lambdify(self.symbols,np.load(path_rf + "03_4_c.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f03_6 = sp.lambdify(self.symbols,np.load(path_rf + "03_6_c.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f04_6 = sp.lambdify(self.symbols,np.load(path_rf + "04_6_c.npy", allow_pickle=True).tolist(),modules='numpy')
        if not self.counter_rot and self.F_states:
            self.f03_4 = sp.lambdify(self.symbols,np.load(path_rf + "03_4_f.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f03_6 = sp.lambdify(self.symbols,np.load(path_rf + "03_6_f.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f04_6 = sp.lambdify(self.symbols,np.load(path_rf + "04_6_f.npy", allow_pickle=True).tolist(),modules='numpy')
        if self.counter_rot and self.F_states:
            self.f03_4 = sp.lambdify(self.symbols,np.load(path_rf + "03_4_cf.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f03_6 = sp.lambdify(self.symbols,np.load(path_rf + "03_6_cf.npy", allow_pickle=True).tolist(),modules='numpy')
            self.f04_6 = sp.lambdify(self.symbols,np.load(path_rf + "04_6_cf.npy", allow_pickle=True).tolist(),modules='numpy')
            
        
        # read in stark shifts
        # fij() computes the jth order stark shift on the qudit state labelled i

        if not self.counter_rot:   # not including counter rotating terms
                                   # dont wast time reading in long functions youre not going to use
            if self.ls_order >= 2:   # read in second order terms           
                self.f02 = sp.lambdify(self.symbols,np.load(path_ls + "02.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f12 = sp.lambdify(self.symbols,np.load(path_ls + "12.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f22 = sp.lambdify(self.symbols,np.load(path_ls + "22.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f32 = sp.lambdify(self.symbols,np.load(path_ls + "32.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f42 = sp.lambdify(self.symbols,np.load(path_ls + "42.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f52 = sp.lambdify(self.symbols,np.load(path_ls + "52.npy", allow_pickle=True).tolist(),modules='numpy')

                if self.ls_order >= 4:   # fourth order terms   
                    self.f04 = sp.lambdify(self.symbols,np.load(path_ls + "04.npy", allow_pickle=True).tolist(),modules='numpy')
                    self.f14 = sp.lambdify(self.symbols,np.load(path_ls + "14.npy", allow_pickle=True).tolist(),modules='numpy')          
                    self.f24 = sp.lambdify(self.symbols,np.load(path_ls + "24.npy", allow_pickle=True).tolist(),modules='numpy')
                    self.f34 = sp.lambdify(self.symbols,np.load(path_ls + "34.npy", allow_pickle=True).tolist(),modules='numpy')          
                    self.f44 = sp.lambdify(self.symbols,np.load(path_ls + "44.npy", allow_pickle=True).tolist(),modules='numpy')
                    self.f54 = sp.lambdify(self.symbols,np.load(path_ls + "54.npy", allow_pickle=True).tolist(),modules='numpy')          

                    if self.ls_order >= 6:   # sixth order terms
                        self.f06 = sp.lambdify(self.symbols,np.load(path_ls + "06.npy", allow_pickle=True).tolist(),modules='numpy')
                        self.f16 = sp.lambdify(self.symbols,np.load(path_ls + "16.npy", allow_pickle=True).tolist(),modules='numpy')
                        self.f26 = sp.lambdify(self.symbols,np.load(path_ls + "26.npy", allow_pickle=True).tolist(),modules='numpy')          
                        self.f36 = sp.lambdify(self.symbols,np.load(path_ls + "36.npy", allow_pickle=True).tolist(),modules='numpy')          
                        self.f46 = sp.lambdify(self.symbols,np.load(path_ls + "46.npy", allow_pickle=True).tolist(),modules='numpy')          
                        self.f56 = sp.lambdify(self.symbols,np.load(path_ls + "56.npy", allow_pickle=True).tolist(),modules='numpy')   

                        if self.ls_order >= 8:   # eighth order terms
                            self.f08 = sp.lambdify(self.symbols,np.load(path_ls + "08.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f18 = sp.lambdify(self.symbols,np.load(path_ls + "18.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f28 = sp.lambdify(self.symbols,np.load(path_ls + "28.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f38 = sp.lambdify(self.symbols,np.load(path_ls + "38.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f48 = sp.lambdify(self.symbols,np.load(path_ls + "48.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f58 = sp.lambdify(self.symbols,np.load(path_ls + "58.npy", allow_pickle=True).tolist(),modules='numpy')

        if self.counter_rot:   # stark shifts including counter rotating terms up to fourth order

            if self.ls_order >= 2:   # read in second order terms           
                self.f02 = sp.lambdify(self.symbols,np.load(path_ls + "02_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f12 = sp.lambdify(self.symbols,np.load(path_ls + "12_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f22 = sp.lambdify(self.symbols,np.load(path_ls + "22_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f32 = sp.lambdify(self.symbols,np.load(path_ls + "32_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f42 = sp.lambdify(self.symbols,np.load(path_ls + "42_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                self.f52 = sp.lambdify(self.symbols,np.load(path_ls + "52_counter.npy", allow_pickle=True).tolist(),modules='numpy')

                if self.ls_order >= 4:   # fourth order terms   
                    self.f04 = sp.lambdify(self.symbols,np.load(path_ls + "04_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                    self.f14 = sp.lambdify(self.symbols,np.load(path_ls + "14_counter.npy", allow_pickle=True).tolist(),modules='numpy')          
                    self.f24 = sp.lambdify(self.symbols,np.load(path_ls + "24_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                    self.f34 = sp.lambdify(self.symbols,np.load(path_ls + "34_counter.npy", allow_pickle=True).tolist(),modules='numpy')          
                    self.f44 = sp.lambdify(self.symbols,np.load(path_ls + "44_counter.npy", allow_pickle=True).tolist(),modules='numpy')
                    self.f54 = sp.lambdify(self.symbols,np.load(path_ls + "54_counter.npy", allow_pickle=True).tolist(),modules='numpy')          

                    if self.ls_order >= 6:   # sixth order terms
                        self.f06 = sp.lambdify(self.symbols,np.load(path_ls + "06.npy", allow_pickle=True).tolist(),modules='numpy')
                        self.f16 = sp.lambdify(self.symbols,np.load(path_ls + "16.npy", allow_pickle=True).tolist(),modules='numpy')
                        self.f26 = sp.lambdify(self.symbols,np.load(path_ls + "26.npy", allow_pickle=True).tolist(),modules='numpy')          
                        self.f36 = sp.lambdify(self.symbols,np.load(path_ls + "36.npy", allow_pickle=True).tolist(),modules='numpy')          
                        self.f46 = sp.lambdify(self.symbols,np.load(path_ls + "46.npy", allow_pickle=True).tolist(),modules='numpy')          
                        self.f56 = sp.lambdify(self.symbols,np.load(path_ls + "56.npy", allow_pickle=True).tolist(),modules='numpy')   

                        if self.ls_order >= 8:   # eighth order terms
                            self.f08 = sp.lambdify(self.symbols,np.load(path_ls + "08.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f18 = sp.lambdify(self.symbols,np.load(path_ls + "18.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f28 = sp.lambdify(self.symbols,np.load(path_ls + "28.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f38 = sp.lambdify(self.symbols,np.load(path_ls + "38.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f48 = sp.lambdify(self.symbols,np.load(path_ls + "48.npy", allow_pickle=True).tolist(),modules='numpy')
                            self.f58 = sp.lambdify(self.symbols,np.load(path_ls + "58.npy", allow_pickle=True).tolist(),modules='numpy')
                            
        return               
                      
    # function to check resonance and rabi oscillations at each power setting corresponding to ind                  
    def spot_check(self, ind):
        plt.figure()
        plt.scatter(self.w_bank[ind]/(2*np.pi*1e6), self.y_bank[ind])
        plt.plot(self.res_fits[ind][0]/(2*np.pi*1e6), self.res_fits[ind][1])
        plt.axvline(self.w_num_bank[ind]/(2*np.pi*1e6))
        plt.show()

        plt.figure()
        plt.scatter(self.t_bank[ind]*1e6, self.p52_bank[ind])
        plt.scatter(self.t_bank[ind]*1e6, self.p32_bank[ind])
        plt.scatter(self.t_bank[ind]*1e6, self.p12_bank[ind])
        plt.scatter(self.t_bank[ind]*1e6, self.m12_bank[ind])
        plt.scatter(self.t_bank[ind]*1e6, self.m32_bank[ind])
        plt.scatter(self.t_bank[ind]*1e6, self.m52_bank[ind])
        
        plt.plot(self.flop_fits[ind][0]*1e6, self.flop_fits[ind][1], color = 'black')
        plt.axvline((np.pi/self.Ω_num_bank[ind])*1e6, color = 'black')
        plt.show()

        return