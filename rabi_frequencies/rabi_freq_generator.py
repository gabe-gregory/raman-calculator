import numpy as np
import sympy as sp
import dill
from tqdm.notebook import tqdm
from itertools import product

# generates up to 6th order Rabi frequency between states in D5/2
# 8th and 10th orders are not tested


class generator:
    def __init__(self, n = 0, f = 1, res = 1,  polarizations = 'impure', terms = 'basic'):
        self.terms = terms

        Ω06r, Ω16r, Ω17r, Ω26r, Ω27r, Ω28r, Ω37r, Ω38r, Ω39r, Ω48r, Ω49r, Ω59r = sp.symbols('Ω06r Ω16r Ω17r Ω26r Ω27r Ω28r Ω37r Ω38r Ω39r Ω48r Ω49r Ω59r')
        Ω06b, Ω16b, Ω17b, Ω26b, Ω27b, Ω28b, Ω37b, Ω38b, Ω39b, Ω48b, Ω49b, Ω59b = sp.symbols('Ω06b Ω16b Ω17b Ω26b Ω27b Ω28b Ω37b Ω38b Ω39b Ω48b Ω49b Ω59b')
        Ω06rc, Ω16rc, Ω17rc, Ω26rc, Ω27rc, Ω28rc, Ω37rc, Ω38rc, Ω39rc, Ω48rc, Ω49rc, Ω59rc = sp.symbols('Ω06rc Ω16rc Ω17rc Ω26rc Ω27rc Ω28rc Ω37rc Ω38rc Ω39rc Ω48rc Ω49rc Ω59rc')
        Ω06bc, Ω16bc, Ω17bc, Ω26bc, Ω27bc, Ω28bc, Ω37bc, Ω38bc, Ω39bc, Ω48bc, Ω49bc, Ω59bc = sp.symbols('Ω06bc Ω16bc Ω17bc Ω26bc Ω27bc Ω28bc Ω37bc Ω38bc Ω39bc Ω48bc Ω49bc Ω59bc')
        Ω012r, Ω112r, Ω113r, Ω212r, Ω213r, Ω214r, Ω313r, Ω314r, Ω315r, Ω414r, Ω415r, Ω515r = sp.symbols('Ω012r Ω112r Ω113r Ω212r Ω213r Ω214r Ω313r Ω314r Ω315r Ω414r Ω415r Ω515r')
        Ω012b, Ω112b, Ω113b, Ω212b, Ω213b, Ω214b, Ω313b, Ω314b, Ω315b, Ω414b, Ω415b, Ω515b = sp.symbols('Ω012b Ω112b Ω113b Ω212b Ω213b Ω214b Ω313b Ω314b Ω315b Ω414b Ω415b Ω515b')
        r, b = sp.symbols('r b')
        
        ω01, ω12, ω23, ω34, ω45 = sp.symbols('ω01 ω12 ω23 ω34 ω45')
        self.Δ = sp.symbols('Δ')
        self.Δc = sp.symbols('Δc')
        self.Δf = sp.symbols('Δf')
        self.ωr = sp.symbols('ωr')
        self.ω0 = sp.symbols('ω0')

        self.r = r
        self.b = b
        
        Ωr = sp.Matrix([
            [0   , 0   , 0   , 0   , 0   , 0   , Ω06r, 0   , 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω16r, Ω17r, 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω26r, Ω27r, Ω28r, 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω37r, Ω38r, Ω39r],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω48r, Ω49r],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω59r],
            [Ω06r, Ω16r, Ω26r, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , Ω17r, Ω27r, Ω37r, 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , Ω28r, Ω38r, Ω48r, 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , 0   , Ω39r, Ω49r, Ω59r, 0   , 0   , 0   , 0   ]])
        
        Ωb = sp.Matrix([
            [0   , 0   , 0   , 0   , 0   , 0   , Ω06b, 0   , 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω16b, Ω17b, 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω26b, Ω27b, Ω28b, 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω37b, Ω38b, Ω39b],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω48b, Ω49b],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω59b],
            [Ω06b, Ω16b, Ω26b, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , Ω17b, Ω27b, Ω37b, 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , Ω28b, Ω38b, Ω48b, 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , 0   , Ω39b, Ω49b, Ω59b, 0   , 0   , 0   , 0   ]])
        
        Ωrc = sp.Matrix([
            [0   , 0   , 0   , 0   , 0   , 0   , Ω06rc, 0   , 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω16rc, Ω17rc, 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω26rc, Ω27rc, Ω28rc, 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω37rc, Ω38rc, Ω39rc],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω48rc, Ω49rc],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω59rc],
            [Ω06rc, Ω16rc, Ω26rc, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , Ω17rc, Ω27rc, Ω37rc, 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , Ω28rc, Ω38rc, Ω48rc, 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , 0   , Ω39rc, Ω49rc, Ω59rc, 0   , 0   , 0   , 0   ]])

        Ωbc = sp.Matrix([
            [0   , 0   , 0   , 0   , 0   , 0   , Ω06bc, 0   , 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω16bc, Ω17bc, 0   , 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , Ω26bc, Ω27bc, Ω28bc, 0   ],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω37bc, Ω38bc, Ω39bc],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω48bc, Ω49bc],
            [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω59bc],
            [Ω06bc, Ω16bc, Ω26bc, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , Ω17bc, Ω27bc, Ω37bc, 0   , 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , Ω28bc, Ω38bc, Ω48bc, 0   , 0   , 0   , 0   , 0   ],
            [0   , 0   , 0   , Ω39bc, Ω49bc, Ω59bc, 0   , 0   , 0   , 0   ]])

        Ωrf = sp.Matrix([
            [0    , 0    , 0    , 0    , 0    , 0    , Ω012r ,  0   , 0    , 0    ],
            [0    , 0    , 0    , 0    , 0    , 0    , Ω112r , Ω113r, 0    , 0    ],
            [0    , 0    , 0    , 0    , 0    , 0    , Ω212r , Ω213r, Ω214r, 0    ],
            [0    , 0    , 0    , 0    , 0    , 0    , 0     , Ω313r, Ω314r, Ω315r],
            [0    , 0    , 0    , 0    , 0    , 0    , 0     , 0    , Ω414r, Ω415r],
            [0    , 0    , 0    , 0    , 0    , 0    , 0     , 0    , 0    , Ω515r],
            [Ω012r, Ω112r, Ω212r, 0    , 0    , 0    , 0     , 0    , 0    , 0    ],
            [0    , Ω113r, Ω213r, Ω313r, 0    , 0    , 0     , 0    , 0    , 0    ],
            [0    , 0    , Ω214r, Ω314r, Ω414r, 0    , 0     , 0    , 0    , 0    ],
            [0    , 0    , 0    , Ω315r, Ω415r, Ω515r, 0     , 0    , 0    , 0    ]])
        
        Ωbf = sp.Matrix([
            [0    , 0    , 0    , 0    , 0    , 0    , Ω012b ,  0   , 0    , 0    ],
            [0    , 0    , 0    , 0    , 0    , 0    , Ω112b , Ω113b, 0    , 0    ],
            [0    , 0    , 0    , 0    , 0    , 0    , Ω212b , Ω213b, Ω214b, 0    ],
            [0    , 0    , 0    , 0    , 0    , 0    , 0     , Ω313b, Ω314b, Ω315b],
            [0    , 0    , 0    , 0    , 0    , 0    , 0     , 0    , Ω414b, Ω415b],
            [0    , 0    , 0    , 0    , 0    , 0    , 0     , 0    , 0    , Ω515b],
            [Ω012b, Ω112b, Ω212b, 0    , 0    , 0    , 0     , 0    , 0    , 0    ],
            [0    , Ω113b, Ω213b, Ω313b, 0    , 0    , 0     , 0    , 0    , 0    ],
            [0    , 0    , Ω214b, Ω314b, Ω414b, 0    , 0     , 0    , 0    , 0    ],
            [0    , 0    , 0    , Ω315b, Ω415b, Ω515b, 0     , 0    , 0    , 0    ]])
        
        
        self.Ωr = Ωr/2
        self.Ωb = Ωb/2
        
        self.Ωrc = Ωrc/2
        self.Ωbc = Ωbc/2

        self.Ωrf = Ωrf/2
        self.Ωbf = Ωbf/2
        
        if polarizations == 'pure':
            
            Ωr = sp.Matrix([
                [0   , 0   , 0   , 0   , 0   , 0   , Ω06r, 0   , 0   , 0   ],
                [0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω17r, 0   , 0   ],
                [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω28r, 0   ],
                [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0  , Ω39r],
                [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [Ω06r, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , Ω17r, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , 0   , Ω28r, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , 0   , 0   , Ω39r, 0   , 0   , 0   , 0   , 0   , 0   ]])

            Ωrc = sp.Matrix([
                [0   , 0   , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0],
                [0   , 0   , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0],
                [0   , 0   , 0    , 0    , 0    , 0    , Ω26rc, 0    , 0    , 0],
                [0   , 0   , 0    , 0    , 0    , 0    , 0    , Ω37rc, 0    , 0],
                [0   , 0   , 0    , 0    , 0    , 0    , 0    , 0    , Ω48rc, 0],
                [0   , 0   , 0    , 0    , 0    , 0    , 0    , 0    , 0    , Ω59rc],
                [0   , 0   , Ω26rc, 0    , 0    , 0    , 0    , 0    , 0    , 0],
                [0   , 0   , 0    , Ω37rc, 0    , 0    , 0    , 0    , 0    , 0],
                [0   , 0   , 0    , 0    , Ω48rc, 0    , 0    , 0    , 0    , 0],
                [0   , 0   , 0    , 0    ,     0, Ω59rc, 0    , 0    , 0    , 0]])
            
            Ωrf = sp.Matrix([
                [0    , 0   , 0   , 0   , 0   , 0   , Ω012r, 0   , 0   , 0   ],
                [0    , 0   , 0   , 0   , 0   , 0   , 0   , Ω113r, 0   , 0   ],
                [0    , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω214r, 0   ],
                [0    , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , Ω315r],
                [0    , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0    , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [Ω012r, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , Ω113r, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , 0   , Ω214r, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                [0   , 0   , 0   , Ω315r, 0   , 0   , 0   , 0   , 0   , 0   ]])
            
            self.Ωr = Ωr/2
            self.Ωrc = Ωrc/2
            self.Ωrf = Ωrf/2
            
        self.inds = list(range(0,10))
        
        if self.terms == 'basic':
            self.Ωs = [self.Ωr, self.Ωb]
        
        if self.terms == 'c':
            self.Ωs = [self.Ωr, self.Ωb, self.Ωrc, self.Ωbc]
            
        if self.terms == 'f':
            self.Ωs = [self.Ωr, self.Ωb, self.Ωrf, self.Ωbf]
            
        if self.terms == 'cf':
            self.Ωs = [self.Ωr, self.Ωb, self.Ωrc, self.Ωbc, self.Ωrf, self.Ωbf]
        
        self.n = n
        self.f = f
        self.res = res
        self.polarizations = polarizations
        
        self.state_energy_dict = {0:0, 1:ω01, 2:(ω01+ ω12), 3:(ω01+ ω12+ ω23), 4:(ω01+ ω12+ ω23+ ω34), 5:(ω01 + ω12 + ω23 + ω34 + ω45), 6:self.Δ, 7:self.Δ, 8:self.Δ, 9:self.Δ}
        
    def calc_beam_energy(self, Ωs, order):
        # calculate where beam energy is at each junction
        self.current_energy_step = np.array([])
        curr = 0
        
        for i in range(int(order)):
            mod = 1
            if i % 2:
                mod = -1
            if Ωs[i] == self.Ωr:
                val = mod*self.r
            if Ωs[i] == self.Ωb:
                val = mod*self.b
            if Ωs[i] == self.Ωb or Ωs[i] == self.Ωr:
                self.current_energy_step = np.append(self.current_energy_step, val)
        
        self.current_energy_letters = np.cumsum(self.current_energy_step)
        
        
        self.current_energy = np.array([])
        curr = 0
        #for i in range(int((order-2)/2)):
        for i in range(int((order)/2)):
            if Ωs[2*i] == Ωs[2*i+1]:
                curr = 0 + curr
            elif Ωs[2*i] == self.Ωr and Ωs[2*i+1] == self.Ωb:
                curr = curr -self.ωr
            elif Ωs[2*i] == self.Ωb and Ωs[2*i+1] == self.Ωr:
                curr = curr + self.ωr 
            elif Ωs[2*i] == self.Ωrf and Ωs[2*i+1] == self.Ωbf:
                curr = curr -self.ωr
            elif Ωs[2*i] == self.Ωbf and Ωs[2*i+1] == self.Ωrf:
                curr = curr + self.ωr 
            elif Ωs[2*i] == self.Ωrc and Ωs[2*i+1] == self.Ωbc:
                curr = curr + self.ωr
            elif Ωs[2*i] == self.Ωbc and Ωs[2*i+1] == self.Ωrc:
                curr = curr - self.ωr 
            else:
                self.final_energy = 1
                
            self.current_energy = np.append(self.current_energy, curr)
            self.final_energy = self.current_energy[-1] 
        return self.current_energy[:-1]   
    
    def calc_second_order(self):
        terms = []
        for Ω1 in self.Ωs:
            for Ω2 in self.Ωs:
                self.Ωs_terms = [Ω1, Ω2]
                for m in self.inds:
                    
                    num = Ω1[self.n,m]*Ω2[m,self.f]
                    if Ω1 == self.Ωr or Ω1 == self.Ωb:
                        den = self.Δ
                    if Ω1 == self.Ωrc or Ω1 == self.Ωbc:
                        den = self.Δc
                    if Ω1 == self.Ωrf or Ω1 == self.Ωbf:
                        den = self.Δf 
                        
                    if num != 0:
                        #print(self.Ωs_terms)
                        self.calc_beam_energy(self.Ωs_terms, 2)
                        if self.final_energy == -self.res*self.ωr:
                            terms.append(2*num/den)
        self.terms = np.array(terms)
        return
    
    def calc_fourth_order(self):
        terms = []
        self.track = []
        for Ω1 in self.Ωs:
            for Ω2 in self.Ωs:
                for Ω3 in self.Ωs:
                    for Ω4 in self.Ωs:
                        Ωs_terms = [Ω1, Ω2, Ω3, Ω4]
                        for k in self.inds:
                            if Ω4[k,self.f] != 0:
                                for l in self.inds:
                                    if Ω3[l,k] != 0:
                                        for m in self.inds:
                                            if Ω2[m,l] != 0: 
                                                num = Ω1[self.n,m]* Ω2[m,l]* Ω3[l,k]* Ω4[k,self.f]
                                                Enm, Enl, Enk = self.get_denominator_4(Ωs_terms, l)

                                                den = Enm*Enl*Enk

                                                if num != 0:
                                                    if self.final_energy == -self.res*self.ωr and den != 0: 
                                                        terms.append(2*num/den)
                                                        self.track.append([2*Ω1[self.n,m], 2*Ω2[m,l], 2*Ω3[l,k], 2*Ω4[k,self.f]])
        self.terms = np.array(terms)
        return
    def calc_state_energy_4(self, l):
        # calculate where state energy is at each junction
        self.unshifted_energies = np.array([-self.ω0*float((l - self.n))])
        return(np.array([-self.state_energy_dict[l] + self.state_energy_dict[self.n]]))
        # return np.array([-self.ω0*(l - self.n)])
    
    def get_denominator_4(self, Ωs, l):
        if (Ωs[0] == self.Ωr or Ωs[0] == self.Ωb) and (Ωs[1] == self.Ωr or Ωs[1] == self.Ωb):
            Enm = self.Δ
        elif (Ωs[0] == self.Ωrc or Ωs[0] == self.Ωbc) and (Ωs[1] == self.Ωrc or Ωs[1] == self.Ωbc):
            Enm = self.Δc    
        elif (Ωs[0] == self.Ωrf or Ωs[0] == self.Ωbf) and (Ωs[1] == self.Ωrf or Ωs[1] == self.Ωbf):
            Enm = self.Δf       
        else:
            Enm = 0
            self.final_energy = 1
        
        if (Ωs[2] == self.Ωr or Ωs[2] == self.Ωb) and (Ωs[3] == self.Ωr or Ωs[3] == self.Ωb):
            Enk = self.Δ
        elif (Ωs[2] == self.Ωrc or Ωs[2] == self.Ωbc) and (Ωs[3] == self.Ωrc or Ωs[3] == self.Ωbc):
            Enk = self.Δc    
        elif (Ωs[2] == self.Ωrf or Ωs[2] == self.Ωbf) and (Ωs[3] == self.Ωrf or Ωs[3] == self.Ωbf):
            Enk = self.Δf  
        else:
            Enk = 0
            self.final_energy = 1
            
        Enl = self.calc_beam_energy(Ωs, 4)[0] - self.calc_state_energy_4(l)[0]
        for i in range(len(self.unshifted_energies)):
            if self.unshifted_energies[i] == self.current_energy[i]*((self.f - self.n)/self.res)*self.ω0/self.ωr:
                Enl = 0
        return Enm, Enl, Enk        
    
    def calc_sixth_order(self):
        terms = []
        self.arrs = []
        self.track = []
        self.ind_arr = []
        
        self.Ωs_list = list(product(self.Ωs, repeat=6))
        self.inds_list = list(product(self.inds, repeat=5))
        
        with tqdm(total=len(self.Ωs_list ), desc="6th order") as pbar:
            for Ω1 in self.Ωs:
                for Ω2 in self.Ωs:
                    for Ω3 in self.Ωs:
                        for Ω4 in self.Ωs:
                            for Ω5 in self.Ωs:
                                for Ω6 in self.Ωs:
                                    pbar.update(1)
                                    Ωs_terms = [Ω1, Ω2, Ω3, Ω4, Ω5, Ω6]
                                    for i in self.inds:
                                        if Ω6[i,self.f] != 0:
                                            for j in self.inds:
                                                if Ω5[j,i] != 0:
                                                    for k in self.inds:
                                                        if Ω4[k,j] != 0:
                                                            for l in self.inds:
                                                                if Ω3[l,k] != 0:
                                                                    for m in self.inds:
                                                                        if Ω2[m,l] != 0 and Ω1[self.n,m] != 0:
                                                                            
                                                                            num = Ω1[self.n,m]* Ω2[m,l]* Ω3[l,k]* Ω4[k,j]* Ω5[j,i]* Ω6[i,self.f]
                                                                            Enm, Enl, Enk, Enj, Eni = self.get_denominator_6(Ωs_terms, l, j)

                                                                            # do not include patways with photons doubling back
                                                                            Ωs = [Ω1[self.n,m], Ω2[m,l], Ω3[l,k], Ω4[k,j], Ω5[j,i], Ω6[i,self.f]]   
                                                                            if any(x == y for x, y in zip(Ωs, Ωs[1:])):
                                                                                Enl = 0

                                                                            den = Enm*Enl*Enk*Enj*Eni

                                                                            if num != 0:
                                                                                if self.final_energy == -self.res*self.ωr and den != 0: 
                                                                                    terms.append(2*num/den)

                                                                                    self.arrs.append([self.unshifted_energies, self.current_energy])
                                                                                    self.track.append([2*Ω1[self.n,m], 2*Ω2[m,l], 2*Ω3[l,k], 2*Ω4[k,j], 2*Ω5[j,i], 2*Ω6[i,self.f]])
                                                                                    self.ind_arr.append([self.n, m, l, k, j, i, self.f])

        self.terms = np.array(terms)
        return
    
    def calc_state_energy_6(self, l, j):
        # calculate where state energy is at each junction
        self.unshifted_energies = np.array([-self.ω0*float((l - self.n)), -self.ω0*float((j - self.n))])
        return(np.array([-self.state_energy_dict[l] + self.state_energy_dict[self.n], -self.state_energy_dict[j] + self.state_energy_dict[self.n]]))

    def get_denominator_6(self, Ωs, l, j):
        if (Ωs[0] == self.Ωr or Ωs[0] == self.Ωb) and (Ωs[1] == self.Ωr or Ωs[1] == self.Ωb):
            Enm = self.Δ
        elif (Ωs[0] == self.Ωrc or Ωs[0] == self.Ωbc) and (Ωs[1] == self.Ωrc or Ωs[1] == self.Ωbc):
            Enm = self.Δc     
        elif (Ωs[0] == self.Ωrf or Ωs[0] == self.Ωbf) and (Ωs[1] == self.Ωrf or Ωs[1] == self.Ωbf):
            Enm = self.Δf   
        else:
            Enm = 0
            self.final_energy = 1
        
        if (Ωs[2] == self.Ωr or Ωs[2] == self.Ωb) and (Ωs[3] == self.Ωr or Ωs[3] == self.Ωb):
            Enk = self.Δ
        elif (Ωs[2] == self.Ωrc or Ωs[2] == self.Ωbc) and (Ωs[3] == self.Ωrc or Ωs[3] == self.Ωbc):
            Enk = self.Δc    
        elif (Ωs[2] == self.Ωrf or Ωs[2] == self.Ωbf) and (Ωs[3] == self.Ωrf or Ωs[3] == self.Ωbf):
            Enk = self.Δf  
        else:
            Enk = 0
            self.final_energy = 1
            
        if (Ωs[4] == self.Ωr or Ωs[4] == self.Ωb) and (Ωs[5] == self.Ωr or Ωs[5] == self.Ωb):
            Eni = self.Δ
        elif (Ωs[4] == self.Ωrc or Ωs[4] == self.Ωbc) and (Ωs[5] == self.Ωrc or Ωs[5] == self.Ωbc):
            Eni = self.Δc    
        elif (Ωs[4] == self.Ωrf or Ωs[4] == self.Ωbf) and (Ωs[5] == self.Ωrf or Ωs[5] == self.Ωbf):
            Eni = self.Δf    
        else:
            Eni = 0
            self.final_energy = 1

        Enl = self.calc_beam_energy(Ωs, 6)[0] - self.calc_state_energy_6(l, j)[0]
        Enj = self.calc_beam_energy(Ωs, 6)[1] - self.calc_state_energy_6(l, j)[1]

        for i in range(len(self.unshifted_energies)):
            if self.unshifted_energies[i] == self.current_energy[i]*((self.f - self.n)/self.res)*self.ω0/self.ωr:
                Enl = 0
        
        return Enm, Enl, Enk, Enj, Eni
    
    def ma(self, x, y):
        return (x[:, None] * y[None, :]).flatten()
        

    def calculate_rabi_freq(self, order):
        
        if order == 2:
            # 2-photon pathways
            self.calc_second_order(0)
            self.E2 = self.terms

        if order == 4:
            # 4-photon pathways
            self.calc_fourth_order(0)
            t1 = self.terms
    
            # 2-photon pathways
            self.calc_second_order(1)
            t2 = self.terms
            
            self.E4 = np.concatenate((t1, -self.ma(self.E2, t2)))
            
        if order == 6:
            # 6-photon pathways
            self.calc_sixth_order(0)
            
            # 4-photon pathways
            t1 = self.terms
            self.calc_fourth_order(11)
            t2 = self.terms
            self.calc_fourth_order(12)
            t3 = self.terms
            self.calc_fourth_order(13)
            t4 = self.terms

            # 2-photon pathways
            self.calc_second_order(1)
            t5 = self.terms
            self.calc_second_order(2)
            t6 = self.terms
            
            self.E6 = np.concatenate((t1, -self.ma(self.E2, np.concatenate((t2, t3, t4))), -self.ma(self.E4, t5), + self.ma(self.ma(self.E2, self.E2), t6)))
            

        