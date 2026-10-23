import numpy as np
import sympy as sp
import dill
from tqdm.notebook import tqdm

# generates up to order 4 ac stark shifts throughout D5/2 manifold, including counter rotating terms

class generator:
    def __init__(self, n = 0):

        Ω06r, Ω16r, Ω17r, Ω26r, Ω27r, Ω28r, Ω37r, Ω38r, Ω39r, Ω48r, Ω49r, Ω59r = sp.symbols('Ω06r Ω16r Ω17r Ω26r Ω27r Ω28r Ω37r Ω38r Ω39r Ω48r Ω49r Ω59r')
        Ω06b, Ω16b, Ω17b, Ω26b, Ω27b, Ω28b, Ω37b, Ω38b, Ω39b, Ω48b, Ω49b, Ω59b = sp.symbols('Ω06b Ω16b Ω17b Ω26b Ω27b Ω28b Ω37b Ω38b Ω39b Ω48b Ω49b Ω59b')
        
        Ω06rc, Ω16rc, Ω17rc, Ω26rc, Ω27rc, Ω28rc, Ω37rc, Ω38rc, Ω39rc, Ω48rc, Ω49rc, Ω59rc = sp.symbols('Ω06rc Ω16rc Ω17rc Ω26rc Ω27rc Ω28rc Ω37rc Ω38rc Ω39rc Ω48rc Ω49rc Ω59rc')
        Ω06bc, Ω16bc, Ω17bc, Ω26bc, Ω27bc, Ω28bc, Ω37bc, Ω38bc, Ω39bc, Ω48bc, Ω49bc, Ω59bc = sp.symbols('Ω06bc Ω16bc Ω17bc Ω26bc Ω27bc Ω28bc Ω37bc Ω38bc Ω39bc Ω48bc Ω49bc Ω59bc')

        
        self.Δ = sp.symbols('Δ')
        self.Δc = sp.symbols('Δc')
        self.ωr = sp.symbols('ωr')
        self.ω0 = sp.symbols('ω0')

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
        
        self.Ωr = Ωr/2
        self.Ωb = Ωb/2

        self.Ωrc = Ωrc/2
        self.Ωbc = Ωbc/2
        
        self.inds = list(range(0,10))
        self.inds.remove(n)
        self.Ωs = [self.Ωr, self.Ωb, self.Ωrc, self.Ωbc]        
        
        self.n = n
        
    def calc_beam_energy(self, Ωs, order):
        # calculate where beam energy is at each junction
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
            elif Ωs[2*i] == self.Ωrc and Ωs[2*i+1] == self.Ωbc:
                curr = curr + self.ωr
            elif Ωs[2*i] == self.Ωbc and Ωs[2*i+1] == self.Ωrc:
                curr = curr - self.ωr 
            else:
                self.final_energy = 1
                return [1]
            
            self.current_energy = np.append(self.current_energy, curr)
            self.final_energy = self.current_energy[-1] 
        return self.current_energy[:-1]    

    def calc_second_order(self, p):
        terms = []
        for Ω1 in self.Ωs:
            for m in self.inds:
                num = Ω1[self.n,m]**2
                if p == 0:
                    if Ω1 == self.Ωr or Ω1 == self.Ωb:
                        den = self.Δ
                    if Ω1 == self.Ωrc or Ω1 == self.Ωbc:
                        den = self.Δc
                if p == 1:
                    den = (self.Δ)**2
                if p == 2:
                    den = (self.Δ)**3
                if p == 3:
                    den = (self.Δ)**4    
                if num != 0:
                    terms.append(num/den)
        self.terms = np.array(terms)
        return
    
    def calc_fourth_order(self, p):
        terms = []
        for Ω1 in self.Ωs:
            for Ω2 in self.Ωs:
                for Ω3 in self.Ωs:
                    for Ω4 in self.Ωs:
                        self.Ωs_terms = [Ω1, Ω2, Ω3, Ω4]
                        for k in self.inds:
                            for l in self.inds:
                                for m in self.inds:
                                    num = Ω1[self.n,m]* Ω2[m,l]* Ω3[l,k]* Ω4[k,self.n]
                                    Enm, Enl, Enk = self.get_denominator_4(self.Ωs_terms, l)
                                    if p == 0:
                                        den = Enm*Enl*Enk
                                    if p == 11:
                                        den = Enm**2*Enl*Enk
                                    if p == 12:
                                        den = Enm*Enl**2*Enk
                                    if p == 13:
                                        den = Enm*Enl*Enk**2
                                    if p == 21:
                                        den = (Enm**3)*Enl*Enk
                                    if p == 22:
                                        den = (Enm**2)*(Enl**2)*Enk
                                    if p == 23:
                                        den = Enm*(Enl**3)*Enk
                                    if p == 24:
                                        den = (Enm**2)*Enl*(Enk**2)
                                    if p == 25:
                                        den = Enm*(Enl**2)*(Enk**2)
                                    if p == 26:
                                        den = Enm*Enl*(Enk**3)                                    
                                    if p == 27:
                                        den = (Enm**2)*Enl*Enk
                                    if p == 28:
                                        den = Enm*(Enl**2)*Enk
                                    if p == 29:
                                        den = Enm*Enl*(Enk**2)
                                    if num != 0:
                                        if self.final_energy == 0: 
                                            terms.append(num/den)
        self.terms = np.array(terms)
        return
    def calc_state_energy_4(self, l):
        # calculate where state energy is at each junction
        return np.array([-self.ω0*(l - self.n)])
    def get_denominator_4(self, Ωs, l):
        
        if (Ωs[0] == self.Ωr or Ωs[0] == self.Ωb) and (Ωs[1] == self.Ωr or Ωs[1] == self.Ωb):
            Enm = self.Δ
        elif (Ωs[0] == self.Ωrc or Ωs[0] == self.Ωbc) and (Ωs[1] == self.Ωrc or Ωs[1] == self.Ωbc):
            Enm = self.Δc            
        else:
            self.final_energy = 1
            return 1, 1, 1
        
        if (Ωs[2] == self.Ωr or Ωs[2] == self.Ωb) and (Ωs[3] == self.Ωr or Ωs[3] == self.Ωb):
            Enk = self.Δ
        elif (Ωs[2] == self.Ωrc or Ωs[2] == self.Ωbc) and (Ωs[3] == self.Ωrc or Ωs[3] == self.Ωbc):
            Enk = self.Δc    
        else:
            self.final_energy = 1
            return 1, 1, 1

        
        Enl = self.calc_beam_energy(Ωs, 4)[0] - self.calc_state_energy_4(l)[0]
        return Enm, Enl, Enk    
    
    def ma(self, x, y):
        return (x[:, None] * y[None, :]).flatten()
        
    def calculate_lightshifts(self, order):
        
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