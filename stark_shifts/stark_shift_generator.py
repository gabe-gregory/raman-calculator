import numpy as np
import sympy as sp
import dill
from tqdm.notebook import tqdm

# generates up to 8th order stark shifts on D5/2 states

class generator:
    def __init__(self, n = 0):

        Ω06r, Ω16r, Ω17r, Ω26r, Ω27r, Ω28r, Ω37r, Ω38r, Ω39r, Ω48r, Ω49r, Ω59r = sp.symbols('Ω06r Ω16r Ω17r Ω26r Ω27r Ω28r Ω37r Ω38r Ω39r Ω48r Ω49r Ω59r')
        Ω06b, Ω16b, Ω17b, Ω26b, Ω27b, Ω28b, Ω37b, Ω38b, Ω39b, Ω48b, Ω49b, Ω59b = sp.symbols('Ω06b Ω16b Ω17b Ω26b Ω27b Ω28b Ω37b Ω38b Ω39b Ω48b Ω49b Ω59b')
        self.Δ = sp.symbols('Δ')
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

        self.Ωr = Ωr/2
        self.Ωb = Ωb/2

        self.inds = list(range(0,10))
        self.inds.remove(n)
        self.Ωs = [self.Ωr, self.Ωb]
        
        self.n = n

    def calc_beam_energy(self, Ωs, order):
        # calculate where beam energy is at each junction
        current_energy = np.array([])
        curr = 0
        #for i in range(int((order-2)/2)):
        for i in range(int((order)/2)):
            if Ωs[2*i] == Ωs[2*i+1]:
                curr = 0 + curr
            if Ωs[2*i] == self.Ωr and Ωs[2*i+1] == self.Ωb:
                curr = curr -self.ωr
            if Ωs[2*i] == self.Ωb and Ωs[2*i+1] == self.Ωr:
                curr = curr + self.ωr 
            current_energy = np.append(current_energy, curr)
            self.final_energy = current_energy[-1] 
        return current_energy[:-1]    

        
    def calc_second_order(self, p):
        terms = []
        for Ω1 in self.Ωs:
            for m in self.inds:
                num = Ω1[self.n,m]**2
                if p == 0:
                    den = self.Δ
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
                        Ωs_terms = [Ω1, Ω2, Ω3, Ω4]
                        for k in self.inds:
                            for l in self.inds:
                                for m in self.inds:
                                    num = Ω1[self.n,m]* Ω2[m,l]* Ω3[l,k]* Ω4[k,self.n]
                                    Enm, Enl, Enk = self.get_denominator_4(Ωs_terms, l)
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
        Enm = self.Δ
        Enk = self.Δ
        Enl = self.calc_beam_energy(Ωs, 4)[0] - self.calc_state_energy_4(l)[0]
        return Enm, Enl, Enk    
    
    def calc_sixth_order(self,p):
        terms = []
        with tqdm(total=2**6, desc="6th order") as pbar:
            for Ω1 in self.Ωs:
                for Ω2 in self.Ωs:
                    for Ω3 in self.Ωs:
                        for Ω4 in self.Ωs:
                            for Ω5 in self.Ωs:
                                for Ω6 in self.Ωs:
                                    pbar.update(1)
                                    Ωs_terms = [Ω1, Ω2, Ω3, Ω4, Ω5, Ω6]
                                    for i in self.inds:
                                        if Ω6[i,self.n] != 0:
                                            for j in self.inds:
                                                if Ω5[j,i] != 0:
                                                    for k in self.inds:
                                                        if Ω4[k,j] != 0:
                                                            for l in self.inds:
                                                                if Ω3[l,k] != 0:
                                                                    for m in self.inds:
                                                                        if Ω2[m,l] != 0 and Ω1[self.n,m] != 0:
                                                                            num = Ω1[self.n,m]* Ω2[m,l]* Ω3[l,k]* Ω4[k,j]* Ω5[j,i]* Ω6[i,self.n]
                                                                            Enm, Enl, Enk, Enj, Eni = self.get_denominator_6(Ωs_terms, l, j)

                                                                            if p == 0:
                                                                                den = Enm*Enl*Enk*Enj*Eni
                                                                            if p == 11:
                                                                                den = (Enm**2)*Enl*Enk*Enj*Eni
                                                                            if p == 12:
                                                                                den = Enm*(Enl**2)*Enk*Enj*Eni
                                                                            if p == 13:
                                                                                den = Enm*Enl*(Enk**2)*Enj*Eni
                                                                            if p == 14:
                                                                                den = Enm*Enl*Enk*(Enj**2)*Eni
                                                                            if p == 15:
                                                                                den = Enm*Enl*Enk*Enj*(Eni**2)
                                                                            if num != 0:
                                                                                if self.final_energy == 0: 
                                                                                    terms.append(num/den)
        self.terms = np.array(terms)
        return
    def calc_state_energy_6(self, l, j):
        # calculate where state energy is at each junction
        return np.array([-self.ω0*(l - self.n), -self.ω0*(j - self.n)])
    def get_denominator_6(self, Ωs, l, j):
        Enm = self.Δ
        Enk = self.Δ
        Eni = self.Δ

        Enl = self.calc_beam_energy(Ωs, 6)[0] - self.calc_state_energy_6(l, j)[0]
        Enj = self.calc_beam_energy(Ωs, 6)[1] - self.calc_state_energy_6(l, j)[1]
        return Enm, Enl, Enk, Enj, Eni
    
    
    def calc_eighth_order(self,p):
        terms = []
        with tqdm(total=2**8, desc="8th order") as pbar:
            for Ω1 in self.Ωs:
                for Ω2 in self.Ωs:
                    for Ω3 in self.Ωs:
                        for Ω4 in self.Ωs:
                            for Ω5 in self.Ωs:
                                for Ω6 in self.Ωs:
                                    for Ω7 in self.Ωs:
                                        for Ω8 in self.Ωs:
                                            pbar.update(1)
                                            Ωs_terms = [Ω1, Ω2, Ω3, Ω4, Ω5, Ω6, Ω7, Ω8]
                                            for g in self.inds:
                                                if Ω8[g,self.n] != 0:
                                                    for h in self.inds:
                                                        if Ω7[h,g] != 0:
                                                            for i in self.inds:
                                                                if Ω6[i,h] != 0:
                                                                    for j in self.inds:
                                                                        if Ω5[j,i] != 0:
                                                                            for k in self.inds:
                                                                                if Ω4[k,j] != 0:
                                                                                    for l in self.inds:
                                                                                        if Ω3[l,k] != 0:
                                                                                            for m in self.inds:
                                                                                                if Ω2[m,l] != 0 and Ω1[self.n,m]!= 0:
                                                                                                    num = Ω1[self.n,m]* Ω2[m,l]* Ω3[l,k]* Ω4[k,j]* Ω5[j,i]* Ω6[i,h]* Ω7[h,g]* Ω8[g,self.n]
                                                                                                    Enm, Enl, Enk, Enj, Eni, Enh, Eng = self.get_denominator_8(Ωs_terms, l, j, h)

                                                                                                    if p == 0:
                                                                                                        den = Enm*Enl*Enk*Enj*Eni*Enh*Eng

                                                                                                    if self.final_energy == 0: 
                                                                                                        terms.append(num/den)

        self.terms = np.array(terms)
        return
    def calc_state_energy_8(self, l, j, h):
        # calculate where state energy is at each junction
        return np.array([-self.ω0*(l - self.n), -self.ω0*(j - self.n), -self.ω0*(h - self.n)])
    def get_denominator_8(self, Ωs, l, j, h):
        Enm = self.Δ
        Enk = self.Δ
        Eni = self.Δ
        Eng = self.Δ

        Enl = self.calc_beam_energy(Ωs, 8)[0] - self.calc_state_energy_8(l, j, h)[0]
        Enj = self.calc_beam_energy(Ωs, 8)[1] - self.calc_state_energy_8(l, j, h)[1]
        Enh = self.calc_beam_energy(Ωs, 8)[2] - self.calc_state_energy_8(l, j, h)[2]
        return Enm, Enl, Enk, Enj, Eni, Enh, Eng
    
    
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
            
        if order == 8:
            # 8-photon pathway
            self.calc_eighth_order(0)
            t1 = self.terms
            
            # 6-photon pathways
            self.calc_sixth_order(11)
            t2 = self.terms
            self.calc_sixth_order(12)
            t3 = self.terms
            self.calc_sixth_order(13)
            t4 = self.terms
            self.calc_sixth_order(14)
            t5 = self.terms
            self.calc_sixth_order(15)
            t6 = self.terms
            
            # 4-photon pathways
            self.calc_fourth_order(21)
            t7 = self.terms
            self.calc_fourth_order(22)
            t8 = self.terms
            self.calc_fourth_order(23)
            t9 = self.terms
            self.calc_fourth_order(24)
            t10 = self.terms
            self.calc_fourth_order(25)
            t11 = self.terms
            self.calc_fourth_order(26)
            t12 = self.terms
            self.calc_fourth_order(27)
            t13 = self.terms
            self.calc_fourth_order(28)
            t14 = self.terms
            self.calc_fourth_order(29)
            t15 = self.terms
            
            # 2-photon pathways
            self.calc_second_order(2)
            t16 = self.terms
            self.calc_second_order(3)
            t17 = self.terms
            self.calc_second_order(1)
            t18 = self.terms

            self.E8 = np.concatenate((t1, -self.ma(self.E2, np.concatenate((t2, t3, t4, t5, t6))),
                                     self.ma(self.ma(self.E2, self.E2), np.concatenate((t7, t8,t9,t10,t11,t12))),
                                     self.ma(self.E4, np.concatenate((t13,t14,t15))),
                                     2*self.ma(self.ma(self.E2, self.E4), t16),
                                     -self.ma(self.ma(self.ma(self.E2, self.E2),self.E2), t17),
                                     -self.ma(self.E6, t18)))
        