# -*- coding: utf-8 -*-
"""
Functions useful for computing photon scattering probabilities
"""

def CMD_manifold(ion_setup, lowerManifold, upperManifold, mu):
    from intermediateField_E1MatrixElement import FmF_Steck_E1MatrixElement
    Nlower = len(lowerManifold)
    Nupper = len(upperManifold)

    # class PsiAsFmF:
    #     def __init__(self, FmFName, FmFComponents, ComponentAmplitudes, I, S, L, J):
    #         self.FmFName = FmFName
    #         self.FmFComponents = FmFComponents
    #         self.ComponentAmplitudes = ComponentAmplitudes
    #         self.I = I
    #         self.S = S
    #         self.L = L
    #         self.J = J

    # FmFlist = np.zeros((int(numStates),2)) # first column = F of state, second column = mF of state
    # Hmatrix = np.zeros((int(numStates),int(numStates)))
    import numpy as np
    dipoleElementsMixedBasis = np.zeros((int(Nlower),int(Nupper)))

    for idl in range(Nlower):
        for idu in range(Nupper):
            dipoleElement = 0.0
            for baseStateLower in range(len(lowerManifold[idl].FmFComponents)):
                for baseStateUpper in range(len(upperManifold[idu].FmFComponents)):
                    # print(lowerManifold[idl].FmFComponents[baseStateLower])
                    # print(upperManifold[idu].FmFComponents[baseStateUpper])
                    Fl = lowerManifold[idl].FmFComponents[baseStateLower][0]
                    mFl = lowerManifold[idl].FmFComponents[baseStateLower][1]
                    Fu = upperManifold[idu].FmFComponents[baseStateUpper][0]
                    mFu = upperManifold[idu].FmFComponents[baseStateUpper][1]
                    Ll = lowerManifold[idl].L
                    Jl = lowerManifold[idl].J
                    Lu = upperManifold[idu].L
                    Ju = upperManifold[idu].J
                    #coeffProduct is the product of the coefficients of the basis states in the superposition
                    coeffProduct = abs((upperManifold[idu].ComponentAmplitudes[baseStateUpper])*(lowerManifold[idl].ComponentAmplitudes[baseStateLower]))
                    dipoleElementsMixedBasis[idl,idu]+= coeffProduct*FmF_Steck_E1MatrixElement(ion_setup,  Ll, Jl, Fl, mFl, Lu, Ju, Fu, mFu)

    # mat_dict = {}
    # for idl in range(Nlower):
    #     for idu in range(Nupper):
    #         key = '{},{}->{},{}'.format(int(lowerManifold[idl].FmFName[0]), int(lowerManifold[idl].FmFName[1]), int(upperManifold[idu].FmFName[0]),
    #                       int(upperManifold[idu].FmFName[1]))
    #         keyVal = np.array([dipoleElementsMixedBasis[idl, idu]])
    #         if keyVal[0] > 1e-10: #let's just skip matrix elements that are less than this arbitrary value (should exclude the 0 values)
    #             mat_dict[key] = keyVal / mu #this is just typecasting because the code broke when returning a double.

    mat_dict = {
        '{},{}->{},{}'.format(int(lowerManifold[idl].FmFName[0]), int(lowerManifold[idl].FmFName[1]), int(upperManifold[idu].FmFName[0]), int(upperManifold[idu].FmFName[1])): np.array([dipoleElementsMixedBasis[idl,idu]]) / mu\
        for idl in range(Nlower) for idu in range(Nupper)}

    return mat_dict

def construct_mat_dict(ion_setup,Ju,Jl,Lu,Ll,I,mu):
    #Create a dict of dipole transition matrix elements in units of the reduced
    #dipole matrix element <L'||d||L>, normalized by mu. dict elements are
    #named like 'F_i,mF_i->F_f,mF_f', F_i,mF_i are the F and mF quantum numbers of
    #the initial state and F_f,mF_f are the F and mF quantum numbers of the final
    #state.
    #
    #
    #ion_setup = intermediateField_setup() output for ion
    #
    #Ju = J quantum number for upper state
    #
    #Jl = J quantum number for lower state
    #
    #Lu = L quantum number for upper state
    #
    #Ll = L quantum number for lower state
    #
    #I = nuclear spin
    #
    #mu = normalization value; output from mu() for chosen upper and lower states
    
    from intermediateField_E1MatrixElement import intermediateField_E1MatrixElement
    
    mat_dict = {'{},{}->{},{}'.format(i,j,n,m): intermediateField_E1MatrixElement(ion_setup,  Ll, Jl, i, j,  Lu, Ju, n, m)/mu\
           for i in range(0,int(Jl+I)+1) for j in range(-i,i+1) for n in range(abs(int(Ju-I)),int(Ju+I)+1) for m in range(-n,n+1) if abs(j-m)<=1}

    return mat_dict

def construct_mat_dict_fine_structure(Ju,Jl,Lu,Ll,mu):
    #Create a dict of fine structure dipole transition matrix elements in units of the reduced
    #dipole matrix element <L'||d||L>, normalized by mu. dict elements are
    #named like 'mJ_i->mJ_f', where mJ_i and mJ_f are the mJ quantum numbers of
    #the initial state and  final state.
    #
    #
    #
    #Ju = J quantum number for upper state
    #
    #Jl = J quantum number for lower state
    #
    #Lu = L quantum number for upper state
    #
    #Ll = L quantum number for lower state
    #
    #mu = normalization value; output from mu() for chosen upper and lower states
    
    # from fineStructureMatrixElement import fineStructureMatrixElement
    from .fineStructureMatrixElement import fineStructureMatrixElement
    
    mat_dict = {'{}/2->{}/2'.format(int(i),int(j)): fineStructureMatrixElement(Ll, Jl, i/2, Lu, Ju, j/2, 1/2)/mu\
           for i in [-2*Jl + 2*k for k in range(int(2*Jl+1))] for j in [-2*Ju + 2*k for k in range(int(2*Ju+1))] if abs(i/2 - j/2)<=1}

    return mat_dict

def TotalScatteringAmp(Fq0,mFq0,Fq1,mFq1,I,Jl,Ju,pols_b,pols_r,mat_dict):
    #Computes the total scattering amplitude for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #In other words, if the total scattering rate R_tot is written as 
    #R_tot = gamma*g^2*A_tot/Delta^2 (where g = E*mu/2*hbar, with mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates A_tot.
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #
    #Fq0 = highest value of F for qubit states
    #
    #mFq0 = mF of qubit states
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of qubit manifold
    #
    #Ju = J quantum number of intermediate manifold
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f't
    
    amp_tot0_bm = 0;
    amp_tot0_b0 = 0;
    amp_tot0_bp = 0;
    amp_tot1_bm = 0;
    amp_tot1_b0 = 0;
    amp_tot1_bp = 0;
    amp_tot0_rm = 0;
    amp_tot0_r0 = 0;
    amp_tot0_rp = 0;
    amp_tot1_rm = 0; 
    amp_tot1_r0 = 0;  
    amp_tot1_rp = 0;      

    
    for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
        j = vec[0];
        k = vec[1];
        
        try:
            if k-mFq0==-1:
                amp_tot0_bm += (pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2;
                amp_tot0_rm += (pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2;
        except:
            pass

        try:
            if k-mFq0==0:
                amp_tot0_b0 += (pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2;
                amp_tot0_r0 += (pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2;
                
        except:
            pass 

        try:
            if k-mFq0==1:
                amp_tot0_bp += (pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2;
                amp_tot0_rp += (pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2;
        except:
            pass

        try:
            if k-mFq1==-1:
                amp_tot1_bm += (pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2;
                amp_tot1_rm += (pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2;
        except:
            pass

        try:
            if k-mFq1==0:
                amp_tot1_b0 += (pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2;
                amp_tot1_r0 += (pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2;
        except:
            pass 

        try:
            if k-mFq1==1:
                amp_tot1_bp += (pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2;
                amp_tot1_rp += (pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2;
        except:
            pass
    
    amp_total = 1/2*(amp_tot0_bm + amp_tot0_b0 + amp_tot0_bp + amp_tot0_rm + amp_tot0_r0 + amp_tot0_rp) + \
    1/2*(amp_tot1_bm + amp_tot1_b0 + amp_tot1_bp + amp_tot1_rm + amp_tot1_r0 + amp_tot1_rp);
    
    return amp_total

def Raman_Sum(Fq0,mFq0,Fq1,mFq1,I,Jl,Ju,pols_b,pols_r,mat_dict,f):
    #Computes the scattering amplitude back into qubit manifold for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*A_ram/Delta^2 (where g = E*mu/2*hbar, with mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates A_ram.
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #
    #Fq0 = F quantum numver for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum numver for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of qubit states
    #
    #Ju = J quantum number of intermediate state
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #f = branching ratio for decay from intermediate manifold to qubit manifold
    
    
    
    amp_sum = 0;
    amp_f0_bm = 0;
    amp_f0_b0 = 0;
    amp_f0_bp = 0;
    amp_f1_bm = 0;
    amp_f1_b0 = 0;
    amp_f1_bp = 0;
    amp_f0_rm = 0;
    amp_f0_r0 = 0;
    amp_f0_rp = 0;
    amp_f1_rm = 0; 
    amp_f1_r0 = 0;  
    amp_f1_rp = 0;  

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp_f0_bm = 0;
        amp_f0_b0 = 0;
        amp_f0_bp = 0;
        amp_f1_bm = 0;
        amp_f1_b0 = 0;
        amp_f1_bp = 0;
        amp_f0_rm = 0;
        amp_f0_r0 = 0;
        amp_f0_rp = 0;
        amp_f1_rm = 0; 
        amp_f1_r0 = 0;  
        amp_f1_rp = 0;      
        
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            try:
                if k-mFq0==-1 and [n,m] != [Fq0,mFq0]:
                    amp_f0_bm += pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f0_rm += pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==-1 and [n,m] != [Fq1,mFq1]:
                    amp_f1_bm += pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f1_rm += pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass

            try:
                if k-mFq0==0 and [n,m] != [Fq0,mFq0]:
                    amp_f0_b0 += pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f0_r0 += pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==0 and [n,m] != [Fq1,mFq1]:
                    amp_f1_b0 += pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f1_r0 += pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass

            try:
                if k-mFq0==1 and [n,m] != [Fq0,mFq0]:
                    amp_f0_bp += pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f0_rp += pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==1 and [n,m] != [Fq1,mFq1]:
                    amp_f1_bp += pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f1_rp += pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass            

        amp_sum += 1/2*abs(amp_f0_bm)**2 + 1/2*abs(amp_f1_bm)**2 + 1/2*abs(amp_f0_rm)**2 + 1/2*abs(amp_f1_rm)**2\
        + 1/2*abs(amp_f0_b0)**2 + 1/2*abs(amp_f1_b0)**2 + 1/2*abs(amp_f0_r0)**2 + 1/2*abs(amp_f1_r0)**2\
        + 1/2*abs(amp_f0_bp)**2 + 1/2*abs(amp_f1_bp)**2 + 1/2*abs(amp_f0_rp)**2 + 1/2*abs(amp_f1_rp)**2;      

    
    return f*amp_sum

def LowRaman_Sum(Fq0,mFq0,Fq1,mFq1,I,Jl,Ju,pols_b,pols_r,mat_dict_i,mat_dict_f,f):
    #Computes the scattering amplitude into non-qubit manifold for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>.
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*A_ram/Delta^2 (where g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates A_ram.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju = J quantum number of intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_i = dict of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #mat_dict_f = dict of transition matrix values from intermediate -> final states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    
    amp_sum = 0;
    amp_f0_bm = 0;
    amp_f0_b0 = 0;
    amp_f0_bp = 0;
    amp_f1_bm = 0;
    amp_f1_b0 = 0;
    amp_f1_bp = 0;
    amp_f0_rm = 0;
    amp_f0_r0 = 0;
    amp_f0_rp = 0;
    amp_f1_rm = 0; 
    amp_f1_r0 = 0;  
    amp_f1_rp = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp_f0_bm = 0;
        amp_f0_b0 = 0;
        amp_f0_bp = 0;
        amp_f1_bm = 0;
        amp_f1_b0 = 0;
        amp_f1_bp = 0;
        amp_f0_rm = 0;
        amp_f0_r0 = 0;
        amp_f0_rp = 0;
        amp_f1_rm = 0; 
        amp_f1_r0 = 0;  
        amp_f1_rp = 0;      
        
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            try:
                if k-mFq0==-1:
                    amp_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f0_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];

            except:
                pass
            
            try:
                if k-mFq1==-1:
                    amp_f1_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f1_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];

            except:
                pass

            try:
                if k-mFq0==0:
                    amp_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==0:
                    amp_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass

            try:
                if k-mFq0==1:
                    amp_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f0_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==1:
                    amp_f1_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                    amp_f1_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass  

        amp_sum += 1/2*abs(amp_f0_bm)**2 + 1/2*abs(amp_f1_bm)**2 + 1/2*abs(amp_f0_rm)**2 + 1/2*abs(amp_f1_rm)**2\
        + 1/2*abs(amp_f0_b0)**2 + 1/2*abs(amp_f1_b0)**2 + 1/2*abs(amp_f0_r0)**2 + 1/2*abs(amp_f1_r0)**2\
        + 1/2*abs(amp_f0_bp)**2 + 1/2*abs(amp_f1_bp)**2 + 1/2*abs(amp_f0_rp)**2 + 1/2*abs(amp_f1_rp)**2; 
        
    return f*amp_sum

def Raman_Sum_2Processes(Fq0,mFq0,Fq1,mFq1,I,Jl,Ju,pols_b,pols_r,mat_dict_i,mat_dict_f,f):
    #Computes the scattering amplitude into non-qubit manifold for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula).
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where 
    #g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates A1_ram and A2_ram.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju = J quantum number of intermediate states
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_i = dict of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #mat_dict_f = dict of transition matrix values from intermediate -> final states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    
    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    amp1_f1_bm = 0;
    amp1_f1_b0 = 0;
    amp1_f1_bp = 0;
    amp1_f0_rm = 0;
    amp1_f0_r0 = 0;
    amp1_f0_rp = 0;
    amp1_f1_rm = 0; 
    amp1_f1_r0 = 0;  
    amp1_f1_rp = 0;
    
    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;
    amp2_f1_bm = 0;
    amp2_f1_b0 = 0;
    amp2_f1_bp = 0;
    amp2_f0_rm = 0;
    amp2_f0_r0 = 0;
    amp2_f0_rp = 0;
    amp2_f1_rm = 0; 
    amp2_f1_r0 = 0;  
    amp2_f1_rp = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        amp1_f1_bm = 0;
        amp1_f1_b0 = 0;
        amp1_f1_bp = 0;
        amp1_f0_rm = 0;
        amp1_f0_r0 = 0;
        amp1_f0_rp = 0;
        amp1_f1_rm = 0; 
        amp1_f1_r0 = 0;  
        amp1_f1_rp = 0;  
        
        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        amp2_f1_bm = 0;
        amp2_f1_b0 = 0;
        amp2_f1_bp = 0;
        amp2_f0_rm = 0;
        amp2_f0_r0 = 0;
        amp2_f0_rp = 0;
        amp2_f1_rm = 0; 
        amp2_f1_r0 = 0;  
        amp2_f1_rp = 0;
        
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            #First scattering process
            try:
                 if k-mFq0==-1:
                     if [Jl,n,m] != [5/2,Fq0,mFq0]:
                         amp1_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                         amp1_f0_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];

            except:
                pass
            
            try:
                if k-mFq1==-1:
                    if [Jl,n,m] != [5/2,Fq1,mFq1]:
                        amp1_f1_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp1_f1_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];

            except:
                pass

            try:
                if k-mFq0==0:
                    if [Jl,n,m] != [5/2,Fq0,mFq0]:
                        amp1_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp1_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==0:
                    if [Jl,n,m] != [5/2,Fq1,mFq1]:
                        amp1_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp1_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except: 
                pass

            try:
                if k-mFq0==1:
                    if [Jl,n,m] != [5/2,Fq0,mFq0]:
                        amp1_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp1_f0_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if k-mFq1==1:
                    if [Jl,n,m] != [5/2,Fq1,mFq1]:
                        amp1_f1_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp1_f1_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            
            #Second scattering process; swap polarization of scattered and incident light
            try:
                if m-k==-1:
                    if [Jl,n,m] != [5/2,Fq0,mFq0]:
                        amp2_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp2_f0_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];

            except:
                pass
            
            try:
                if m-k==-1:
                    if [Jl,n,m] != [5/2,Fq1,mFq1]:
                        amp2_f1_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp2_f1_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];

            except:
                pass

            try:
                if m-k==0:
                    if [Jl,n,m] != [5/2,Fq0,mFq0]:
                        amp2_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp2_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if m-k==0:
                    if [Jl,n,m] != [5/2,Fq1,mFq1]:
                        amp2_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp2_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass

            try:
                if m-k==1:
                    if [Jl,n,m] != [5/2,Fq0,mFq0]:
                        amp2_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp2_f0_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
            
            try:
                if m-k==1:
                    if [Jl,n,m] != [5/2,Fq1,mFq1]:
                        amp2_f1_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
                        amp2_f1_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0];
            except:
                pass
        
        
        amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
        + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
        + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2; 
        
        cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
                        1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
                        1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
                        1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
                        1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
                        1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);
                        
        amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
        + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
        + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;
        
#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#                        
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#            
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#        
    return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

# def Raman2Procs_Delta(Fq0,mFq0,Fq1,mFq1,I,Jq,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list,f,tot_scat=False,stim_emit=False):
#     #Computes the i->f scattering rate for Raman transitions
#     #between |Fq0,mFq0> and |Fq1,mFq1>; includes contributions from the typically
#     #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
#     #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
#     #
#     #In other words, if the Raman scattering rate R_ram is written as 
#     #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where 
#     #g = E*mu/2*hbar, with E the electric field, mu the 
#     #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
#     #manifold, and Delta the detuning from the intermediate manifold), this code 
#     #calculates the occupation-probability weighted sum over the qubit states of (A1_ram/Delta+A2_ram/(2*wl+Delta))^2.
#     #
#     #
#     #In general cases, this code will only work where only one intermediate
#     #manifold is relevant to scattering, as the amplitudes to scatter from two
#     #intermediate manifolds must be added before squaring.
#     #
#     #Fq0 = F quantum number for |0>
#     #
#     #mFq0 = mF quantum number for |0>
#     #
#     #Fq1 = F quantum number for |1>
#     #
#     #mFq1 = mF quantum number for |1>
#     #
#     #I = nuclear spin
#     #
#     #Jq = J quantum number of qubit manifold
#     #
#     #Jl = J quantum number of final, scattered states
#     #
#     #Ju_list = list of J quantum number for intermediate states
#     #
#     #q = polarization type (1, 0, or -1)
#     #
#     #pols_i = list of fractional instensities in order [-,0,+] in beam i
#     #
#     #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states; 
#     #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
#     #each sublist is in the order 'initial -> intermediate levels matrix dict'
#     #first, followed 'intermediate -> final levels matrix dict' second.
#     #
#     #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
#     #first scattering process (-1*detuning in THz); first element in each
#     #sublist is the frequency denominator for the standard scattering
#     #process, while the second element of the sublist correspons to the often
#     #neglected second scattering process frequency denominator
#     #
#     #f = branching ratio to manifold containing f
#     #
#     import numpy as np
    
#     amp1_sum = 0;
#     cross_sum = 0;
#     amp2_sum = 0;
#     amp1_f0_bm = 0;
#     amp1_f0_b0 = 0;
#     amp1_f0_bp = 0;
#     amp1_f1_bm = 0;
#     amp1_f1_b0 = 0;
#     amp1_f1_bp = 0;
#     amp1_f0_rm = 0;
#     amp1_f0_r0 = 0;
#     amp1_f0_rp = 0;
#     amp1_f1_rm = 0; 
#     amp1_f1_r0 = 0;  
#     amp1_f1_rp = 0;
    
#     amp2_f0_bm = 0;
#     amp2_f0_b0 = 0;
#     amp2_f0_bp = 0;
#     amp2_f1_bm = 0;
#     amp2_f1_b0 = 0;
#     amp2_f1_bp = 0;
#     amp2_f0_rm = 0;
#     amp2_f0_r0 = 0;
#     amp2_f0_rp = 0;
#     amp2_f1_rm = 0; 
#     amp2_f1_r0 = 0;  
#     amp2_f1_rp = 0;

#     for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
#         n = i[0];
#         m = i[1];
        
#         amp1_f0_bm = 0;
#         amp1_f0_b0 = 0;
#         amp1_f0_bp = 0;
#         amp1_f1_bm = 0;
#         amp1_f1_b0 = 0;
#         amp1_f1_bp = 0;
#         amp1_f0_rm = 0;
#         amp1_f0_r0 = 0;
#         amp1_f0_rp = 0;
#         amp1_f1_rm = 0; 
#         amp1_f1_r0 = 0;  
#         amp1_f1_rp = 0;  
        
#         amp2_f0_bm = 0;
#         amp2_f0_b0 = 0;
#         amp2_f0_bp = 0;
#         amp2_f1_bm = 0;
#         amp2_f1_b0 = 0;
#         amp2_f1_bp = 0;
#         amp2_f0_rm = 0;
#         amp2_f0_r0 = 0;
#         amp2_f0_rp = 0;
#         amp2_f1_rm = 0; 
#         amp2_f1_r0 = 0;  
#         amp2_f1_rp = 0;
        
#         for q in range(0,len(Ju_list)):
#             Ju = Ju_list[q];
#             mat_dicts = mat_dict_list[q]
#             Deltas = Delta_list[q];
            
#             mat_dict_i = mat_dicts[0];
#             mat_dict_f = mat_dicts[1];
            
#             Delta_1 = Deltas[0];
#             Delta_2 = Deltas[1];
            
#             for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
#                 j = vec[0];
#                 k = vec[1];
                
#                 #First scattering process
#                 try:
#                     if k-mFq0==-1:
#                         if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
#                             amp1_f0_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
#                             amp1_f0_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
#                 except:
#                     pass
                
#                 try:
#                     if k-mFq1==-1:
#                         if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
#                             amp1_f1_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
#                             amp1_f1_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
#                 except:
#                     pass
    
#                 try:
#                     if k-mFq0==0:
#                         if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
#                             amp1_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
#                             amp1_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
#                 except:
#                     pass
                
#                 try:
#                     if k-mFq1==0:
#                         if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
#                             amp1_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
#                             amp1_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
#                 except:
#                     pass
    
#                 try:
#                     if k-mFq0==1:
#                         if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
#                             amp1_f0_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
#                             amp1_f0_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
#                 except:
#                     pass
                
#                 try:
#                     if k-mFq1==1:
#                         if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
#                             amp1_f1_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
#                             amp1_f1_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
#                 except:
#                     pass
                
                
#                 #Second scattering process; swap polarization of scattered and incident light
#                 try:
#                     if m-k==-1:
#                         if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
#                             amp2_f0_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
#                             amp2_f0_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
#                 except:
#                     pass
                
#                 try:
#                     if m-k==-1:
#                         if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
#                             amp2_f1_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
#                             amp2_f1_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
    
#                 except:
#                     pass
    
#                 try:
#                     if m-k==0:
#                         if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
#                             amp2_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
#                             amp2_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
#                 except:
#                     pass
                
#                 try:
#                     if m-k==0:
#                         if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
#                             amp2_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
#                             amp2_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
#                 except:
#                     pass
    
#                 try:
#                     if m-k==1:
#                         if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
#                             amp2_f0_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
#                             amp2_f0_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
#                 except:
#                     pass
                
#                 try:
#                     if m-k==1:
#                         if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
#                             amp2_f1_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
#                             amp2_f1_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
#                 except:
#                     pass
        
#         amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
#         + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
#         + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2; 
        
#         cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
#                         1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
#                         1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
#                         1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
#                         1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
#                         1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);
                        
#         amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
#         + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
#         + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;
        
# #    #Get rid of Rayleigh amplitude, if applicable
# #    if Jl == 5/2:
# #        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
# #                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
# #                        
# #        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
# #                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
# #            
# #        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
# #                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
# #        
#     return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

def Raman2Procs_Delta(Fq0,mFq0,Fq1,mFq1,I,Jq,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list,f,tot_scat=False,stim_emit=False):
    #Computes the i->f scattering rate for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
    #
    #In other words, if the Raman scattering rate R_ram is written as
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where
    #g = E*mu/2*hbar, with E the electric field, mu the
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate menifold), this code
    #calculates the occupation-probability weighted sum over the qubit states of (A1_ram/Delta+A2_ram/(2*wl+Delta))^2.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jq = J quantum number of qubit manifold
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju_list = list of J quantum number for intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states;
    #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
    #each sublist is in the order 'initial -> intermediate levels matrix dict'
    #first, followed 'intermediate -> final levels matrix dict' second.
    #
    #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
    #first scattering process (-1*detuning in THz); first element in each
    #sublist is the frequency denominator for the standard scattering
    #process, while the second element of the sublist correspons to the often
    #neglected second scattering process frequency denominator
    #
    #f = branching ratio to manifold containing f
    #
    import numpy as np

    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    amp1_f1_bm = 0;
    amp1_f1_b0 = 0;
    amp1_f1_bp = 0;
    amp1_f0_rm = 0;
    amp1_f0_r0 = 0;
    amp1_f0_rp = 0;
    amp1_f1_rm = 0;
    amp1_f1_r0 = 0;
    amp1_f1_rp = 0;

    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;
    amp2_f1_bm = 0;
    amp2_f1_b0 = 0;
    amp2_f1_bp = 0;
    amp2_f0_rm = 0;
    amp2_f0_r0 = 0;
    amp2_f0_rp = 0;
    amp2_f1_rm = 0;
    amp2_f1_r0 = 0;
    amp2_f1_rp = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];

        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        amp1_f1_bm = 0;
        amp1_f1_b0 = 0;
        amp1_f1_bp = 0;
        amp1_f0_rm = 0;
        amp1_f0_r0 = 0;
        amp1_f0_rp = 0;
        amp1_f1_rm = 0;
        amp1_f1_r0 = 0;
        amp1_f1_rp = 0;

        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        amp2_f1_bm = 0;
        amp2_f1_b0 = 0;
        amp2_f1_bp = 0;
        amp2_f0_rm = 0;
        amp2_f0_r0 = 0;
        amp2_f0_rp = 0;
        amp2_f1_rm = 0;
        amp2_f1_r0 = 0;
        amp2_f1_rp = 0;

        for q in range(0,len(Ju_list)):
            Ju = Ju_list[q];
            mat_dicts = mat_dict_list[q]
            Deltas = Delta_list[q];

            mat_dict_i = mat_dicts[0];
            mat_dict_f = mat_dicts[1];

            Delta_1 = Deltas[0];
            Delta_2 = Deltas[1];

            for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
                j = vec[0];
                k = vec[1];

                #First scattering process
                try:
                    if k-mFq0==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f0_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;

                except:
                    pass

                try:
                    if k-mFq1==-1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f1_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;

                except:
                    pass

                try:
                    if k-mFq0==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;

                except:
                    pass

                try:
                    if k-mFq1==0:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;

                except:
                    pass

                try:
                    if k-mFq0==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f0_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;

                except:
                    pass

                try:
                    if k-mFq1==1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f1_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;

                except:
                    pass


                #Second scattering process; swap polarization of scattered and incident light
                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bm += (-1)**(k - m)*(-1)**(k - mFq0)*pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f0_rm += (-1)**(k - m)*(-1)**(k - mFq0)*pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;

                except:
                    pass

                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_bm += (-1)**(k - m)*(-1)**(k - mFq1)*pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f1_rm += (-1)**(k - m)*(-1)**(k - mFq1)*pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;


                except:
                    pass

                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_b0 += (-1)**(k - m)*(-1)**(k - mFq0)*pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f0_r0 += (-1)**(k - m)*(-1)**(k - mFq0)*pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;

                except:
                    pass

                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_b0 += (-1)**(k - m)*(-1)**(k - mFq1)*pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f1_r0 += (-1)**(k - m)*(-1)**(k - mFq1)*pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;

                except:
                    pass

                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bp += (-1)**(k - m)*(-1)**(k - mFq0)*pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f0_rp += (-1)**(k - m)*(-1)**(k - mFq0)*pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;

                except:
                    pass

                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_bp += (-1)**(k - m)*(-1)**(k - mFq1)*pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f1_rp += (-1)**(k - m)*(-1)**(k - mFq1)*pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;

                except:
                    pass

        amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
        + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
        + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2;

        cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
                        1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
                        1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
                        1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
                        1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
                        1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);

        amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
        + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
        + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;

#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#
    return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

def Raman2Procs_Delta_Fine_Structure(mJq0,mJq1,Jq,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list,f,tot_scat=False,stim_emit=False):
    #Computes the i->f scattering rate for Raman transitions
    #between |mJq0> and |mJq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where 
    #g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates the occupation-probability weighted sum over the qubit states of (A1_ram/Delta+A2_ram/(2*wl+Delta))^2.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #mJq0 = mJ quantum number for |0>
    #
    #mJq1 = mJ quantum number for |1>
    #
    #Jq = J quantum number of qubit manifold
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju_list = list of J quantum number for intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
    #each sublist is in the order 'initial -> intermediate levels matrix dict'
    #first, followed 'intermediate -> final levels matrix dict' second.
    #
    #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
    #first scattering process (-1*detuning in THz); first element in each
    #sublist is the frequency denominator for the standard scattering
    #process, while the second element of the sublist correspons to the often
    #neglected second scattering process frequency denominator
    #
    #f = branching ratio to manifold containing f
    #
    import numpy as np
    
    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    amp1_f1_bm = 0;
    amp1_f1_b0 = 0;
    amp1_f1_bp = 0;
    amp1_f0_rm = 0;
    amp1_f0_r0 = 0;
    amp1_f0_rp = 0;
    amp1_f1_rm = 0; 
    amp1_f1_r0 = 0;  
    amp1_f1_rp = 0;
    
    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;
    amp2_f1_bm = 0;
    amp2_f1_b0 = 0;
    amp2_f1_bp = 0;
    amp2_f0_rm = 0;
    amp2_f0_r0 = 0;
    amp2_f0_rp = 0;
    amp2_f1_rm = 0; 
    amp2_f1_r0 = 0;  
    amp2_f1_rp = 0;

    for n in [-1*Jl + k for k in range(int(2*Jl+1))]:
        
        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        amp1_f1_bm = 0;
        amp1_f1_b0 = 0;
        amp1_f1_bp = 0;
        amp1_f0_rm = 0;
        amp1_f0_r0 = 0;
        amp1_f0_rp = 0;
        amp1_f1_rm = 0; 
        amp1_f1_r0 = 0;  
        amp1_f1_rp = 0;  
        
        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        amp2_f1_bm = 0;
        amp2_f1_b0 = 0;
        amp2_f1_bp = 0;
        amp2_f0_rm = 0;
        amp2_f0_r0 = 0;
        amp2_f0_rp = 0;
        amp2_f1_rm = 0; 
        amp2_f1_r0 = 0;  
        amp2_f1_rp = 0;
        
        for q in range(0,len(Ju_list)):
            Ju = Ju_list[q];
            mat_dicts = mat_dict_list[q]
            Deltas = Delta_list[q];
            
            mat_dict_i = mat_dicts[0];
            mat_dict_f = mat_dicts[1];
            
            Delta_1 = Deltas[0];
            Delta_2 = Deltas[1];
            
            for j in [-1*Ju + k for k in range(int(2*Ju+1))]:
                
                #First scattering process
                try:
                    if j-mJq0==-1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += pols_b[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f0_rm += pols_r[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if j-mJq1==-1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bm += pols_b[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f1_rm += pols_r[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
    
                try:
                    if j-mJq0==0:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f0_rm += pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if j-mJq1==0:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_b0 += pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f1_r0 += pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
    
                try:
                    if j-mJq0==1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bp += pols_b[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f0_rp += pols_r[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if j-mJq1==1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bp += pols_b[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f1_rp += pols_r[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                
                #Second scattering process; swap polarization of scattered and incident light
                try:
                    if n-j==-1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += (-1)**(j - n)*(-1)**(j - mJq0)*pols_b[0+2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f0_rm += (-1)**(j - n)*(-1)**(j - mJq0)*pols_r[0+2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if n-j==-1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bm += (-1)**(j - n)*(-1)**(j - mJq1)*pols_b[0+2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f1_rm += (-1)**(j - n)*(-1)**(j - mJq1)*pols_r[0+2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
    
                try:
                    if n-j==0:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += (-1)**(j - n)*(-1)**(j - mJq0)*pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f0_rm += (-1)**(j - n)*(-1)**(j - mJq0)*pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if n-j==0:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_b0 += (-1)**(j - n)*(-1)**(j - mJq1)*pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f1_r0 += (-1)**(j - n)*(-1)**(j - mJq1)*pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
    
                try:
                    if n-j==1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bp += (-1)**(j - n)*(-1)**(j - mJq0)*pols_b[2-2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f0_rp += (-1)**(j - n)*(-1)**(j - mJq0)*pols_r[2-2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if n-j==1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bp += (-1)**(j - n)*(-1)**(j - mJq1)*pols_b[2-2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f1_rp += (-1)**(j - n)*(-1)**(j - mJq1)*pols_r[2-2*stim_emit]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
        
        amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
        + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
        + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2; 
        
        cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
                        1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
                        1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
                        1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
                        1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
                        1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);
                        
        amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
        + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
        + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;
        
#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#                        
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#            
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#        
    return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

def Raman2Procs_Delta_Print(Fq0,mFq0,Fq1,mFq1,I,Jq,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list,f,g2,tot_scat=False,stim_emit=False):
    #Computes the i->f scattering rate for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where 
    #g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates the occupation-probability weighted sum over the qubit states of (A1_ram/Delta+A2_ram/(2*wl+Delta))^2.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jq = J quantum number of qubit manifold
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju_list = list of J quantum number for intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
    #each sublist is in the order 'initial -> intermediate levels matrix dict'
    #first, followed 'intermediate -> final levels matrix dict' second.
    #
    #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
    #first scattering process (-1*detuning in THz); first element in each
    #sublist is the frequency denominator for the standard scattering
    #process, while the second element of the sublist correspons to the often
    #neglected second scattering process frequency denominator
    #
    #f = branching ratio to manifold containing f
    #
    import numpy as np
    
    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    amp1_f1_bm = 0;
    amp1_f1_b0 = 0;
    amp1_f1_bp = 0;
    amp1_f0_rm = 0;
    amp1_f0_r0 = 0;
    amp1_f0_rp = 0;
    amp1_f1_rm = 0; 
    amp1_f1_r0 = 0;  
    amp1_f1_rp = 0;
    
    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;
    amp2_f1_bm = 0;
    amp2_f1_b0 = 0;
    amp2_f1_bp = 0;
    amp2_f0_rm = 0;
    amp2_f0_r0 = 0;
    amp2_f0_rp = 0;
    amp2_f1_rm = 0; 
    amp2_f1_r0 = 0;  
    amp2_f1_rp = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        amp1_f1_bm = 0;
        amp1_f1_b0 = 0;
        amp1_f1_bp = 0;
        amp1_f0_rm = 0;
        amp1_f0_r0 = 0;
        amp1_f0_rp = 0;
        amp1_f1_rm = 0; 
        amp1_f1_r0 = 0;  
        amp1_f1_rp = 0;  
        
        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        amp2_f1_bm = 0;
        amp2_f1_b0 = 0;
        amp2_f1_bp = 0;
        amp2_f0_rm = 0;
        amp2_f0_r0 = 0;
        amp2_f0_rp = 0;
        amp2_f1_rm = 0; 
        amp2_f1_r0 = 0;  
        amp2_f1_rp = 0;
        
        for q in range(0,len(Ju_list)):
            Ju = Ju_list[q];
            mat_dicts = mat_dict_list[q]
            Deltas = Delta_list[q];
            
            mat_dict_i = mat_dicts[0];
            mat_dict_f = mat_dicts[1];
            
            Delta_1 = Deltas[0];
            Delta_2 = Deltas[1];
            
            for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
                j = vec[0];
                k = vec[1];
                
                #First scattering process
                try:
                    if k-mFq0==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2))
                            amp1_f0_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2))
                except:
                    pass
                
                try:
                    if k-mFq1==-1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2))
                            amp1_f1_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2));
                            
                except:
                    pass
    
                try:
                    if k-mFq0==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2))
                            amp1_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2));
                            
                except:
                    pass
                
                try:
                    if k-mFq1==0:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2))
                            amp1_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2))
                            
                except:
                    pass
    
                try:
                    if k-mFq0==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2));
                            amp1_f0_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2));
                            
                except:
                    pass
                
                try:
                    if k-mFq1==1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2));
                            amp1_f1_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            print('P{}/2 {},{}->{},{} normal amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1)**2));
                            
                except:
                    pass
                
                
                #Second scattering process; swap polarization of scattered and incident light
                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            amp2_f0_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            
                except:
                    pass
                
                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_bm += pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            amp2_f1_rm += pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[0+2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            
    
                except:
                    pass
    
                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            amp2_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            
                except:
                    pass
                
                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            amp2_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            
                except:
                    pass
    
                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            amp2_f0_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            
                except:
                    pass
                
                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_bp += pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_b[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            amp2_f1_rp += pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            print('P{}/2 {},{}->{},{} counter amp squared: {}'.format(2*Ju,j,k,n,m,g2*(pols_r[2-2*stim_emit]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2)**2));
                            
                except:
                    pass
        
        amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
        + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
        + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2; 
        
        cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
                        1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
                        1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
                        1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
                        1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
                        1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);
                        
        amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
        + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
        + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;
        
        if n == 2 & m ==2:
            print('P{}/2 {},{} normal amp squared: {}'.format(int(2*Ju),n,m,g2*(1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
            + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
            + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2)))
            print('P{}/2 {},{} counter amp squared: {}'.format(int(2*Ju),n,m,g2*(1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
            + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
            + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2)))
        
#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#                        
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#            
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#        
    return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

def Raman2Procs_Delta_Fine_Structure(mJq0,mJq1,Jq,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list,f,tot_scat=False):
    #Computes the i->f scattering rate for Raman transitions
    #between |mJq0> and |mJq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where 
    #g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates the occupation-probability weighted sum over the qubit states of (A1_ram/Delta+A2_ram/(2*wl+Delta))^2.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #mJq0 = mJ quantum number for |0>
    #
    #mJq1 = mJ quantum number for |1>
    #
    #Jq = J quantum number of qubit manifold
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju_list = list of J quantum number for intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
    #each sublist is in the order 'initial -> intermediate levels matrix dict'
    #first, followed 'intermediate -> final levels matrix dict' second.
    #
    #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
    #first scattering process (-1*detuning in THz); first element in each
    #sublist is the frequency denominator for the standard scattering
    #process, while the second element of the sublist correspons to the often
    #neglected second scattering process frequency denominator
    #
    #f = branching ratio to manifold containing f
    #
    import numpy as np
    
    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    amp1_f1_bm = 0;
    amp1_f1_b0 = 0;
    amp1_f1_bp = 0;
    amp1_f0_rm = 0;
    amp1_f0_r0 = 0;
    amp1_f0_rp = 0;
    amp1_f1_rm = 0; 
    amp1_f1_r0 = 0;  
    amp1_f1_rp = 0;
    
    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;
    amp2_f1_bm = 0;
    amp2_f1_b0 = 0;
    amp2_f1_bp = 0;
    amp2_f0_rm = 0;
    amp2_f0_r0 = 0;
    amp2_f0_rp = 0;
    amp2_f1_rm = 0; 
    amp2_f1_r0 = 0;  
    amp2_f1_rp = 0;

    for n in [-1*Jl + k for k in range(int(2*Jl+1))]:
        
        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        amp1_f1_bm = 0;
        amp1_f1_b0 = 0;
        amp1_f1_bp = 0;
        amp1_f0_rm = 0;
        amp1_f0_r0 = 0;
        amp1_f0_rp = 0;
        amp1_f1_rm = 0; 
        amp1_f1_r0 = 0;  
        amp1_f1_rp = 0;  
        
        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        amp2_f1_bm = 0;
        amp2_f1_b0 = 0;
        amp2_f1_bp = 0;
        amp2_f0_rm = 0;
        amp2_f0_r0 = 0;
        amp2_f0_rp = 0;
        amp2_f1_rm = 0; 
        amp2_f1_r0 = 0;  
        amp2_f1_rp = 0;
        
        for q in range(0,len(Ju_list)):
            Ju = Ju_list[q];
            mat_dicts = mat_dict_list[q]
            Deltas = Delta_list[q];
            
            mat_dict_i = mat_dicts[0];
            mat_dict_f = mat_dicts[1];
            
            Delta_1 = Deltas[0];
            Delta_2 = Deltas[1];
            
            for j in [-1*Ju + k for k in range(int(2*Ju+1))]:
                
                #First scattering process
                try:
                    if j-mJq0==-1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += pols_b[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f0_rm += pols_r[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if j-mJq1==-1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bm += pols_b[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f1_rm += pols_r[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
    
                try:
                    if j-mJq0==0:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f0_rm += pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if j-mJq1==0:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_b0 += pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f1_r0 += pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
    
                try:
                    if j-mJq0==1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bp += pols_b[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f0_rp += pols_r[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if j-mJq1==1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bp += pols_b[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            amp1_f1_rp += pols_r[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_1;
                            
                except:
                    pass
                
                
                #Second scattering process; swap polarization of scattered and incident light
                try:
                    if n-j==-1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += pols_b[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f0_rm += pols_r[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if n-j==-1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bm += pols_b[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f1_rm += pols_r[0]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
    
                try:
                    if n-j==0:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bm += pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f0_rm += pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if n-j==0:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_b0 += pols_b[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f1_r0 += pols_r[1]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
    
                try:
                    if n-j==1:
                        if [Jl,n] != [Jq,mJq0] or tot_scat:
                            amp1_f0_bp += pols_b[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f0_rp += pols_r[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq0),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if n-j==1:
                        if [Jl,n] != [Jq,mJq1] or tot_scat:
                            amp1_f1_bp += pols_b[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            amp1_f1_rp += pols_r[2]*mat_dict_i['{}/2->{}/2'.format(int(2*mJq1),int(2*j))]*mat_dict_f['{}/2->{}/2'.format(int(2*n),int(2*j))]/Delta_2;
                            
                except:
                    pass
        
        amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
        + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
        + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2; 
        
        cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
                        1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
                        1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
                        1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
                        1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
                        1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);
                        
        amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
        + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
        + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;
        
#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#                        
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#            
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#        
    return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

def ScatteringRateCalc(Fq0,mFq0,I,Jq,Jl,Ju_list,pols_b,mat_dict_list,Delta_list,f,gamma,Pow,alpha,omega32,w0,tot_scat=False):
    #Computes the i->f scattering rate for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
    #
    #In other words, this code calculates 
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jq = J quantum number of qubit manifold
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju_list = list of J quantum number for intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
    #each sublist is in the order 'initial -> intermediate levels matrix dict'
    #first, followed 'intermediate -> final levels matrix dict' second.
    #
    #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
    #first scattering process (-1*detuning in THz); first element in each
    #sublist is the frequency denominator for the standard scattering
    #process, while the second element of the sublist correspons to the often
    #neglected second scattering process frequency denominator
    #
    #f = branching ratio to manifold containing f
    #
    import numpy as np
    
    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    
    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        
        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        
        for q in range(0,len(Ju_list)):
            Ju = Ju_list[q];
            mat_dicts = mat_dict_list[q]
            Deltas = Delta_list[q];
            
            mat_dict_i = mat_dicts[0];
            mat_dict_f = mat_dicts[1];
            
            Delta_1 = Deltas[0];
            Delta_2 = Deltas[1];
            
            for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
                j = vec[0];
                k = vec[1];
                
                #First scattering process
                try:
                    if k-mFq0==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                
                try:
                    if k-mFq0==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                
                try:
                    if k-mFq0==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                
                #Second scattering process; swap polarization of scattered and incident light
                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
                
                
                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
                
        
        amp1_sum += abs(amp1_f0_bm)**2 + \
        + abs(amp1_f0_b0)**2 + \
        + abs(amp1_f0_bp)**2; 
        
        cross_sum += 2*(amp1_f0_bm*amp2_f0_bm + \
                        amp1_f0_b0*amp2_f0_b0 + \
                        amp1_f0_bp*amp2_f0_bp);
                        
        amp2_sum += abs(amp2_f0_bm)**2 + \
        + abs(amp2_f0_b0)**2 + \
        + abs(amp2_f0_bp)**2;
        
#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#                        
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#            
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#

    c = 299792458;
    hbar = hbar = 1.055*1e-34;
    g2_gamma = gamma*3*gamma*c**2*Pow*alpha/(hbar*omega32**3*w0**2)

        
    return [g2_gamma*f*amp1_sum, g2_gamma*f*cross_sum, g2_gamma*f*amp2_sum, g2_gamma*f*(amp1_sum + cross_sum + amp2_sum)]

def Raman2Procs_Delta_Eric(Fq0,mFq0,Fq1,mFq1,I,Jq,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list,f,tot_scat=False):
    #Computes the i->f scattering rate for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>; includes contributions from the typically
    #neglected auxilliary scattering process (see Loudon 2000, chapter 8 on the
    #Kramers-Heisenberg formula). Neglects Rayleigh scattering.
    #
    #In other words, if the Raman scattering rate R_ram is written as 
    #R_ram = gamma*g^2*(A1_ram/Delta+A2_ram/(2*wl+Delta))^2 (where 
    #g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates the occupation-probability weighted sum over the qubit states of (A1_ram/Delta+A2_ram/(2*wl+Delta))^2.
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jq = J quantum number of qubit manifold
    #
    #Jl = J quantum number of final, scattered states
    #
    #Ju_list = list of J quantum number for intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of lists of dicts of transition matrix values from initial -> intermediate states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'. List takes form [[mat_dict_i,mat_dict_f], ...];
    #each sublist is in the order 'initial -> intermediate levels matrix dict'
    #first, followed 'intermediate -> final levels matrix dict' second.
    #
    #Delta_list = [[Delta_1, Delta_2], ...] list of frequency denominators for
    #first scattering process (-1*detuning in THz); first element in each
    #sublist is the frequency denominator for the standard scattering
    #process, while the second element of the sublist correspons to the often
    #neglected second scattering process frequency denominator
    #
    #f = branching ratio to manifold containing f
    #
    import numpy as np
    
    amp1_sum = 0;
    cross_sum = 0;
    amp2_sum = 0;
    amp1_f0_bm = 0;
    amp1_f0_b0 = 0;
    amp1_f0_bp = 0;
    amp1_f1_bm = 0;
    amp1_f1_b0 = 0;
    amp1_f1_bp = 0;
    amp1_f0_rm = 0;
    amp1_f0_r0 = 0;
    amp1_f0_rp = 0;
    amp1_f1_rm = 0; 
    amp1_f1_r0 = 0;  
    amp1_f1_rp = 0;
    
    amp2_f0_bm = 0;
    amp2_f0_b0 = 0;
    amp2_f0_bp = 0;
    amp2_f1_bm = 0;
    amp2_f1_b0 = 0;
    amp2_f1_bp = 0;
    amp2_f0_rm = 0;
    amp2_f0_r0 = 0;
    amp2_f0_rp = 0;
    amp2_f1_rm = 0; 
    amp2_f1_r0 = 0;  
    amp2_f1_rp = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp1_f0_bm = 0;
        amp1_f0_b0 = 0;
        amp1_f0_bp = 0;
        amp1_f1_bm = 0;
        amp1_f1_b0 = 0;
        amp1_f1_bp = 0;
        amp1_f0_rm = 0;
        amp1_f0_r0 = 0;
        amp1_f0_rp = 0;
        amp1_f1_rm = 0; 
        amp1_f1_r0 = 0;  
        amp1_f1_rp = 0;  
        
        amp2_f0_bm = 0;
        amp2_f0_b0 = 0;
        amp2_f0_bp = 0;
        amp2_f1_bm = 0;
        amp2_f1_b0 = 0;
        amp2_f1_bp = 0;
        amp2_f0_rm = 0;
        amp2_f0_r0 = 0;
        amp2_f0_rp = 0;
        amp2_f1_rm = 0; 
        amp2_f1_r0 = 0;  
        amp2_f1_rp = 0;
        
        for q in range(0,len(Ju_list)):
            Ju = Ju_list[q];
            mat_dicts = mat_dict_list[q]
            Deltas = Delta_list[q];
            
            mat_dict_i = mat_dicts[0];
            mat_dict_f = mat_dicts[1];
            
            Delta_1 = Deltas[0];
            Delta_2 = Deltas[1];
            
            for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
                j = vec[0];
                k = vec[1];
                
                #First scattering process
                try:
                    if k-mFq0==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f0_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if k-mFq1==-1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f1_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
    
                try:
                    if k-mFq0==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if k-mFq1==0:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
    
                try:
                    if k-mFq0==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp1_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f0_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                try:
                    if k-mFq1==1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp1_f1_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            amp1_f1_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_1;
                            
                except:
                    pass
                
                
                #Second scattering process; swap polarization of scattered and incident light
                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f0_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if m-k==-1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f1_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
    
                except:
                    pass
    
                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f0_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if m-k==0:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_b0 += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f1_r0 += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
    
                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq0,mFq0] or tot_scat:
                            amp2_f0_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f0_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
                
                try:
                    if m-k==1:
                        if [Jl,n,m] != [Jq,Fq1,mFq1] or tot_scat:
                            amp2_f1_bp += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            amp2_f1_rp += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/Delta_2;
                            
                except:
                    pass
        
        amp1_sum += 1/2*abs(amp1_f0_bm)**2 + 1/2*abs(amp1_f1_bm)**2 + 1/2*abs(amp1_f0_rm)**2 + 1/2*abs(amp1_f1_rm)**2\
        + 1/2*abs(amp1_f0_b0)**2 + 1/2*abs(amp1_f1_b0)**2 + 1/2*abs(amp1_f0_r0)**2 + 1/2*abs(amp1_f1_r0)**2\
        + 1/2*abs(amp1_f0_bp)**2 + 1/2*abs(amp1_f1_bp)**2 + 1/2*abs(amp1_f0_rp)**2 + 1/2*abs(amp1_f1_rp)**2; 
        
        cross_sum += 2*(1/2*amp1_f0_bm*amp2_f0_bm + 1/2*amp1_f0_rm*amp2_f0_rm + \
                        1/2*amp1_f0_b0*amp2_f0_b0 + 1/2*amp1_f0_r0*amp2_f0_r0 + \
                        1/2*amp1_f0_bp*amp2_f0_bp + 1/2*amp1_f0_rp*amp2_f0_rp + \
                        1/2*amp1_f1_bm*amp2_f1_bm + 1/2*amp1_f1_rm*amp2_f1_rm + \
                        1/2*amp1_f1_b0*amp2_f1_b0 + 1/2*amp1_f1_r0*amp2_f1_r0 + \
                        1/2*amp1_f1_bp*amp2_f1_bp + 1/2*amp1_f1_rp*amp2_f1_rp);
                        
        amp2_sum += 1/2*abs(amp2_f0_bm)**2 + 1/2*abs(amp2_f1_bm)**2 + 1/2*abs(amp2_f0_rm)**2 + 1/2*abs(amp2_f1_rm)**2\
        + 1/2*abs(amp2_f0_b0)**2 + 1/2*abs(amp2_f1_b0)**2 + 1/2*abs(amp2_f0_r0)**2 + 1/2*abs(amp2_f1_r0)**2\
        + 1/2*abs(amp2_f0_bp)**2 + 1/2*abs(amp2_f1_bp)**2 + 1/2*abs(amp2_f0_rp)**2 + 1/2*abs(amp2_f1_rp)**2;
        
#    #Get rid of Rayleigh amplitude, if applicable
#    if Jl == 5/2:
#        amp1_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#                        
#        cross_sum -= pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                     pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#            
#        amp2_sum -= 1/2*pols_b[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2) +\
#                    1/2*pols_r[1]**2*((mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq0,mFq0,Fq1,mFq1)][0])**2 + (mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0]*mat_dict_f['{},{}->{},{}'.format(Fq1,mFq1,Fq1,mFq1)][0])**2)
#        
    return [f*amp1_sum, f*cross_sum, f*amp2_sum, f*(amp1_sum + cross_sum + amp2_sum)]

def Rayleigh_Sum_Uys(Fq0,mFq0,Fq1,mFq1,I,Ju,pols_b,pols_r,mat_dict,f):
    #Computes the scattering amplitude (Uys style) back into qubit manifold for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #In other words, if the Rayleigh scattering rate R_ray is written as 
    #R_ray = gamma*g^2*A_ray/Delta^2 (where g = E*mu/2*hbar, with mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates A_ray.
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #
    #Fq0 = F quantum numver for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum numver for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Ju = J quantum number of intermediate state
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #f = branching ratio for decay from intermediate manifold to qubit manifold
    
    amp_f0_bm = 0;
    amp_f0_b0 = 0;
    amp_f0_bp = 0;
    amp_f1_bm = 0;
    amp_f1_b0 = 0;
    amp_f1_bp = 0;
    amp_f0_rm = 0;
    amp_f0_r0 = 0;
    amp_f0_rp = 0;
    amp_f1_rm = 0; 
    amp_f1_r0 = 0;  
    amp_f1_rp = 0;        
    
    for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
        j = vec[0];
        k = vec[1];
        
        try:
            if k-mFq0==-1:
                amp_f0_bm += pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
                amp_f0_rm += pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
        except:
                pass
            
        try:
            if k-mFq1==-1:
                amp_f1_bm += pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
                amp_f1_rm += pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
        except:
            pass
        
        try:
            if k-mFq0==0:
                amp_f0_b0 += pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
                amp_f0_r0 += pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
        except:
            pass
            
        try:
            if k-mFq1==0:
                amp_f1_b0 += pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
                amp_f1_r0 += pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
        except:
            pass
        
        try:
            if k-mFq0==1:
                amp_f0_bp += pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
                amp_f0_rp += pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
        except:
            pass
            
        try:
            if k-mFq1==1:
                amp_f1_bp += pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
                amp_f1_rp += pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
        except:
            pass      
 
    amp_sum = abs(amp_f0_bm - amp_f1_bm)**2 + abs(amp_f0_rm - amp_f1_rm)**2\
    + abs(amp_f0_b0 - amp_f1_b0)**2 + abs(amp_f0_r0 - amp_f1_r0)**2\
    + abs(amp_f0_bp - amp_f1_bp)**2 + abs(amp_f0_rp - amp_f1_rp)**2;
           
    return f*amp_sum

def Rayleigh_Sum(Fq0,mFq0,Fq1,mFq1,I,Ju,pols_b,pols_r,mat_dict,f):
    #Computes the scattering amplitude back into qubit manifold for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #In other words, if the Rayleigh scattering rate R_ray is written as 
    #R_ray = gamma*g^2*A_ray/Delta^2 (where g = E*mu/2*hbar, with mu the 
    #normalization of mat_dict_i elements, gamma the linewidth of the intermediate
    #manifold, and Delta the detuning from the intermediate manifold), this code 
    #calculates A_ray.
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering, as the amplitudes to scatter from two
    #intermediate manifolds must be added before squaring.
    #
    #
    #Fq0 = F quantum numver for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum numver for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Ju = J quantum number of intermediate state
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #f = branching ratio for decay from intermediate manifold to qubit manifold
    
    amp_f0_bm = 0;
    amp_f0_b0 = 0;
    amp_f0_bp = 0;
    amp_f1_bm = 0;
    amp_f1_b0 = 0;
    amp_f1_bp = 0;
    amp_f0_rm = 0;
    amp_f0_r0 = 0;
    amp_f0_rp = 0;
    amp_f1_rm = 0; 
    amp_f1_r0 = 0;  
    amp_f1_rp = 0;        
    
    for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
        j = vec[0];
        k = vec[1];
        
        try:
            if k-mFq0==-1:
                amp_f0_bm += pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
                amp_f0_rm += pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
        except:
                pass
            
        try:
            if k-mFq1==-1:
                amp_f1_bm += pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
                amp_f1_rm += pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
        except:
            pass
        
        try:
            if k-mFq0==0:
                amp_f0_b0 += pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
                amp_f0_r0 += pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
        except:
            pass
            
        try:
            if k-mFq1==0:
                amp_f1_b0 += pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
                amp_f1_r0 += pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
        except:
            pass
        
        try:
            if k-mFq0==1:
                amp_f0_bp += pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
                amp_f0_rp += pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0];
        except:
            pass
            
        try:
            if k-mFq1==1:
                amp_f1_bp += pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
                amp_f1_rp += pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
        except:
            pass      
 
    amp_sum = 1/2*abs(amp_f0_bm)**2 + 1/2*abs(amp_f1_bm)**2 + 1/2*abs(amp_f0_rm)**2 + 1/2*abs(amp_f1_rm)**2\
    + 1/2*abs(amp_f0_b0)**2 + 1/2*abs(amp_f1_b0)**2 + 1/2*abs(amp_f0_r0)**2 + 1/2*abs(amp_f1_r0)**2\
    + 1/2*abs(amp_f0_bp)**2 + 1/2*abs(amp_f1_bp)**2 + 1/2*abs(amp_f0_rp)**2 + 1/2*abs(amp_f1_rp)**2;
           
    return f*amp_sum

def Rabi_Sum(Fq0,mFq0,Fq1,mFq1,I,Jl,Ju,pols_b,pols_r,mat_dict):
    #Computes the Rabi amplitude back for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #In other words, if the Rabi freq. O_R is written as 
    #O_R = g^2*A_R/Delta (where g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict elements, and Delta the detuning from the 
    #intermediate manifold), this code calculates A_R.
    #
    #Code works for calculating Rabi amplitude due to one intermediate level.
    #
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of qubit states
    #
    #Ju = J quantum number of intermediate state
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    
    amp_rabi = 0;
    
    for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
        j = vec[0];
        k = vec[1];
        
        if k-mFq0==-1:
            try:
                amp_rabi += pols_b[0]*pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
            except:
                pass
        
        if k-mFq0==0:
            try:
                amp_rabi += pols_b[1]*pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
            except:
                pass
            
        if k-mFq0==1:
            try:
                amp_rabi += pols_b[2]*pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0];
            except:
                pass  
    
    return amp_rabi

def Rabi_Sum_higher(Fq0,mFq0,Fq1,mFq1,I,Jl,Ju_list,pols_b,pols_r,mat_dict_list,Delta_list):
    #Computes the Rabi amplitude back for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1> including the effect of higher manifolds
    #
    #In other words, if the Rabi freq. O_R is written as 
    #O_R = g^2*A_R/Delta (where g = E*mu/2*hbar, with E the electric field, mu the 
    #normalization of mat_dict elements, and Delta the detuning from the 
    #intermediate manifold), this code calculates A_R.
    #
    #Code works for calculating Rabi amplitude due to one intermediate level.
    #
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of qubit states
    #
    #Ju_list = list of J quantum numbers of intermediate states
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #mat_dict_list = list of dicts of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    
    amp_rabi = 0;
    
    for i in range(0,len(Ju_list)):
        
        Ju = Ju_list[i];
        mat_dict = mat_dict_list[i];
        Delta = Delta_list[i];
        
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            if k-mFq0==-1:
                try:
                    amp_rabi += pols_b[0]*pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]/Delta;
                except:
                    pass
            
            if k-mFq0==0:
                try:
                    amp_rabi += pols_b[1]*pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]/Delta;
                except:
                    pass
                
            if k-mFq0==1:
                try:
                    amp_rabi += pols_b[2]*pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]/Delta;
                except:
                    pass  
    
    return amp_rabi

def SymbolicAmps(Fq0,mFq0,Fq1,mFq1,I,Jl,pols_b,pols_r,levels):
    #Computes the scattering amplitude back into qubit manifold for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #
    #In general cases, this code will only work where only one intermediate
    #manifold is relevant to scattering.
    #
    #Fq0 = F quantum number for |0>
    #
    #mFq0 = mF quantum number for |0>
    #
    #Fq1 = F quantum number for |1>
    #
    #mFq1 = mF quantum number for |1>
    #
    #I = nuclear spin
    #
    #Jl = J quantum number of states to which system scatters
    #
    #q = polarization type (1, 0, or -1)
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #levels = [[Ju1,mat_dicti1,mat_dictf1,delta1],[Ju2,mat_dicti2,mat_dictf2,delta2],...]
    #
    #Jl = J quantum number of qubit states
    #
    #Ju = J quantum number of intermediate state
    #
    #mat_dicti = dict of transition matrix values for initial to excited states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #mat_dictf = dict of transition matrix values for final to excited states; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'    
    #
    #delta = sympy symbols for detuning
    
    from sympy import simplify
    
    amp_sum = 0;
    amp_f0_bm = 0;
    amp_f0_b0 = 0;
    amp_f0_bp = 0;
    amp_f1_bm = 0;
    amp_f1_b0 = 0;
    amp_f1_bp = 0;
    amp_f0_rm = 0;
    amp_f0_r0 = 0;
    amp_f0_rp = 0;
    amp_f1_rm = 0; 
    amp_f1_r0 = 0;  
    amp_f1_rp = 0;  
    amp_rabi = 0;

    for i in [[x,y] for x in range(int(abs(I-Jl)),int(I+Jl+1)) for y in range(-x,x+1)]:
        n = i[0];
        m = i[1];
        
        amp_f0_bm = 0;
        amp_f0_b0 = 0;
        amp_f0_bp = 0;
        amp_f1_bm = 0;
        amp_f1_b0 = 0;
        amp_f1_bp = 0;
        amp_f0_rm = 0;
        amp_f0_r0 = 0;
        amp_f0_rp = 0;
        amp_f1_rm = 0; 
        amp_f1_r0 = 0;  
        amp_f1_rp = 0;      
        
        for level in levels:
            Ju = level[0];
            mat_dict_i = level[1];
            mat_dict_f = level[2];
            delta = level[3];
            
            for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
                j = vec[0];
                k = vec[1];
                
                try:
                    if k-mFq0==-1 and [n,m] != [Fq0,mFq0]:
                        amp_f0_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                        amp_f0_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                except:
                    pass
                
                try:
                    if k-mFq1==-1 and [n,m] != [Fq1,mFq1]:
                        amp_f1_bm += pols_b[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                        amp_f1_rm += pols_r[0]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                except:
                    pass
    
                try:
                    if k-mFq0==0 and [n,m] != [Fq0,mFq0]:
                        amp_f0_bm += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                        amp_f0_rm += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                except:
                    pass
                
                try:
                    if k-mFq1==0 and [n,m] != [Fq1,mFq1]:
                        amp_f1_bm += pols_b[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                        amp_f1_rm += pols_r[1]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                except:
                    pass
    
                try:
                    if k-mFq0==1 and [n,m] != [Fq0,mFq0]:
                        amp_f0_bm += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                        amp_f0_rm += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                except:
                    pass
                
                try:
                    if k-mFq1==1 and [n,m] != [Fq1,mFq1]:
                        amp_f1_bm += pols_b[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                        amp_f1_rm += pols_r[2]*mat_dict_i['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]*mat_dict_f['{},{}->{},{}'.format(n,m,j,k)][0]/delta;
                except:
                    pass            

        amp_sum += 1/2*(amp_f0_bm)**2 + 1/2*(amp_f1_bm)**2 + 1/2*(amp_f0_rm)**2 + 1/2*(amp_f1_rm)**2\
        + 1/2*(amp_f0_b0)**2 + 1/2*(amp_f1_b0)**2 + 1/2*(amp_f0_r0)**2 + 1/2*(amp_f1_r0)**2\
        + 1/2*(amp_f0_bp)**2 + 1/2*(amp_f1_bp)**2 + 1/2*(amp_f0_rp)**2 + 1/2*(amp_f1_rp)**2;
    
    for level in levels:
        Ju = level[0];
        mat_dict = level[1];
        delta = level[3];
            
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            if k-mFq0==-1:
                try:
                    amp_rabi += pols_b[0]*pols_r[k-mFq1+1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]/delta;
                    #print('{},{}->{},{};{},{}->{},{}:'.format(Fq0,mFq0,j,k,Fq1,mFq1,j,k),pols_b[0]*pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])
                except:
                    pass
            
            if k-mFq0==0:
                try:
                    amp_rabi += pols_b[1]*pols_r[k-mFq1+1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]/delta;
                    #print('{},{}->{},{};{},{}->{},{}:'.format(Fq0,mFq0,j,k,Fq1,mFq1,j,k),pols_b[1]*pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])
                except:
                    pass
                
            if k-mFq0==1:
                try:
                    amp_rabi += pols_b[2]*pols_r[k-mFq1+1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0]/delta;
                    #print('{},{}->{},{};{},{}->{},{}:'.format(Fq0,mFq0,j,k,Fq1,mFq1,j,k),pols_b[2]*pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])
                except:
                    pass        

    from sympy import preorder_traversal
    from sympy import Float
    
    amp_sum_rounded = simplify(amp_sum)
    amp_rabi_rounded = simplify(amp_rabi)       
    
    #Hacky solution since sympy is weird about round(). The following code
    #neatly rounds all floats in the expressions to 4 digits.
    #
    #Only works if you repeat the code twice for some reason.
    
    for a in preorder_traversal(amp_sum_rounded):
        if isinstance(a, Float):
            amp_sum_rounded = amp_sum_rounded.subs(a, round(a, 4))

    for a in preorder_traversal(amp_rabi_rounded):
        if isinstance(a, Float):
            amp_rabi_rounded = amp_rabi_rounded.subs(a, round(a, 4))    
            
    for a in preorder_traversal(amp_sum_rounded):
        if isinstance(a, Float):
            amp_sum_rounded = amp_sum_rounded.subs(a, round(a, 4))

    for a in preorder_traversal(amp_rabi_rounded):
        if isinstance(a, Float):
            amp_rabi_rounded = amp_rabi_rounded.subs(a, round(a, 4))    
    
    return [amp_sum_rounded/amp_rabi_rounded**2, amp_sum_rounded, amp_rabi_rounded]

def SymbolicTotalAmp(Fq0,mFq0,Fq1,mFq1,I,Jl,pols_b,pols_r,levels):
    #Computes the total scattering amplitude for Raman transitions
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #
    #Fq0 = highest value of F for qubit states
    #
    #mFq0 = mF of qubit states
    #
    #I = nuclear spin
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #levels = [[Ju1,mat_dict1,delta1],[Ju2,mat_dict2,delta2],...]; list
    #characterizing intermediate levels
    #
    #Jl = J quantum number of qubit manifold
    #
    #Ju = J quantum number of intermediate state
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #delta = sympy symbols for detuning    
    
    
    amp_tot0_bm = 0;
    amp_tot0_b0 = 0;
    amp_tot0_bp = 0;
    amp_tot1_bm = 0;
    amp_tot1_b0 = 0;
    amp_tot1_bp = 0;
    amp_tot0_rm = 0;
    amp_tot0_r0 = 0;
    amp_tot0_rp = 0;
    amp_tot1_rm = 0; 
    amp_tot1_r0 = 0;  
    amp_tot1_rp = 0;      

    
    for level in levels:
        Ju = level[0];
        mat_dict = level[1];
        delta = level[2];
        
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            try:
                if k-mFq0==-1:
                    amp_tot0_bm += (pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta**2;
                    amp_tot0_rm += (pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta**2;
            except:
                pass
    
            try:
                if k-mFq0==0:
                    amp_tot0_b0 += (pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta**2;
                    amp_tot0_r0 += (pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta**2;
            except:
                pass 
    
            try:
                if k-mFq0==1:
                    amp_tot0_bp += (pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta**2;
                    amp_tot0_rp += (pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta**2;
            except:
                pass
    
            try:
                if k-mFq1==-1:
                    amp_tot1_bm += (pols_b[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta**2;
                    amp_tot1_rm += (pols_r[0]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta**2;
            except:
                pass
    
            try:
                if k-mFq1==0:
                    amp_tot1_b0 += (pols_b[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta**2;
                    amp_tot1_r0 += (pols_r[1]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta**2;
            except:
                pass 
    
            try:
                if k-mFq1==1:
                    amp_tot1_bp += (pols_b[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta**2;
                    amp_tot1_rp += (pols_r[2]*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta**2;
            except:
                pass
    
    amp_total = 1/2*(amp_tot0_bm + amp_tot0_b0 + amp_tot0_bp + amp_tot0_rm + amp_tot0_r0 + amp_tot0_rp) + \
    1/2*(amp_tot1_bm + amp_tot1_b0 + amp_tot1_bp + amp_tot1_rm + amp_tot1_r0 + amp_tot1_rp);
    
    from sympy import simplify
    from sympy import preorder_traversal  
    from sympy import Float      

    amp_total_rounded = simplify(amp_total)
    
    for a in preorder_traversal(amp_total_rounded):
        if isinstance(a, Float):
            amp_total_rounded = amp_total_rounded.subs(a, round(a, 4))

    for a in preorder_traversal(amp_total_rounded):
        if isinstance(a, Float):
            amp_total_rounded = amp_total_rounded.subs(a, round(a, 4))
    
    return amp_total_rounded

def LightShiftCalc(Fq0,mFq0,Fq1,mFq1,I,Jl,levels):
    #Computes the difference in lightshifts
    #between |Fq0,mFq0> and |Fq1,mFq1>
    #
    #
    #Fq0 = highest value of F for qubit states
    #
    #mFq0 = mF of qubit states
    #
    #I = nuclear spin
    #
    #pols_i = list of fractional instensities in order [-,0,+] in beam i
    #
    #levels = [[Ju1,mat_dict1,delta1],[Ju2,mat_dict2,delta2],...]; list
    #characterizing intermediate levels
    #
    #Jl = J quantum number of qubit manifold
    #
    #Ju = J quantum number of intermediate state
    #
    #mat_dict = dict of transition matrix values; 
    #names must be in the format 'F_i,mF_i->F_f,mF_f'
    #
    #delta = sympy symbols for detuning    
    
    
    amp_tot0_bm = 0;
    amp_tot0_b0 = 0;
    amp_tot0_bp = 0;
    amp_tot1_bm = 0;
    amp_tot1_b0 = 0;
    amp_tot1_bp = 0;
    amp_tot0_rm = 0;
    amp_tot0_r0 = 0;
    amp_tot0_rp = 0;
    amp_tot1_rm = 0; 
    amp_tot1_r0 = 0;  
    amp_tot1_rp = 0;      

    from sympy import symbols
    
    bm, b0, bp = symbols('bm b0 bp');
    rm, r0, rp = symbols('rm r0 rp');
    ω0 = symbols('ω0');
    
    for level in levels:
        Ju = level[0];
        mat_dict = level[1];
        delta = level[2];
        
        for vec in [[a,b] for a in range(int(abs(I-Ju)),int(I+Ju+1)) for b in range(-a,a+1)]:
            j = vec[0];
            k = vec[1];
            
            try:
                if k-mFq0==-1:
                    amp_tot0_bm += (bm*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta;
                    amp_tot0_rm += (rm*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/(delta - ω0);
            except:
                pass
    
            try:
                if k-mFq0==0:
                    amp_tot0_b0 += (b0*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta;
                    amp_tot0_r0 += (r0*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/(delta - ω0);
            except:
                pass 
    
            try:
                if k-mFq0==1:
                    amp_tot0_bp += (bp*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/delta;
                    amp_tot0_rp += (rp*mat_dict['{},{}->{},{}'.format(Fq0,mFq0,j,k)][0])**2/(delta - ω0);
            except:
                pass
    
            try:
                if k-mFq1==-1:
                    amp_tot1_bm += (bm*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/(delta + ω0);
                    amp_tot1_rm += (rm*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta;
            except:
                pass
    
            try:
                if k-mFq1==0:
                    amp_tot1_b0 += (b0*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/(delta + ω0);
                    amp_tot1_r0 += (r0*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta;
            except:
                pass 
    
            try:
                if k-mFq1==1:
                    amp_tot1_bp += (bp*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/(delta + ω0);
                    amp_tot1_rp += (rp*mat_dict['{},{}->{},{}'.format(Fq1,mFq1,j,k)][0])**2/delta;
            except:
                pass
    
    amp_total = symbols('g')**2*(amp_tot1_bm + amp_tot1_b0 + amp_tot1_bp + amp_tot1_rm + amp_tot1_r0 + amp_tot1_rp) -\
    symbols('g')**2*(amp_tot0_bm + amp_tot0_b0 + amp_tot0_bp + amp_tot0_rm + amp_tot0_r0 + amp_tot0_rp)
    
    
    from sympy import simplify
    from sympy import preorder_traversal  
    from sympy import Float      

#    amp_total_rounded = simplify(amp_total)
#    
#    for a in preorder_traversal(amp_total_rounded):
#        if isinstance(a, Float):
#            amp_total_rounded = amp_total_rounded.subs(a, round(a, 4))
#
#    for a in preorder_traversal(amp_total_rounded):
#        if isinstance(a, Float):
#            amp_total_rounded = amp_total_rounded.subs(a, round(a, 4))
    
    return amp_total

def LightShiftWesCalc_m(eta, tau, omega, omega_0, K, Delta):
    #Calculates the single-beam, absolute, differential light shift for a two-qubit gate in g-types
    #including Wes's terms
    #
    #eta = Lamb-Dicke parameter of system
    #
    #tau = gate time in seconds
    #
    #omega = P <-> D transition frequency
    #
    #K = number of completed loops in phase space during gate
    #
    #Delta = detuning
    
    import math
    
    Delta_LS = math.pi*math.sqrt(K)/(2*eta*tau)/(-1/(Delta)+1/(Delta + 2*omega))*\
               (omega_0/(Delta*(Delta - omega_0)) + omega_0/((Delta + 2*omega)*(Delta + 2*omega - omega_0)))
    
    return abs(Delta_LS)

def LightShiftWesCalc_g(eta, tau, omega, omega_0, omega_f, K, Delta, simple=False):
    #Calculates the single-beam, absolute, differential light shift for a two-qubit gate in g-types
    #including Wes's terms
    #
    #eta = Lamb-Dicke parameter of system
    #
    #tau = gate time in seconds
    #
    #omega = P <-> D transition frequency
    #
    #K = number of completed loops in phase space during gate
    #
    #Delta = detuning
    
    import math
    
    if simple:
        Delta_LS = math.pi*math.sqrt(K)/(8*eta*tau)*\
                   (omega_0*Delta/((omega_f)*(Delta + omega_f + omega_0)) + 2*omega_0*(Delta + omega_f)/((omega_f)*(Delta + omega_0)))
                   
    
    if not simple:
        Delta_LS = math.pi*math.sqrt(K)/(8*eta*tau)/(omega_f/(Delta*(Delta + omega_f))+omega_f/((Delta + 2*omega)*(Delta + 2*omega - omega_f )))*\
                   (omega_0/((Delta + omega_f)*(Delta + omega_f + omega_0)) + 2*omega_0/((Delta)*(Delta + omega_0))+\
                   omega_0/((Delta - omega_f + 2*omega)*(Delta - omega_f - omega_0 + 2*omega)) + 2*omega_0/((Delta + 2*omega)*(Delta - omega_0 + 2*omega)))
    
    return abs(Delta_LS)

def mu(Ju,Jl,Lu,Ll):
    #Compute normalization factor mu for matrix elements. Lu is the orbital
    #angular momentum quantum number of the upper state and Ju is the combined
    #orbital ang. mom. and spin for the upper state; Jl and Ll are similarly
    #defined for the lower state.
    #
    #J_u = J quantum number for upper state
    #J_l = same, for lower state
    #L_u = L quantum number for upper state
    #L_l = same, for lower state
    
    import math
    from sympy.physics.wigner import wigner_6j
    
    mu = math.sqrt((2*Jl+1)/(2*Ju+1))*(-1)**(Ju+Ll+1+1/2)*math.sqrt(2*Ju+1)*math.sqrt(2*Ll+1)*float(wigner_6j(Ll,Lu,1, Ju,Jl,1/2));
    
    return mu

def EleList(Fmax, mat_dict):
    #Gets list of the six relevant matrix elements for metastable qubits.
    #
    #Fmax = highest value of F for D5/2 manifold of the ion
    #mat_dict = dict of matrix values for the ion; names must be in the format
    #'F_i,mF_i->F_f,mF_f'    
    
    EleList = [mat_dict['{},{}->{},{}'.format(i,Fmax-1,j,k)][0] for i in [Fmax, Fmax-1] for j, k in zip([Fmax-1, Fmax-1, Fmax-2],[Fmax-1, Fmax-2, Fmax-2])]
    
    return EleList

def NegEleList(Fmax, mat_dict):
    #Gets list of the six relevant matrix elements for metastable qubits.
    #
    #Fmax = highest value of F for D5/2 manifold of the ion
    #mat_dict = dict of matrix values for the ion; names must be in the format
    #'F_i,mF_i->F_f,mF_f'    
    
    EleList = [mat_dict['{},{}->{},{}'.format(i,-(Fmax-1),j,-k)][0] for i in [Fmax, Fmax-1] for j, k in zip([Fmax-1, Fmax-1, Fmax-2],[Fmax-1, Fmax-2, Fmax-2])]
    
    return EleList

def Pow1q(ratio,err,lam,w32,w0,Or,br):
    #Gives power in Watts required to achieve certain Raman scattering 
    #probability into D5/2
    #
    #ratio = ratio of Raman scattering amplitude to the square of the Rabi
    #amplitude. So if Raman scattering prob. is (gamma*g^2/delta)*A and Rabi
    #freq. is (g^2/delta)*B, ratio = A/B^2
    #
    #err = probability of Raman scattering into D5/2 during a pi-rotation
    #
    #lam = wavelength of D5/2->P3/2 transition in meters
    #
    #w32 = frequency of D5/2->P3/2 transition in rad/s
    #
    #w0 = beam waist in meters
    #
    #Or = Rabi frequency in rad/s
    
    import math
    
    hbar = 1.055*1e-34;
    
    Pow = ratio*math.pi/(6*err*br)*((2*math.pi*w0/(lam))**2)*hbar*w32*Or
    return Pow

def Pow2q(ratio,err,lam,w32,w0,Or,K,eta,br):
    #Gives power in Watts required to achieve Raman scattering probabilty err for a two-
    #qubit phase gate. Same parameters as Pow1q, but K is the number of circles
    #made in phase space during the gate and eta is the Lamb-Dicke parameter.
    
    import math
    
    hbar = 1.055*1e-34;
    
    Pow = Pow1q(ratio,err,lam,w32,w0,Or,br)*4*math.sqrt(K)/eta
    
    return Pow

def Pow1qPlots(Ratios,Freqs,BRs):
    
    #Makes plots of single qubit gate power requirements for given error probs
    #
    #Ratios = list of Raman/Rabi^2 ratios; if Raman scattering prob. is 
    #(gamma*g^2/delta)*A and Rabi freq. is (g^2/delta)*B, then Ratios is the
    #list of terms of the form ratio = A/B^2.
    #
    #Freqs = list of transition frequencies for the different ion species
    #
    #Ratios list and Freqs list must have the order [Ca,Sr,Ba133,Ba135,Ba137]
    
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    Err = np.linspace(1e-4,1e-2,1000);
    
    plt.loglog(Err, Pow1q(Ratios[0],Err,2*math.pi*3e8/(Freqs[0]),Freqs[0],20*1e-6,0.25*1e6*2*math.pi,BRs[0]), label='Ca43')
    plt.loglog(Err, Pow1q(Ratios[1],Err,2*math.pi*3e8/(Freqs[1]),Freqs[1],20*1e-6,0.25*1e6*2*math.pi,BRs[1]), label='Sr87')   
    plt.loglog(Err, Pow1q(Ratios[2],Err,2*math.pi*3e8/(Freqs[2]),Freqs[2],20*1e-6,0.25*1e6*2*math.pi,BRs[2]), label='Ba133')
    plt.loglog(Err, Pow1q(Ratios[3],Err,2*math.pi*3e8/(Freqs[3]),Freqs[3],20*1e-6,0.25*1e6*2*math.pi,BRs[3]), label='Ba135')
    plt.loglog(Err, Pow1q(Ratios[4],Err,2*math.pi*3e8/(Freqs[4]),Freqs[4],20*1e-6,0.25*1e6*2*math.pi,BRs[4]), label='Ba137')
    
    plt.xlabel('Gate Error Probability')
    plt.ylabel('Power (Watts)')
    plt.legend()
    
    return

def Pow2qPlots(Masses,Ratios,Freqs,BRs):
    #Makes plots of two qubit gate power requirements for given error probs
    #
    #Ratios = list of Raman/Rabi^2 ratios; if Raman scattering prob. is 
    #(gamma*g^2/delta)*A and Rabi freq. is (g^2/delta)*B, then Ratios is the
    #list of terms of the form ratio = A/B^2.
    #
    #Freqs = list of transition frequencies for the different ion species
    #
    #Ratios list and Freqs list must have the order [Ca,Sr,Ba133,Ba135,Ba137]
    
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    Err = np.linspace(1e-4,1e-2,1000);
    
    hbar = 1.055*1e-34;
    wtrap = 2*math.pi*5e6; #Trap frequency
    
    etas = []; #Empty list, to be populated with Lamb-Dicke parameters
    
    for i in range(len(Freqs)):
        dk = math.sqrt(2)*Freqs[i]/(3e8); #delta k for perpendicular beams
        z0 = math.sqrt(hbar/(4*Masses[i]*wtrap));
        
        etas.append(dk*z0)
    
    plt.loglog(Err, Pow2q(Ratios[0],Err,2*math.pi*3e8/(Freqs[0]),Freqs[0],20*1e-6,0.25*1e6*2*math.pi,1,etas[0],BRs[0]), label='Ca43')
    plt.loglog(Err, Pow2q(Ratios[1],Err,2*math.pi*3e8/(Freqs[1]),Freqs[1],20*1e-6,0.25*1e6*2*math.pi,1,etas[1],BRs[1]), label='Sr87')   
    plt.loglog(Err, Pow2q(Ratios[2],Err,2*math.pi*3e8/(Freqs[2]),Freqs[2],20*1e-6,0.25*1e6*2*math.pi,1,etas[2],BRs[2]), label='Ba133')
    plt.loglog(Err, Pow2q(Ratios[3],Err,2*math.pi*3e8/(Freqs[3]),Freqs[3],20*1e-6,0.25*1e6*2*math.pi,1,etas[3],BRs[3]), label='Ba135')
    plt.loglog(Err, Pow2q(Ratios[4],Err,2*math.pi*3e8/(Freqs[4]),Freqs[4],20*1e-6,0.25*1e6*2*math.pi,1,etas[4],BRs[4]), label='Ba137')
    
    plt.xlabel('Two-Qubit Gate Error Probability')
    plt.ylabel('Power (Watts)')
    plt.legend()
    
    return


def Pow2qtau(ratio,err,lam,w32,w0,Or,K,eta,tau,br):
    #Gives power in Watts required to achieve Raman scattering probabilty err for a two-
    #qubit phase gate. Same parameters as Pow1q, but K is the number of circles
    #made in phase space during the gate and eta is the Lamb-Dicke parameter.
    
    import math
    
    hbar = 1.055*1e-34;
    
    Pow = ratio*math.pi/(6*err*br)*((2*math.pi*w0/(lam))**2)*hbar*w32*math.pi/(2*tau)*4*K/eta**2
    
    return Pow

def Pow2qtauTrue(ratio,err,lam,w32,w0,Or,K,eta,tau,br):
    #Gives power in Watts required to achieve Raman scattering probabilty err for a two-
    #qubit phase gate. Same parameters as Pow1q, but K is the number of circles
    #made in phase space during the gate and eta is the Lamb-Dicke parameter.
    
    import math
    
    hbar = 1.055*1e-34;
    
    Pow = ratio*math.pi/(6*err*br)*((2*math.pi*w0/(lam))**2)*hbar*w32*math.pi/(2*tau)*4*K/eta**2
    
    return Pow

def Pow2qPlotstau(Masses,Ratios,Freqs,BRs,tau):
    #Makes plots of two qubit gate power requirements for given error probs
    #
    #Ratios = list of Raman/Rabi^2 ratios; if Raman scattering prob. is 
    #(gamma*g^2/delta)*A and Rabi freq. is (g^2/delta)*B, then Ratios is the
    #list of terms of the form ratio = A/B^2.
    #
    #Freqs = list of transition frequencies for the different ion species
    #
    #Ratios list and Freqs list must have the order [Ca,Sr,Ba133,Ba135,Ba137]
    
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    Err = np.linspace(1e-4,1e-2,1000);
    
    hbar = 1.055*1e-34;
    wtrap = 2*math.pi*5e6; #Trap frequency
    
    etas = []; #Empty list, to be populated with Lamb-Dicke parameters
    
    for i in range(len(Freqs)):
        dk = math.sqrt(2)*Freqs[i]/(3e8); #delta k for perpendicular beams
        z0 = math.sqrt(hbar/(4*Masses[i]*wtrap));
        
        etas.append(dk*z0)
    
    plt.loglog(Err, Pow2qtau(Ratios[0],Err,2*math.pi*3e8/(Freqs[0]),Freqs[0],20*1e-6,0.25*1e6*2*math.pi,1,etas[0],tau,BRs[0]), label='Ca43')
    plt.loglog(Err, Pow2qtau(Ratios[1],Err,2*math.pi*3e8/(Freqs[1]),Freqs[1],20*1e-6,0.25*1e6*2*math.pi,1,etas[1],tau,BRs[1]), label='Sr87')   
    plt.loglog(Err, Pow2qtau(Ratios[2],Err,2*math.pi*3e8/(Freqs[2]),Freqs[2],20*1e-6,0.25*1e6*2*math.pi,1,etas[2],tau,BRs[2]), label='Ba133')
    plt.loglog(Err, Pow2qtau(Ratios[3],Err,2*math.pi*3e8/(Freqs[3]),Freqs[3],20*1e-6,0.25*1e6*2*math.pi,1,etas[3],tau,BRs[3]), label='Ba135')
    plt.loglog(Err, Pow2qtau(Ratios[4],Err,2*math.pi*3e8/(Freqs[4]),Freqs[4],20*1e-6,0.25*1e6*2*math.pi,1,etas[4],tau,BRs[4]), label='Ba137')
    
    plt.xlabel('Two-Qubit Gate Error Probability')
    plt.ylabel('Power (Watts)')
    plt.legend()
    
    return

def OneQErrorCalculator(err,gamma,A_tot,A_R,w_trans):
    #Calculates laser wavelength (blue or red detuned) needed to achieve given
    #error probability
    import math
    
    #Get requisite detuning
    Delta = math.pi*gamma*A_tot/(2*abs(A_R)*2*math.pi*err)
    
    #Compute laser wavelength
    #[blue-detuned wavelength, red-detuned wavelength]
    las_wav = [3e8/(w_trans/(2*math.pi) - Delta),3e8/(w_trans/(2*math.pi) + Delta)]
    
    return las_wav

def TwoQErrorCalculator(err,gamma,A_tot,A_R,eta,w_trans):
    #Calculates laser wavelength (blue or red detuned) needed to achieve given
    #error probability
    from sympy.solvers import solve
    from sympy import Symbol
    
    import math
    
    #Get requisite detuning
    Delta_r  = Symbol('Delta_r', real=True, positive=True)
    Delta_b  = Symbol('Delta_b', real=True, positive=True)
    
    Delta_red = solve(abs(Delta_r) - math.pi*gamma*A_tot/(2*abs(A_R)*2*math.pi*err)*4/eta\
          *abs((w_trans/(2*math.pi))/(w_trans/(2*math.pi) - Delta_r)), Delta_r)
    
    Delta_blue = solve(abs(Delta_b) - math.pi*gamma*A_tot/(2*abs(A_R)*2*math.pi*err)*4/eta\
          *abs((w_trans/(2*math.pi))/(w_trans/(2*math.pi) - Delta_b)), Delta_b)
    
    #Compute laser wavelength
    #[blue-detuned wavelength, red-detuned wavelength]
    las_wav = [3e8/(w_trans/(2*math.pi) - abs(max(Delta_red))),3e8/(w_trans/(2*math.pi) + abs(max(Delta_blue)))]
    
    return las_wav

def FullNumModel(Delta, freqs, eta, gamma, mat_eles, two_q=True):
    #Delta = detuning in THz
    #
    #freqs = P3/2 transitions frequencies in THz in the order S1/2, D3/2, D5/2
    #
    #eta = LD parameter for laser tuned to D5/2 <-> P3/2 transition frequency
    #
    #gamma = P3/2 decay rate in THz
    #
    #brs = List of fraction of Raman scattering to S1/2, D3/2, and D5/2 in simple
    #      model.
    #
    #mat_eles = List of outputs of Raman_Sum_2Processes for decays to S1/2, D3/2, and
    #           D5/2, in that order.
    
    import math
    
    ws12 = freqs[0];
    wd32 = freqs[1];
    wd52 = freqs[2];
    
    mat_ele_s12 = mat_eles[0];
    mat_ele_d32 = mat_eles[1];
    mat_ele_d52 = mat_eles[2];
    
    err = math.pi/2*gamma*((mat_ele_s12[0]*(1 + Delta/(ws12))**3\
                          +mat_ele_d32[0]*\
                          (1 + Delta/(wd32))**3+\
                          mat_ele_d52[0]*\
                          (1 + Delta/(wd52))**3)/Delta**2 +\
                          (mat_ele_s12[1]*\
                          (1 + Delta/(ws12))**3)\
                          /(-1*Delta*(ws12 + wd52 + Delta)) +\
                          (mat_ele_d32[1]*\
                          (1 + Delta/(wd32))**3)\
                          /(-1*Delta*(wd32 + wd52 + Delta)) +\
                          (mat_ele_d52[1]*\
                          (1 + Delta/(wd52))**3)\
                          /(-1*Delta*(wd52 + wd52 + Delta)) +\
                          (mat_ele_s12[2]*\
                          (1 + Delta/(ws12))**3)\
                          /(ws12 + wd52 + Delta)**2 +\
                          (mat_ele_d32[2]*\
                          (1 + Delta/(wd32))**3)\
                          /(wd32 + wd52 + Delta)**2 +\
                          (mat_ele_d52[2]*\
                          (1 + Delta/(wd52))**3)/(wd52 + wd52 + Delta)**2\
                          )*15*abs(Delta)/2

    if two_q:
        err *= 4/eta*wd52/abs(wd52 + Delta)
    
    
    return err

def FullNumPow_higher(Delta, w32, w0, gamma, alpha, tau, eta, K, Rabi, two_q=True):
    #Calculates power requirements in full model, now including Wes's RII process AND higher levels!!
    
    import math
    hbar = 1.055*1e-34;
    
    Pow = hbar*w32**3*w0**2/(3*(3e8)**2*alpha*gamma)*math.pi/(2*tau)/abs(Rabi)
    
    if two_q:
        Pow *= math.sqrt(K)/eta*w32/abs(w32 + Delta)
    
    return Pow

def FullNumPow(Delta, w32, w0, gamma, alpha, tau, eta, K, two_q=True):
    #Calculates power requirements in full model, now including Wes's RII process!
    
    import math
    hbar = 1.055*1e-34;
    
    Pow = hbar*w32**3*w0**2/(3*(3e8)**2*alpha*gamma)*math.pi/(2*tau)/abs(2/15*(1/(-1*Delta)+1/(Delta + 2*w32)))
    
    if two_q:
        Pow *= math.sqrt(K)/eta*w32/abs(w32 + Delta)
    
    return Pow

def FullNumPow_simp(Delta, w32, w0, gamma, alpha, tau, eta, K, two_q=True):
    
    import math
    hbar = 1.055*1e-34;
    
    Pow = hbar*w32**3*w0**2/(3*(3e8)**2*alpha*gamma)*math.pi/(2*tau)*15*abs(Delta)/2
    
    if two_q:
        Pow *= math.sqrt(K)/eta
    
    return Pow

def FullNumPow_g(Delta, wf, w32, w0, gamma, alpha, tau, eta, K, two_q=True):
    
    import math
    hbar = 1.055*1e-34;
    
    Pow = hbar*w32**3*w0**2/(3*(3e8)**2*alpha*gamma)*math.pi/(2*tau)/abs((wf/3)/(Delta*(Delta + wf)) + (wf/3)/((Delta + 2*w32)*(Delta + 2*w32 + wf)))
    
    if two_q:
        Pow *= math.sqrt(K)/eta*w32/abs(w32 + Delta)
    
    return Pow