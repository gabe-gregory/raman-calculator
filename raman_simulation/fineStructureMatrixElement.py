class persist:
    saved_elVec = [];
    saved_paramVec = [];
    pass

def fineStructureMatrixElement(Ll,Jl,ml, Lu,Ju,mu, S):
    # el = fineStructureMatrixElement(Ll,Jl,ml, Lu,Ju,mu, S)
    # Calculate the dipole matrix element between (lower) state Ll,Jl,ml to
    # (upper) state Lu,Ju,mu with spin S. Returns the matrix element in units
    # of <L'||d||L> (the Reduced Matrix Element). This function caches previous
    # results for speed
    import numpy as np
    import math
    
    # Quick check of (some) selection rules for speed
    if  abs(mu-ml)>1 or abs(Lu-Ll)>1 or abs(Ju-Jl)>1:
        el = 0;
        return el
    
    # Calculate the parameter vector for this call
    thisParam = [ Ll,Jl,ml, Lu,Ju,mu, S ];
    
    # Check if cache initialised, if not, initialise
    if np.shape(persist.saved_paramVec)==(0,):
        persist.saved_elVec = [];
        persist.saved_paramVec = [];
    
    for i in np.arange(0,np.size(persist.saved_paramVec)):
        if np.all(persist.saved_paramVec[i]==thisParam):
            el = persist.saved_elVec[i];
            return
        
    ## Calculate and save in cache
    from sympy.physics.wigner import wigner_3j
    from sympy.physics.wigner import wigner_6j
    
    el = (-1)**(2*Ju+Ll+S-ml)*math.sqrt(2*Jl+1)*math.sqrt(2*Ju+1)*math.sqrt(2*Ll+1) * float(wigner_3j(Ju,1,Jl, mu,ml-mu,-ml))*float(wigner_6j(Ll,Lu,1, Ju,Jl,S));
    
    np.append(persist.saved_elVec, el);
    np.append(persist.saved_paramVec, thisParam);
    
    return el