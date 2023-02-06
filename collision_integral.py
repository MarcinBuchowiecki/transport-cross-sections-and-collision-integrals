# This file allows to calculate collision integral from transport cross section

import numpy as np
import scipy as sp
import math
from scipy.integrate import nquad
from scipy.interpolate import interp1d

############################

data_file = np.loadtxt('NHpQcl.txt')

x = data_file[:,0]

# set l = 1, 2 or 3 for Q^(1), Q^(2) or Q^(3) 
l = 1   
tcs = data_file[:,l]


########## calculation of collision integral ####################
k=3.166812*10**-6
def Q(E):
    inter = interp1d(x, tcs, kind=3, fill_value = 'extrapolate') 
    return inter(E)

def omega(T, l, s): 
    integ, err = nquad(lambda E: np.exp(-E/(k*T))*E**(s+1)*Q(E)/(k*T)**(s+2), [[0.0000, np.inf]], opts={"epsrel":0.001, "limit":1000})
    return (0.529177249)**2*1/(1*math.pi*math.factorial(s+1)*(1-(1+(-1)**l)/(2*(1+l))))*integ
####################################

# set temperature and s for appropriate reduced collision integral sigma^2*Omega^(l,s)(T)
T= 1000
s=1
print(omega(T,l,s))



