import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
plt.style.use(['science','notebook','grid'])
from scipy.integrate import quadrature
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.integrate import dblquad
from scipy.integrate import nquad
from scipy.optimize import fsolve
from scipy import optimize
import math
from scipy.interpolate import interp1d
from scipy.interpolate import interpolate

####################

k=3.166812*10**-6

###############
de = 0.1 
s = 1.5

def Ve(r):
    return 4*de*((s/r)**12 - (s/r)**6)
##############

################################

def chi(b, E, T, V):
    lim = optimize.root_scalar(lambda r,b,E: V(r) - E*(1 - b**2/r**2), bracket=[0.00001,30], method='brenth', args=(b,E,))
    integ, err = quad(lambda r, b, E: 1/(r**2*np.sqrt( 1-b**2/r**2 - V(r)/E if 1-b**2/r**2 - V(r)/E>0 else 0.01  )), lim.root, np.inf, args=(b,E,), epsrel=0.01, limit=1000)
    return math.pi-2*b*integ

def Q(l, E, T, V): 
    integ, err = nquad(lambda b, E: (1-np.cos(chi(b,E,T,V))**l)*b, [[0, 30]], args=(E,),  opts={"epsrel":0.01, "limit":1000})
    return 2*math.pi*integ


def omega(T, l, s, V): 
    integ, err = nquad(lambda E: np.exp(-E/(k*T))*E**(s+1)*Q(l,E,T,V), [[0, np.inf]],  opts={"epsrel":0.01, "limit":1000})
    return 1/((k*T)**(s+2)*math.pi*math.factorial(s+1)*(1-(1+(-1)**l)/(2*(1+l))))*integ


###### omega11 ####### 

for temp in np.array([0.01,0.1,0.3]):
    t=temp*de/k
    print(temp, format(t, '.6g'), format(omega(t,1,1,Ve)/s**2, '.6g') )

###### omega22 ########

#for temp in np.array([0.01,0.1,0.3]):
#    t=temp*de/k
#    print(temp, format(t, '.6g'), format(omega(t,2,2,Ve)/s**2, '.6g') )

######################
















