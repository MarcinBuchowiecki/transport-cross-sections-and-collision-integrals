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
import sympy as sp
from scipy.optimize import curve_fit


####################

k=3.166812*10**-6

###############
de = 0.1 
s = 1.5

def Ve(r):
    return 4*de*((s/r)**12 - (s/r)**6)
##############



#### CLASSICAL Q(E) ##########
def chi(b, E, V):

    U = np.linspace(0.000001, 30, 100)
    def aux(r):
        return V(r) - E*(1 - b**2/r**2)
    VV = np.vectorize(aux)
    c = VV(U)
    si = np.sign(c)
    tp=30
    for i in range(100-1):
        if si[i] + si[i+1] == 0: # oposite signs
            lim = optimize.root_scalar(aux, bracket=[U[i],U[i+1]], method='brenth')
            tp=lim.root

    integ, err = quad(lambda r, b, E: 1/(r**2*np.sqrt( 1-b**2/r**2 - V(r)/E if 1-b**2/r**2 - V(r)/E>=0 else 0.01  )), tp, np.inf, args=(b,E,), epsrel=0.01, limit=1000)
    return math.pi-2*b*integ


def Qcl(l, E, V): 
    integ, err = nquad(lambda b, E: (1-np.cos(chi(b,E,V))**l)*b, [[0, 30]], args=(E,),  opts={"epsrel":0.01, "limit":1000})
    return 2*math.pi*integ
    
  

#######   POINTS TO FILE momentum transfer cross section plots  #######

#energies = np.logspace(np.log10(0.000001),np.log10(1),200)
#for E in energies:
#    print("{:.5e}".format(E), "{:.4e}".format( Qcl(1,E,Ve))  )



####### ############

#data_file1 = np.loadtxt('Q1_LJ.txt')
#x1 = data_file1[:,0] 
#cl1 = data_file1[:,2]


##### Omega ######


#def Q(E):
#    inter = interp1d(x1, cl1, kind=3, fill_value = 'extrapolate') 
#    return inter(E)


#def omega(T, l, s): 
#    integ, err = nquad(lambda E: np.exp(-E/(k*T))*E**(s+1)*Q(E)/(k*T)**(s+2), [[0.00000, np.inf]], opts={"epsrel":0.001, "limit":1000})
#    return (0.529177249)**2*1/(1*math.pi*math.factorial(s+1)*(1-(1+(-1)**l)/(2*(1+l))))*integ


#for temp in np.array([0.01,0.02,0.03,0.05,0.06,0.1,0.3]):
#    t=temp*de/k
#    print(temp, format(t, '.6g'), format(omega(t,1,1,Ve)/s**2, '.6g') )


