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

############################


data_file = np.loadtxt('NHp1.dat',skiprows=1)
x1 = data_file[:,0]

data_file2 = np.loadtxt('NHp2.dat',skiprows=1)
x2 = data_file2[:,0]
    
##### a4Sigma  ######

y5 = data_file2[:,2]
y5 = y5 - y5[-1]

def V5p(r):
    inter = interp1d(x2, y5, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<26.456164 else 0.0

def func(r,a1,a2,a3,a4,a5,a6):
	return (np.exp(-a1*r)*(1 + a2/r) + np.exp(-a3*r)*(1/r - a4*r))/(a6 + a5*np.exp(-r)*(1 + r + r**2/3))

popt, pcov = curve_fit(func, x2, y5)
a1, a2, a3, a4, a5, a6 = popt

def Vfit(r):
	return func(r,a1,a2,a3,a4,a5,a6) 

##### r->0 extrapolation #####

x2ext = x2[0:3]
y5ext = y5[0:3]


x2ext1 = x2[0:6]
y5ext1 = y5[0:6]

def fext(r,a1e,a2e):
	return a1e*np.exp(-a2e*r)/r

popte, pcove = curve_fit(fext, x2ext, y5ext)
a1e, a2e = popte

def Vext(r):
	return fext(r,a1e,a2e) 

#####

def fext1(r,a1e1,a2e1):
	return a1e1*np.exp(-a2e1*r)/r

popte1, pcove1 = curve_fit(fext1, x2ext1, y5ext1)
a1e1, a2e1 = popte1

def Vext1(r):
	return fext1(r,a1e1,a2e1) 
	
####### PEC TOTAL ##########


def Vtot(r):
    return Vfit(r) if r>0.5669178 else Vext(r)


def Vtot1(r):
    return Vfit(r) if r>0.5669178 else Vext1(r)
	

############################
# N-H
m=1/(1/(14.003074*1822.888) + 1/(1.007825*1822.888))


#### CLASSICAL Q(E) ##########
def chi(b, E, V):
    lim = optimize.root_scalar(lambda r,b,E: V(r) - E*(1 - b**2/r**2), bracket=[0.00001,30], method='brenth', args=(b,E,))
    integ, err = quad(lambda r, b, E: 1/(r**2*np.sqrt( 1-b**2/r**2 - V(r)/E if 1-b**2/r**2 - V(r)/E>=0 else 0.01  )), lim.root, np.inf, args=(b,E,), epsrel=0.01, limit=1000)
    return math.pi-2*b*integ


def Qcl(l, E, V): 
    integ, err = nquad(lambda b, E: (1-np.cos(chi(b,E,V))**l)*b, [[0, 30]], args=(E,),  opts={"epsrel":0.01, "limit":1000})
    return 2*math.pi*integ
    
    
######  QUANTUM Q(E)  #########

def delta(E,l,V):
    U = np.linspace(0.00001, 30, 100)
    def aux(r):
        return V(r) - E*(1 - (l+1/2)**2/(2*m*E*r**2))
    VV = np.vectorize(aux)
    c = VV(U)
    si = np.sign(c)
    tp=30
    for i in range(100-1):
        if si[i] + si[i+1] == 0: # oposite signs
            lim = optimize.root_scalar(aux, bracket=[U[i],U[i+1]], method='brenth')
            tp=lim.root

    if tp<30:
        integ1, err = quad(lambda r, E: np.sqrt(1- V(r)/E - (l+1/2)**2/(2*m*E*r**2) if 1- V(r)/E - (l+1/2)**2/(2*m*E*r**2)>=0 else 0.0 ), tp, 30, args=(E,), limit=1000)
    else:
        integ1=0.0
    if (l+1/2)/np.sqrt(2*m*E)<30:
        integ2, err = quad(lambda r, E: np.sqrt(1- (l+1/2)**2/(2*m*E*r**2)  ), (l+1/2)/np.sqrt(2*m*E), 30, args=(E,), limit=1000)
    else:
        integ2=0.0
    return (integ1-integ2)*np.sqrt(2*m*E)


def Qqm(E,V):
    s = sum( (2*l+1)*np.sin(delta(E,l,V))**2  for l in range(12000) )
    return 4*math.pi*s/(2*m*E)


def Qqm1(E,V):
    s = sum( (l+1)*np.sin(delta(E,l,V)-delta(E,l+1,V))**2  for l in range(12000) )
    return 4*math.pi*s/(2*m*E)


def Qqm2(E,V):
    s = sum( ((l+1)*(l+2)/(2*l+3))*np.sin(delta(E,l,V)-delta(E,l+2,V))**2  for l in range(12000) )
    return 4*math.pi*s/(2*m*E)


#######   POINTS TO FILE momentum transfer cross section plots #######

# classical

#energies = np.logspace(np.log10(0.000001),np.log10(37),200)
#for E in energies:
#    print("{:.5e}".format(E), "{:.4e}".format( Qcl(1,E,Vtot1)), "{:.4e}".format( Qcl(2,E,Vtot1)), "{:.4e}".format( Qcl(3,E,Vtot1)) )

# WKB

#energies = np.logspace(np.log10(0.000001),np.log10(400),200)
#for E in energies:
#    print("{:.5e}".format(E), "{:.4e}".format( Qqm(E,Vtot1)), "{:.4e}".format( Qqm1(E,Vtot1)), "{:.4e}".format( Qqm2(E,Vtot1)) )




