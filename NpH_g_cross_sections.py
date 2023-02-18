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
    
##### X2Pi points #####

y2 = data_file[:,3]
y2 = y2 - y2[-1]

def V2p(r):
    inter = interp1d(x1, y2, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0


##### a2Sigma- #######

y1 = data_file[:,6]
y1 = y1 - y1[-1]

def V1p(r):
    inter = interp1d(x1, y1, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0

##### 4Pi ######

y4 = data_file2[:,1]
y4 = y4 - y4[-1]

def V4p(r):
    inter = interp1d(x2, y4, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<26.456164 else 0.0
    
    
##### 4Sigma-  ######

y3 = data_file2[:,3]
y3 = y3 - y3[-1]

def V3p(r):
    inter = interp1d(x2, y3, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<26.456164 else 0.0



##### r->0 extrapolation #####

x1ext = x1[0:3]
y2ext = y2[0:3]


def fext2(r,a2,b2):
	return a2*np.exp(-b2*r)/r

popte2, pcove2 = curve_fit(fext2, x1ext, y2ext)
a2, b2 = popte2

def Vext2(r):
	return fext2(r,a2,b2) 

#####

x1ext = x1[0:3]
y1ext = y1[0:3]

def fext1(r,a1,b1):
	return a1*np.exp(-b1*r)/r

popte1, pcove1 = curve_fit(fext1, x1ext, y1ext)
a1, b1 = popte1

def Vext1(r):
	return fext1(r,a1,b1) 


#####

x2ext = x2[0:3]
y4ext = y4[0:3]

def fext4(r,a4,b4):
	return a4*np.exp(-b4*r)/r

popte4, pcove4 = curve_fit(fext4, x2ext, y4ext)
a4, b4 = popte4

def Vext4(r):
	return fext4(r,a4,b4) 


#####

x2ext = x2[0:3]
y3ext = y3[0:3]

def fext3(r,a3,b3):
	return a3*np.exp(-b3*r)/r

popte3, pcove3 = curve_fit(fext3, x2ext, y3ext)
a3, b3 = popte3

def Vext3(r):
	return fext3(r,a3,b3) 

	
####### PEC TOTAL ##########


def Vtot2(r):
    return V2p(r) if r>0.5669178 else Vext2(r)


def Vtot1(r):
    return V1p(r) if r>0.5669178 else Vext1(r)
    
    
def Vtot4(r):
    return V4p(r) if r>0.5669178 else Vext4(r)
    
    
def Vtot3(r):
    return V3p(r) if r>0.5669178 else Vext3(r)

	

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


#######   POINTS TO FILE momentum transfer cross section plots - classical #######

# classical

#energies = np.logspace(np.log10(0.000001),np.log10(37),200)
#for E in energies:
#    print("{:.5e}".format(E), "{:.4e}".format( (2*Qcl(1,E,Vtot1)+4*Qcl(1,E,Vtot2)+4*Qcl(1,E,Vtot3)+8*Qcl(1,E,Vtot4))/(2+4+4+8)   ), "{:.4e}".format( (2*Qcl(2,E,Vtot1)+4*Qcl(2,E,Vtot2)+4*Qcl(2,E,Vtot3)+8*Qcl(2,E,Vtot4))/(2+4+4+8) ), "{:.4e}".format( (2*Qcl(3,E,Vtot1)+4*Qcl(3,E,Vtot2)+4*Qcl(3,E,Vtot3)+8*Qcl(3,E,Vtot4))/(2+4+4+8)  ) )
#     print("{:.5e}".format(E), "{:.4e}".format( Qcl(1,E,Vtot2)), "{:.4e}".format( Qcl(1,E,Vtot4)) )


