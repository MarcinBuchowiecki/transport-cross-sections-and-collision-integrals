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
    
##### 2Delta  ######

y1 = data_file[:,7]
y1 = y1 - y1[-1]

def V1p(r):
    inter = interp1d(x1, y1, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0


##### 2Pi ########################

y2 = data_file[:,4]
y2 = y2 - y2[-1]

def V2p(r):
    inter = interp1d(x1, y2, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0

##### 2Sigma- #####

y3 = data_file[:,8]
y3 = y3 - y3[-1]

def V3p(r):
    inter = interp1d(x1, y3, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0

####################################
####################################


#######  2Sigma+   ######################

y4 = data_file[:,1]
y4 = y4 - y4[-1]

def V4p(r):
    inter = interp1d(x1, y4, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0

#######  2Pi   ######################

y5 = data_file[:,5]
y5 = y5 - y5[-1]

def V5p(r):
    inter = interp1d(x1, y5, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0

#######  2Sigma+   ######################

y6 = data_file[:,2]
y6 = y6 - y6[-1]

def V6p(r):
    inter = interp1d(x1, y6, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0


#######  2Delta   ######################

y7 = data_file[:,9]
y7 = y7 - y7[-1]

def V7p(r):
    inter = interp1d(x1, y7, kind=3, fill_value = 'extrapolate') 
    return inter(r) if r<16.440616 else 0.0

#############################
##### r->0 extrapolation #####

x1ext = x1[0:3]
y1ext = y1[0:3]


def fext1(r,a1,b1):
	return a1*np.exp(-b1*r)/r

popte1, pcove1 = curve_fit(fext1, x1ext, y1ext)
a1, b1 = popte1

def Vext1(r):
	return fext1(r,a1,b1) 

######

y2ext = y2[0:3]


def fext2(r,a2,b2):
	return a2*np.exp(-b2*r)/r

popte2, pcove2 = curve_fit(fext2, x1ext, y2ext)
a2, b2 = popte2

def Vext2(r):
	return fext2(r,a2,b2) 
	
#####

y3ext = y3[0:3]


def fext3(r,a3,b3):
	return a3*np.exp(-b3*r)/r

popte3, pcove3 = curve_fit(fext3, x1ext, y3ext)
a3, b3 = popte3

def Vext3(r):
	return fext3(r,a3,b3) 
	
######

y4ext = y4[0:3]


def fext4(r,a4,b4):
	return a4*np.exp(-b4*r)/r

popte4, pcove4 = curve_fit(fext4, x1ext, y4ext)
a4, b4 = popte4

def Vext4(r):
	return fext4(r,a4,b4) 
	
#####

y5ext = y5[0:3]


def fext5(r,a5,b5):
	return a5*np.exp(-b5*r)/r

popte5, pcove5 = curve_fit(fext5, x1ext, y5ext)
a5, b5 = popte5

def Vext5(r):
	return fext5(r,a5,b5) 
	
#####

y6ext = y6[0:3]


def fext6(r,a6,b6):
	return a6*np.exp(-b6*r)/r

popte6, pcove6 = curve_fit(fext6, x1ext, y6ext)
a6, b6 = popte6

def Vext6(r):
	return fext6(r,a6,b6) 
	
#####

y7ext = y7[0:3]


def fext7(r,a7,b7):
	return a7*np.exp(-b7*r)/r

popte7, pcove7 = curve_fit(fext7, x1ext, y7ext)
a7, b7 = popte7

def Vext7(r):
	return fext7(r,a7,b7) 			
	
	
####### PEC TOTAL ##########


def V1tot(r):
    return V1p(r) if r>0.5669178 else Vext1(r)

def V2tot(r):
    return V2p(r) if r>0.5669178 else Vext2(r)

def V3tot(r):
    return V3p(r) if r>0.5669178 else Vext3(r)

def V4tot(r):
    return V4p(r) if r>0.5669178 else Vext4(r)

def V5tot(r):
    return V5p(r) if r>0.5669178 else Vext5(r)
	
def V6tot(r):
    return V6p(r) if r>0.5669178 else Vext6(r)

def V7tot(r):
    return V7p(r) if r>0.5669178 else Vext7(r)
	

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

energies = np.logspace(np.log10(0.000001),np.log10(37),200)
## N(2D)-H+
for E in energies:
    print("{:.5e}".format(E), "{:.4e}".format( (4*Qcl(1,E,Vtot1)+4*Qcl(1,E,Vtot2)+2*Qcl(1,E,Vtot3))/(4+4+2)   ), "{:.4e}".format( (4*Qcl(2,E,Vtot1)+4*Qcl(2,E,Vtot2)+2*Qcl(2,E,Vtot3))/(4+4+2) ), "{:.4e}".format( (4*Qcl(3,E,Vtot1)+4*Qcl(3,E,Vtot2)+2*Qcl(3,E,Vtot3))/(4+4+2)  ) )
# N+(1D)-H(2S)
    print("{:.5e}".format(E), "{:.4e}".format( (2*Qcl(1,E,Vtot4)+4*Qcl(1,E,Vtot5)+2*Qcl(1,E,Vtot6)+4*Qcl(1,E,Vtot7))/(2+4+2+4) ), "{:.4e}".format( (2*Qcl(2,E,Vtot4)+4*Qcl(2,E,Vtot5)+2*Qcl(2,E,Vtot6)+4*Qcl(2,E,Vtot7))/(2+4+2+4) ),  "{:.4e}".format( (2*Qcl(3,E,Vtot4)+4*Qcl(3,E,Vtot5)+2*Qcl(3,E,Vtot6)+4*Qcl(3,E,Vtot7))/(2+4+2+4) )  )


















