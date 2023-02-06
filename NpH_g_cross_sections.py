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


# 37 hartree-> Qel l=5000
# 400 hartree -> Qel l=12000
#print(Qqm(400,Vtot1))

#print(Qcl(1,11,Vfit),Qcl(1,11,Vtot),Qcl(1,11,Vtot1))
#print(Qcl(2,11,Vfit),Qcl(2,11,Vtot),Qcl(2,11,Vtot1))

#print(Qcl(1,37,Vfit),Qcl(1,37,Vtot),Qcl(1,37,Vtot1))
#print(Qcl(2,37,Vfit),Qcl(2,37,Vtot),Qcl(2,37,Vtot1))

#print(Qqm(11,Vfit),Qqm(11,Vtot),Qqm(11,Vtot1))
#print(Qqm(37,Vfit),Qqm(37,Vtot),Qqm(37,Vtot1))

#print(Qqm(400,Vfit),Qqm(400,Vtot),Qqm(400,Vtot1))

#print(Qcl(1,400,Vfit),Qcl(1,400,Vtot),Qcl(1,400,Vtot1))

#######   POINTS TO FILE momentum transfer cross section plots - classical #######

# classical

#energies = np.logspace(np.log10(0.000001),np.log10(37),200)
#for E in energies:
#    print("{:.5e}".format(E), "{:.4e}".format( (2*Qcl(1,E,Vtot1)+4*Qcl(1,E,Vtot2)+4*Qcl(1,E,Vtot3)+8*Qcl(1,E,Vtot4))/(2+4+4+8)   ), "{:.4e}".format( (2*Qcl(2,E,Vtot1)+4*Qcl(2,E,Vtot2)+4*Qcl(2,E,Vtot3)+8*Qcl(2,E,Vtot4))/(2+4+4+8) ), "{:.4e}".format( (2*Qcl(3,E,Vtot1)+4*Qcl(3,E,Vtot2)+4*Qcl(3,E,Vtot3)+8*Qcl(3,E,Vtot4))/(2+4+4+8)  ) )
#     print("{:.5e}".format(E), "{:.4e}".format( Qcl(1,E,Vtot2)), "{:.4e}".format( Qcl(1,E,Vtot4)) )

# WKB

#energies = np.logspace(np.log10(0.000001),np.log10(400),200)
#for E in energies:
#    print("{:.5e}".format(E), "{:.4e}".format( (2*Qqm(E,Vtot1)+4*Qqm(E,Vtot2)+4*Qqm(E,Vtot3)+8*Qqm(E,Vtot4))/(2+4+4+8)   ), "{:.4e}".format( (2*Qqm1(E,Vtot1)+4*Qqm1(E,Vtot2)+4*Qqm1(E,Vtot3)+8*Qqm1(E,Vtot4))/(2+4+4+8) ), "{:.4e}".format( (2*Qqm2(E,Vtot1)+4*Qqm2(E,Vtot2)+4*Qqm2(E,Vtot3)+8*Qqm2(E,Vtot4))/(2+4+4+8)  ) )
#    print("{:.5e}".format(E), "{:.4e}".format( Qqm1(E,Vtot2)), "{:.4e}".format( Qqm1(E,Vtot4)) )

####### PLOTS ############

#data_file1 = np.loadtxt('NpH_Qsc_X2Pi_4Pi.txt')
#data_file2 = np.loadtxt('NpH_Qcl_X2Pi_4Pi.txt')
#x1 = data_file1[:,0]
#x2 = data_file2[:,0]

#### X2Pi
#sc1 = data_file1[:,1]
#cl1 = data_file2[:,1]
#### 4Pi
#sc2 = data_file1[:,2]
#cl2 = data_file2[:,2]

#plt.xlim(0.000001,37)
#plt.ylim(0.1,10000)
#plt.xscale('log')
#plt.yscale('log')

#plt.plot(x1, sc2, '--', color = 'black', label = 'WKB')
#plt.plot(x2, cl2, color = 'black', label = 'CM')
#plt.xlabel('E(E$_{\mathrm{h}}$)')
#plt.ylabel('Q$^{(1)}$(a$_{0}^{2}$)')
#plt.legend(loc = 'upper right')
#plt.savefig("NpHQ14Pi.pdf")


########## 

#data_file1 = np.loadtxt('NpHQsc.txt')
#data_file2 = np.loadtxt('NpHQcl.txt')
#x1 = data_file1[:,0]
#x2 = data_file2[:,0]

#sc = data_file1[:,2]
#cl = data_file2[:,1]

#plt.xlim(0.000001,37)
#plt.ylim(0.1,10000)
#plt.xscale('log')
#plt.yscale('log')

#plt.plot(x1, sc, '--', color = 'black', label = 'WKB')
#plt.plot(x2, cl, color = 'black', label = 'CM')
#plt.xlabel('E(E$_{\mathrm{h}}$)')
#plt.ylabel('Q$^{(1)}$(a$_{0}^{2}$)')
#plt.legend(loc = 'upper right')
#plt.savefig("NpHQ1.pdf")



##### test Omega11 ######


#data_file2 = np.loadtxt('NpHQsc.txt')
#data_file2 = np.loadtxt('NpHQcl.txt')
data_file2 = np.loadtxt('NpH_Qcl_X2Pi_4Pi.txt')
#data_file2 = np.loadtxt('NpH_Qsc_X2Pi_4Pi.txt')

x = data_file2[:,0]    
cl1 = data_file2[:,1]

def Q(E):
    inter = interp1d(x, cl1, kind=3, fill_value = 'extrapolate') 
    return inter(E)

def omega(T, l, s): 
    integ, err = nquad(lambda E: np.exp(-E/(k*T))*E**(s+1)*Q(E)/(k*T)**(s+2), [[0.0000, np.inf]], opts={"epsrel":0.001, "limit":1000})
    return (0.529177249)**2*1/(1*math.pi*math.factorial(s+1)*(1-(1+(-1)**l)/(2*(1+l))))*integ

print(omega(300,1,1))



