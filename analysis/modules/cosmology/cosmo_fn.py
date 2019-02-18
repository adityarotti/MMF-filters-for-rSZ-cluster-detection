import numpy as np
from scipy.integrate import romberg

const_Om=0.3 
const_Ol=0.7 
const_h=0.7 
const_c=299792458.0

def Ez(z,Om=const_Om,Ol=const_Ol):
    return np.sqrt(Om*(1.+z)**3 + Ol)

def chi_intg(z):
    return ((const_c/1000.)/(const_h*100.))/Ez(z)

# Proper distance
def d0(z):
    y=romberg(chi_intg,0.,z)
    return y
d0=np.vectorize(d0)

# Angular diameter distance
def dA(z):
    return d0(z)/(1.+z)

# Luminosity distance
def dL(z):
    return d0(z)*(1.+z)

# T500-M500-z relationship
def convert_M500_T500(M500,z):
    y=5.*((M500/(3./const_h))**(2./3.))*(Ez(z)**(2./3.))
    return y

def convert_T500_M500(T500,z):
    y=(3./const_h)*((T500/5.)**1.5)/Ez(z)
    return y
#	--------------------------------

# T500-theta500-z relationship
def convert_T500_theta500(T500,z):
	M500=convert_T500_M500(T500,z)
	theta500=convert_M500_theta500(M500,z,beta=0.66,massbias=0.2,h=const_h)
	return theta500

def convert_theta500_T500(theta500,z):
	M500=convert_theta500_M500(theta500,z)
	y=convert_M500_T500(M500,z)
	return y
#	--------------------------------

# M500-theta500-z relationship
def convert_theta500_M500(theta500,z,beta=0.66,massbias=0.2,h=const_h):
	c0=6.997*(((h/0.7)*Ez(z))**-beta)
	y=(((theta500/c0)*(dA(z)/500.))**3.)*3./(1.-massbias)
	return y

def convert_M500_theta500(M500,z,beta=0.66,massbias=0.2,h=const_h):
    rhs=6.997*(((h/0.7)*Ez(z))**-beta)*(((1.-massbias)*M500/3.)**(1./3.))
    y=rhs*(500./dA(z))
    return y
#	--------------------------------

# Y-M relationship
conv_y2y500=0.5567  # Planck 2015

# Planck 2015
def convert_y500_M500(y500,z,beta=0.66,alpha=1.79,massbias=0.2,h=const_h):
    lhs=(Ez(z)**(-beta))*((dA(z)**2.)*y500*(10**(-3.))*((np.pi/180./60.)**2.)/1e-4)*(10**0.19)*((h/0.7)**(2.-alpha))
    y=(6.*const_h/(1.-massbias))*(lhs**(1./alpha))
    return y

#def pdf_nz(z,ncnt,zl,zh):
#    y=np.zeros(np.size(z),float)
#    for i in range(np.size(z)):
#        res=ncnt[(z[i]>=zl[:])*(z[i]<zh[:])]
#        if len(res)>0:
#            y[i]=res[0]
#    return y

# # Planck 2013
# conv_y2y500=1./1.79 # Planck 2013
# def return_m500(z,y500,beta=0.3,alpha=1.79,massbias=0.2,h=const_h):
#     lhs=(Ez(z)**(-beta))*((dA(z)**2.)*y500*(10**(-3.))*((pi/180./60.)**2.)/1e-4)*(10**0.19)
#     y=6.*(lhs**(1./alpha))
#     return y
