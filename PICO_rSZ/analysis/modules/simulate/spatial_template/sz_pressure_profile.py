import numpy as np
from scipy.integrate import quad

def gnfw_3D_pressure_profile(z,rho,R500):
	'''This is the 3D pressure profile'''
	# The variables are set in cylindrical coordinates.
	P0=8.403 ; c500=1.177
	gamma=0.3081 ; alpha=1.0510 ; beta=5.4905
	r=np.sqrt(z**2. + rho**2.)
	den1=(c500*r/R500)**gamma
	den2=(1.+(c500*r/R500)**alpha)**((beta-gamma)/alpha)
	p3d=P0/(den1*den2)
	return p3d

def gnfw_2D_pressure_profile(rho,R500,limits=5.):
	'''
	The nomalization is such that the center of the cluster is set to unity.
	'''
	if np.size(rho)>1:
		p2d=np.zeros(np.size(rho),float)
		p2d0=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(rho[0],R500))[0]
		for i,rhop in enumerate(rho):
			p2d[i]=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(rhop,R500))[0]/p2d0
	else:
		p2d0=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(0.01*R500,R500))[0]
		p2d=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(rho,R500))[0]/p2d0

	return p2d


def beta_3D_pressure_profile(z,rho,R500):
	'''This is infact the electron density profile. Assuming the cluster to be isothermal, the pressure follows the same profile'''
	beta=2./3. ; P0=8.403 # This is arbitrarily set to the GNFW normalization.
	r=np.sqrt(z**2. + rho**2.)
	p3d=P0*(1. + (r**2.)/(R500**2.))**(-3.*beta/2.)
	return p3d

def beta_2D_pressure_profile(rho,R500,limits=500.):
	if np.size(rho)>1:
		p2d=np.zeros(np.size(rho),float)
		p2d0=quad(beta_3D_pressure_profile,-limits*R500,limits*R500,args=(rho[0],R500))[0]
		for i,rhop in enumerate(rho):
			p2d[i]=quad(beta_3D_pressure_profile,-limits*R500,limits*R500,args=(rhop,R500))[0]/p2d0
	else:
		p2d0=quad(beta_3D_pressure_profile,-limits*R500,limits*R500,args=(0.01*R500,R500))[0]
		p2d=quad(beta_3D_pressure_profile,-limits*R500,limits*R500,args=(rho,R500))[0]/p2d0
	return p2d

def analytical_beta_2D_profile_profile(rho,R500):
	'''
	distance: Distance from the center of the cluster.
	thetac: Size of the cluster.
	y0 : Compton y-parameter.
	'''
	beta=2./3.
	projy=(1.+ (rho/R500)**2.)**(-(3.*beta-1.)/2.)
	return projy
