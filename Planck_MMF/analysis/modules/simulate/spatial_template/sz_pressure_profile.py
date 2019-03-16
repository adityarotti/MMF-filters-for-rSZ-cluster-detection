##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
import upp_param_arnaud2011 as upp

# The zero of the radial profile is set at frac*R500

def gnfw_3D_pressure_profile(z,rho,R500):
	'''This is the 3D pressure profile'''
	# The variables are set in cylindrical coordinates.
	r=np.sqrt(z**2. + rho**2.)
	den1=(upp.c500*r/R500)**upp.gamma
	den2=(1.+(upp.c500*r/R500)**upp.alpha)**((upp.beta-upp.gamma)/upp.alpha)
	p3d=upp.P0/(den1*den2)
	return p3d

def gnfw_2D_pressure_profile(rho,R500,limits=50.,frac=1e-6):
	'''
	The nomalization is such that the center of the cluster is set to unity.
	'''
	p2d0=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(frac*R500,R500))[0]
	if np.size(rho)>1:
		p2d=np.zeros(np.size(rho),float)
		for i,rhop in enumerate(rho):
			p2d[i]=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(rhop,R500))[0]/p2d0
	else:
		p2d=quad(gnfw_3D_pressure_profile,-limits*R500,limits*R500,args=(rho,R500))[0]/p2d0

	return p2d


def beta_3D_pressure_profile(z,rho,R500):
	'''This is infact the electron density profile. Assuming the cluster to be isothermal, the pressure follows the same profile'''
	beta=2./3. ; P0=upp.P0 # This is arbitrarily set to the GNFW normalization.
	r=np.sqrt(z**2. + rho**2.)
	p3d=P0*(1. + (r**2.)/(R500**2.))**(-3.*beta/2.)
	return p3d

def beta_2D_pressure_profile(rho,R500,limits=500.,frac=1e-6):
	p2d0=quad(beta_3D_pressure_profile,-limits*R500,limits*R500,args=(frac*R500,R500))[0]
	if np.size(rho)>1:
		p2d=np.zeros(np.size(rho),float)
		for i,rhop in enumerate(rho):
			p2d[i]=quad(beta_3D_pressure_profile,-limits*R500,limits*R500,args=(rhop,R500))[0]/p2d0
	else:
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

def convert_Ycyl_xR500_Ysph_xR500(R500=10.,xcyl=5.,xsph=1.,limits=50.,frac=1e-6):
	Ycyl_xR500,norm=return_Ycyl_xR500(R500=R500,xcyl=xcyl,limits=limits,frac=frac)
	Ysph_xcylR500=return_Ysph_xR500(R500=R500,xsph=xcyl)
	Ysph_xR500=return_Ysph_xR500(R500=R500,xsph=xsph)
	return Ysph_xR500/Ycyl_xR500

def return_Ycyl_xR500(R500,xcyl,limits=50.,frac=1e-6):
	fn=lambda rho,z: 2.*np.pi*gnfw_3D_pressure_profile(z,rho,R500)*rho
	Ycyl_xR500=2.*dblquad(fn,0,limits*R500,lambda rho: 0., lambda rho: xcyl*R500)[0]
	norm=2.*quad(gnfw_3D_pressure_profile,0.,limits*R500,args=(frac*R500,R500))[0]
	return Ycyl_xR500,norm

def return_Ysph_xR500(R500,xsph):
	fn=lambda x: 4.*np.pi*gnfw_3D_pressure_profile(x,0.,R500)*x*x
	Ysph_xR500=quad(fn,0,xsph*R500)[0]
	return Ysph_xR500
