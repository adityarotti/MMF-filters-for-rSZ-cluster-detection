##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################


import numpy as np
import sz_pressure_profile as szp
from scipy.interpolate import interp1d

def gen_cluster_template(npix,R500,pixel_size,y0=1.,cutoff=10.,dorandom=False,profile="GNFW"):
	'''
	npix: Assuming its a square patch, the number of pixels on either side.
	R500: Radius of cluster core.
	y0 : Compton y-parameter.
	cutoff : The template is cutoff at cutoff*thetac.
	dorandom : If true, then the cluster positions are set randomly.
	'''
	delta=np.int(cutoff*R500/pixel_size)
	if dorandom:
		xc=np.random.randint(0+delta,npix-1-delta)
		yc=np.random.randint(0+delta,npix-1-delta)
	else:
		xc=np.int(npix/2)
		yc=xc

	distance=np.zeros((npix,npix),float)
	y,x=np.indices((distance.shape))
	distance=np.sqrt((x-xc)**2. +(y-yc)**2.)*pixel_size
	mask=np.ones((npix,npix),float)
	mask[distance>=cutoff*R500]=0.

	rhop=np.linspace(0.001*R500,1.2*cutoff*R500,500.)
	yprofile=np.zeros(np.size(rhop),float)
	if profile=="GNFW":
		yprofile=szp.gnfw_2D_pressure_profile(rhop,R500)
	elif profile=="beta":
		yprofile=szp.analytical_beta_2D_profile_profile(rhop,R500)

	fn_yprofile=interp1d(rhop,yprofile,kind="cubic",bounds_error=False,fill_value=(yprofile[0],yprofile[-1]))

	template=(fn_yprofile(distance.ravel()).reshape(npix,npix))*mask*y0

	return template

def gen_field_cluster_template(xc,yc,R500,y0,npix,pixel_size,cutoff=10.,profile="GNFW"):
	'''
	npix: Assuming its a square patch, the number of pixels on either side.
	R500: Radius of cluster core.
	y0 : Compton y-parameter.
	cutoff : The template is cutoff at cutoff*thetac.
	dorandom : If true, then the cluster positions are set randomly.
	'''
	
	distance=np.zeros((npix,npix),float)
	x,y=np.indices((distance.shape))
	distance=np.sqrt((x-xc)**2. +(y-yc)**2.)*pixel_size
	mask=np.ones((npix,npix),float)
	mask[distance>=cutoff*R500]=0.

	rhop=np.linspace(0.001*R500,1.2*cutoff*R500,500.)
	yprofile=np.zeros(np.size(rhop),float)
	if profile=="GNFW":
		yprofile=szp.gnfw_2D_pressure_profile(rhop,R500)
	elif profile=="beta":
		yprofile=szp.analytical_beta_2D_profile_profile(rhop,R500)

	fn_yprofile=interp1d(rhop,yprofile,kind="cubic",bounds_error=False,fill_value=(yprofile[0],yprofile[-1]))

	template=(fn_yprofile(distance.ravel()).reshape(npix,npix))*mask*y0

	return template


def gen_cluster_template_elliptical(npix,R500,pixel_size,e,phi0,y0=1.,cutoff=10.,dorandom=False,profile="GNFW"):
    '''
    npix: Assuming its a square patch, the number of pixels on either side.
    R500: Radius of cluster core.
    y0 : Compton y-parameter.
    cutoff : The template is cutoff at cutoff*thetac.
    dorandom : If true, then the cluster positions are set randomly.
    '''
    delta=np.int(cutoff*R500/pixel_size)
    if dorandom:
        xc=np.random.randint(0+delta,npix-1-delta)
        yc=np.random.randint(0+delta,npix-1-delta)
    else:
        xc=np.int(npix/2)
        yc=xc

    distance=np.zeros((npix,npix),float)
    y,x=np.indices((distance.shape),float)
    d=np.sqrt((x-xc)**2. + (y-yc)**2.)*pixel_size
    phi=np.arctan2((x-xc),(y-yc))
    distance=np.sqrt((d*np.cos(phi+phi0)/1.)**2. + ((d*np.sin(phi+phi0))/(1.-e*e))**2.)
    mask=np.ones((npix,npix),float)
    mask[distance>=cutoff*R500]=0.

    rhop=np.linspace(0.001*R500,1.2*cutoff*R500,500.)
    yprofile=np.zeros(np.size(rhop),float)
    if profile=="GNFW":
        yprofile=szp.gnfw_2D_pressure_profile(rhop,R500)
    elif profile=="beta":
        yprofile=szp.analytical_beta_2D_profile_profile(rhop,R500)

    fn_yprofile=interp1d(rhop,yprofile,kind="cubic",bounds_error=False,fill_value=(yprofile[0],yprofile[-1]))

    template=(fn_yprofile(distance.ravel()).reshape(npix,npix))*mask*y0

    return template
