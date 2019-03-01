import sys
import numpy as np
import healpy as h
from settings import mmf_settings as mmfset
from flat_sky_codes import flat_sky_analysis as fsa

def return_center_mask(radius=50.):
	mask=np.zeros((mmfset.npix,mmfset.npix),float)
	distance=np.zeros((mmfset.npix,mmfset.npix),float)
	y,x=np.indices((distance.shape))
	xc=np.int(mmfset.npix/2) ; yc=xc
	distance=np.sqrt((x-xc)**2. +(y-yc)**2.)*mmfset.reso
	maxnpix=2*int(radius/mmfset.reso)
	cpix=int(mmfset.npix/2.)
	mask[distance<=radius]=1.
	return mask

def return_edge_apodized_mask(edge_width=17.,fwhm=20.):
	mask=np.ones((mmfset.npix,mmfset.npix),float)
	epix=np.int(np.ceil(edge_width/mmfset.reso))
	mask[:epix,:]=0 ; mask[mmfset.npix-epix:,:]=0
	mask[:,:epix]=0 ; mask[:,mmfset.npix-epix:]=0
	ell,bl=fsa.get_gauss_beam(fwhm,20000)
	mask=fsa.filter_map(mask,mmfset.reso,bl,ell)
	return mask

def return_gal_mask():
	mask=h.read_map(mmfset.gal_mask_name,verbose=False)
	mask=(2.-mask)/2.
	nside=h.get_nside(mask)
	
	lmc_glon=280.4652
	lmc_glat=-32.8884
	lmc_pix=h.ang2pix(nside,lmc_glon,lmc_glat,lonlat=True)
	lmc_vec=h.pix2vec(nside,lmc_pix)
	lmc_pix=h.query_disc(nside,lmc_vec,radius=(3.*60./60.)*np.pi/180.)
	mask[lmc_pix]=0.

	smc_glon=302.8084
	smc_glat=-44.3277
	smc_pix=h.ang2pix(nside,smc_glon,smc_glat,lonlat=True)
	smc_vec=h.pix2vec(nside,smc_pix)
	smc_pix=h.query_disc(nside,smc_vec,radius=(1.5*60./60.)*np.pi/180.)
	mask[smc_pix]=0.
	
	return mask
