import sys

import numpy as np
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
