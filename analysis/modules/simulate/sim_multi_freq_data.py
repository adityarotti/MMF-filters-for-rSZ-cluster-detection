import sys

import numpy as np
from flat_sky_codes import flat_sky_analysis as fsa
from spectral_template import sz_spec as szsed

def sim_planck_data(clfname,npix,pixel_size,channels,clusters=[],frgs={},fwhm={},sigma={},T=0.,cmbnorm=1.):
	numch=np.size(channels)
	temp=np.loadtxt(clfname)
	ell=temp[:,0] ; lmax=max(ell) ; cl=temp[:,1]*2.*np.pi/(ell*(ell+1.))
	
	# Getting the Planck band passed SZ spectrum
	plbp_sz_spec=szsed.return_planck_bp_sz_spec(T)
	
	cmb=fsa.gensim(cl=cl,nxpix=npix,pixel_size=pixel_size,ell=ell)
	data=np.zeros((numch,npix,npix),float)

	for i, ch in enumerate(channels):
		sim=cmb*cmbnorm*1e-6
		
		if clusters!=[]:
			sim=sim+clusters*plbp_sz_spec[ch]
	
		if frgs!={}:
			sim=sim+frg[ch]

		if fwhm!={}:
			ellp,bl=fsa.get_gauss_beam(fwhm[ch],lmax)
			sim=fsa.filter_map(sim,pixel_size,bl,ell=ellp)

		if sigma!={}:
			stddev=(sigma[ch]/pixel_size)
			noise=np.random.normal(scale=stddev,size=(npix,npix))
			sim=sim+noise*1e-6

		data[i,:,:]=sim
	
	if numch==1:
		return data[0,:,:]
	else:
		return data
