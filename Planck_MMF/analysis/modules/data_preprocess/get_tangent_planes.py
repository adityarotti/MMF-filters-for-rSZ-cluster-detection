##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
from astropy.io import fits
from modules.settings import global_mmf_settings as gset

def return_data(filename):
	channels=fits.getdata(filename,extname="Channels")
	if gset.mmfset.use_psf_data:
		data=fits.getdata(filename,extname="PS inpainted data tangent plane")
	else:
		data=fits.getdata(filename,extname="Data tangent plane")
	chidx=[np.where(channels==ch)[0][0] for ch in gset.mmfset.channels]
	return data[chidx] #,channels[chidx]

def return_ps_mask(filename):
	mask=fits.getdata(filename,extname="Point Source Mask")
	if gset.mmfset.use_psf_data:
#		print "You want to use PSF data, so no masking"
		mask[:,:]=1.
	return mask

def return_ext_ps_mask(filename):
	mask=fits.getdata(filename,extname="Extended point Source Mask")
	return mask

def return_galactic_mask(filename):
	mask=fits.getdata(filename,extname="Galactic mask")
	return mask
