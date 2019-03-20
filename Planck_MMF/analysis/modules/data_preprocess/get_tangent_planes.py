##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
from astropy.io import fits
from modules.settings import global_mmf_settings as gset

def return_data(filename):
	channels=fits.getdata(filename,ext=1)
	if gset.mmfset.use_psf_data:
		data=fits.getdata(filename,ext=5)
	else:
		data=fits.getdata(filename,ext=2)
	chidx=[np.where(channels==ch)[0][0] for ch in gset.mmfset.channels]
	return data[chidx] #,channels[chidx]

def return_ps_mask(filename):
	mask=fits.getdata(filename,ext=3)
	if gset.mmfset.use_psf_data:
		mask[:,:]=1.
	return mask

def return_ext_ps_mask(filename):
	mask=fits.getdata(filename,ext=4)
	return mask
