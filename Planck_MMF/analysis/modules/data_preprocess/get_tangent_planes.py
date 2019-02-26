import numpy as np
from astropy.io import fits
from settings import mmf_settings as mmfset

def return_data(filename):
	channels=fits.getdata(filename,ext=1)
	data=fits.getdata(filename,ext=2)
	chidx=[np.where(channels==ch)[0][0] for ch in mmfset.channels]
	return data[chidx] #,channels[chidx]

def return_ps_mask(filename):
	mask=fits.getdata(filename,ext=3)
	return mask

def return_ext_ps_mask(filename):
	mask=fits.getdata(filename,ext=4)
	return mask
