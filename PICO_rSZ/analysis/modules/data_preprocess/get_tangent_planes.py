import numpy as np
from astropy.io import fits
from settings import mmf_settings as mmfset

def return_data(filename):
	channels=fits.getdata(filename,ext=1)
	data=fits.getdata(filename,ext=2)
	chidx=[np.where(channels==ch)[0][0] for ch in mmfset.channels]
	return data[chidx],channels[chidx]
