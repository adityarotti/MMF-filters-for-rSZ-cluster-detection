import os
import healpy as h
import numpy as np

from flat_sky_codes import tangent_plane_analysis as tpa
from experiments.simulations import *


nside=2048		# Healpix resolution of the maps
pwc=True
xsize=10. 		# Degrees.
projection_operator=tpa.tangent_plane_setup(nside,xsize,0.,0.,rescale=1.)
reso=projection_operator.pixel_size # arcminutes
npix=projection_operator.npix
mask_planck_maps=True
mask_tangent_planes=True

channels=channels[3:]
paths["templates"]="../data/template_bank/" + str(int(xsize)) + "deg_patches/"
paths["tplanes"]="../data/tangent_planes/simulations/" + str(int(xsize)) + "deg_patches/"
paths["result_data"]="../results/simulations/" + str(int(xsize)) + "deg_patches/data/"
paths["result_figs"]="../results/simulations/" + str(int(xsize)) + "deg_patches/figs/"

def init():
	ensure_dir(paths["templates"])
	ensure_dir(paths["tplanes"])
	ensure_dir(paths["result_data"])
	ensure_dir(paths["result_figs"])

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)



