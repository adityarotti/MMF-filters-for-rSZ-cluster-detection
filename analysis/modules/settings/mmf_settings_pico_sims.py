import os
import healpy as h
import numpy as np

from flat_sky_codes import tangent_plane_analysis as tpa
from experiments.pico_sims import *

nside=4096		# Healpix resolution of the maps
pwc=True
xsize=10. 		# Degrees.
projection_operator=tpa.tangent_plane_setup(nside,xsize,0.,0.,rescale=1.)
reso=projection_operator.pixel_size # arcminutes
npix=projection_operator.npix
mask_planck_maps=True
mask_tangent_planes=True

paths["templates"]="../data/template_bank/" + str(int(xsize)) + "deg_patches/"
paths["tplanes"]="../data/tangent_planes/pico_sims/" + str(int(xsize)) + "deg_patches/"
paths["result_data"]="../results/pico_sims/" + str(int(xsize)) + "deg_patches/data/"
paths["result_figs"]="../results/pico_sims/" + str(int(xsize)) + "deg_patches/figs/"

def init():
	for key in paths.keys():
		ensure_dir(paths[key])
#	ensure_dir(paths["templates"])
#	ensure_dir(paths["tplanes"])
#	ensure_dir(paths["result_data"])
#	ensure_dir(paths["result_figs"])
#	ensure_dir(paths["result_figs_dump"])
#	ensure_dir(paths["result_img"])

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)



