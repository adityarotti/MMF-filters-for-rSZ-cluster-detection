import os
import healpy as h
import numpy as np

from flat_sky_codes import tangent_plane_analysis as tpa
from experiments.planck import *

nside=2048		# Healpix resolution of the maps
pwc=True
xsize=10. 		# Degrees.
projection_operator=tpa.tangent_plane_setup(nside,xsize,0.,0.,rescale=1.)
reso=projection_operator.pixel_size # arcminutes
npix=projection_operator.npix
mask_planck_maps=True
mask_tangent_planes=True

channels=np.copy(planck_channels)
channels=channels[3:]

ps_mask_weights={}
ps_mask_weights[30]=0.
ps_mask_weights[44]=0.
ps_mask_weights[70]=0.
ps_mask_weights[100]=1.
ps_mask_weights[143]=1.
ps_mask_weights[217]=1.
ps_mask_weights[353]=1.
ps_mask_weights[545]=1.
ps_mask_weights[857]=1.

projdir="/Users/adityarotti/Documents/Work/Projects/Relativistic-SZ/MMF-filters-for-rSZ-cluster-detection/Planck_MMF/"
paths["templates"]=projdir + "/data/template_bank/" + str(int(xsize)) + "deg_patches/"
paths["tplanes"]= projdir + "/data/tangent_planes/planck/" + str(int(xsize)) + "deg_patches/"
paths["result_data"]=projdir + "/results/planck/" + str(int(xsize)) + "deg_patches/data/"
paths["result_figs"]=projdir + "/results/planck/" + str(int(xsize)) + "deg_patches/figs/"
paths["result_figs_dump"]=projdir + "/results/planck/" + str(int(xsize)) + "deg_patches/figs_dump/"
paths["result_img"]=projdir + "/results/planck/" + str(int(xsize)) + "deg_patches/img/"

def init():
	ensure_dir(paths["templates"])
	ensure_dir(paths["tplanes"])
	ensure_dir(paths["result_data"])
	ensure_dir(paths["result_figs"])
	ensure_dir(paths["result_figs_dump"])
	ensure_dir(paths["result_img"])

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)



