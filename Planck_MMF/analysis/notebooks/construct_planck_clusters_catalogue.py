##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  17 September 2020		     				 		                             #
# Date modified: 17 September 2020																 #
##################################################################################################

import sys,os
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../modules/"))
import healpy as h
import multiprocessing as mp
from modules.settings import global_mmf_settings as gset
from data_preprocess import tiling_the_sphere as tts
from data_preprocess import tile_planck_data as tpd
from filters import modular_multi_matched_filter as mmf
from simulate import cluster_templates as cltemp
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import time
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

dataset="planck_pr3"
whichch="hfi"
mask_point_sources=False
numprocs=35

if whichch=="allch":
    chmin=30.
elif whichch=="hfi":
    chmin=100.

use_psf_data=True
ps_suffix="_inpainted_data"
if mask_point_sources:
    use_psf_data=False
    ps_suffix="_masked_data"


figstamp=dataset + "_" + whichch + ps_suffix
dir_suffix=whichch + ps_suffix

save_results=False
run_analysis=False

figstamp,dir_suffix

outpath="/results/" + dataset + "/planck_cat_" + dir_suffix + "/"
tempdatapath="/tempdata/" + dataset + "/planck_tiles/"
gset.setup_mmf_config(dataset=dataset,outpath=outpath,tempdatapath=tempdatapath,chmin=chmin,xsize=15.,do_band_pass=True,use_psf_data=use_psf_data)

figpath=gset.mmfset.paths["result_figs"]
datapath=gset.mmfset.paths["result_data"]
logging.basicConfig(filename=figpath + 'extract_planck_cluster_cat.log',level=logging.INFO)



tile_map,fsky_map,apo_mask=tts.return_sky_tile_map()
tiledef=tpd.get_tangent_plane_fnames(fsky_map=fsky_map,fsky_thr=0.3)
plt.ioff()
h.mollview(tile_map)
filename=figpath + "tile_the_sky.pdf"
plt.savefig(filename,bbox_inches="tight",dpi=200)
plt.clf()
plt.close()
logging.info("================>Completed Sky tiling definitions")

tpd.extract_data_tiles(tiledef)
logging.info("================>Completed extracting sky tiles, point source mask and galactic mask")

def gen_ps_inpainted_data(px):
    tpd.gen_ps_inpainted_data(px,tiledef)

def parallel_psfill(numprocs):
    pool=mp.Pool(processes=numprocs)
    pool.map(gen_ps_inpainted_data,tiledef.keys())
    pool.close()
    pool.join()

start=time.time()
parallel_psfill(4)
print time.time()-start
logging.info("================>Completed inpaiting the point sources")

tmplt=cltemp.cluster_spectro_spatial_templates(T_step=1.,theta500_min=1.,theta500_max=100.,theta_step=2.)
tmplt.setup_templates()
logging.info("================>Completed constructing the spatial and spectral templates")

from automated_detection import extract_tile_cluster_catalogue as etcc

def wrap_extract_tile_cluster_catalogue(px):
    etcc.extract_tile_cluster_catalogue(px,tiledef,tmplt)

def parallel_etcc(numprocs):
    pool=mp.Pool(processes=numprocs)
    pool.map(wrap_extract_tile_cluster_catalogue,tiledef.keys())
    pool.close()
    pool.join()

start=time.time()
parallel_etcc(5)
print time.time()-start
logging.info("================>Completed extracting the cluster catalogues for all sky tiles")


summary_cat=etcc.return_final_cluster_catalogue(tiledef,verbose=False)
logging.info("================>Completed constructing the final Planck catalogue")
