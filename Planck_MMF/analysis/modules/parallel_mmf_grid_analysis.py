import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from settings import mmf_settings as mmfset
from filters import modular_multi_matched_filter as mmf
from data_preprocess import preprocess_planck_data as ppd
from simulate import cluster_templates as cltemp
from masking import gen_masks as gm

mmfset.init()
mmf_cat=ppd.get_tangent_plane_fnames()
mmf_cat=ppd.eval_M500_T500_theta500(mmf_cat)
tmplt=cltemp.cluster_spectro_spatial_templates(T_step=5.)
tmplt.setup_templates()
op=mmf.multi_matched_filter(tmplt.sp_ft_bank,tmplt.sz_spec_bank,tmplt.chfiltr,tmplt.fn_yerr_norm)
idx_list=np.arange(np.size(mmf_cat["SNR"]))
emask=gm.return_edge_apodized_mask()

def run_mmf_on_grid_in_parallel(numprocs):
    pool=mp.Pool(processes=numprocs)
    snr=pool.map(run_mmf_on_grid,idx_list)
    pool.close()
    pool.join()

def run_mmf_on_grid(idx):
	filename=mmf_cat["FILENAME"][idx]
	mask=fits.getdata(filename,2)
	data=fits.getdata(filename)
	op.get_data_ft(data*mask*emask,smwin=5)
	filename=mmfset.paths["result_data"] + "grid_" + mmf_cat["NAME"][idx][5:] + ".fits"
	theta500,T500,ans=op.eval_mmf_theta500_T500(maskthr=3.,mask_fdata=True,write_data=True,filename=filename)
