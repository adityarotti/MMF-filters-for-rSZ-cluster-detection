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
tmplt=cltemp.cluster_spectro_spatial_templates(T_step=1.,T_max=50.)
tmplt.setup_templates()
op=mmf.multi_matched_filter(tmplt.sp_ft_bank,tmplt.sz_spec_bank,tmplt.chfiltr,tmplt.fn_yerr_norm)
idx_list=np.arange(np.size(mmf_cat["SNR"]))
emask=gm.return_edge_apodized_mask()

def run_mmf_on_cgrid_in_parallel(numprocs):
    p=mp.Pool(processes=numprocs)
    snr=p.map(run_mmf_on_cgrid,idx_list)
    return snr

def run_mmf_on_cgrid(idx):
    filename=mmf_cat["FILENAME"][idx]
    mask=fits.getdata(filename,2)
    data=fits.getdata(filename)
    op.get_data_ft(data*mask*emask,smwin=5)
    filename=mmfset.paths["result_data"] + "cgrid_" + mmf_cat["NAME"][idx][5:] + ".fits"
	
    redshift=mmf_cat["REDSHIFT"][idx]
    err,snr_max,yc,oT500,ans=op.eval_mmf_theta500_T500_constrained(redshift,maskthr=3.,mask_fdata=True,write_data=True,filename=filename)

    print idx,mmf_cat["SNR"][idx],snr_max
    return snr_max
