##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
import multiprocessing as mp
from astropy.io import fits
from settings import global_mmf_settings as gset
from filters import modular_multi_matched_filter as mmf
from data_preprocess import preprocess_planck_data as ppd
from data_preprocess import get_tangent_planes as gtp
from simulate import cluster_templates as cltemp
from masking import gen_masks as gm

outpath="/Users/adityarotti/Documents/Work/Projects/Relativistic-SZ/MMF-filters-for-rSZ-cluster-detection/Planck_MMF/results/planck_pr1/"
#outpath="/nvme/arotti/mmf_dataout/planck_pr1/mmf_blind/"
gset.setup_mmf_config(dataset="planck_pr1",outpath=outpath,chmin=100.,xsize=10.,result_midfix="",do_band_pass=True)
tmplt=cltemp.cluster_spectro_spatial_templates(T_step=1.)
tmplt.setup_templates()
mmf_cat=ppd.get_tangent_plane_fnames()
mmf_cat=ppd.eval_M500_T500_theta500(mmf_cat)
idx_list=np.arange(np.size(mmf_cat["SNR"]))
#emask=gm.return_edge_apodized_mask(15.,20.)
emask[:,:]=1.
op=mmf.multi_matched_filter(tmplt.sp_ft_bank,tmplt.sz_spec_bank,tmplt.chfiltr,tmplt.fn_yerr_norm)

def run_mmf_in_parallel(numprocs):
    pool=mp.Pool(processes=numprocs)
    snr=pool.map(run_mmf,idx_list)
    pool.close()
    pool.join()

def run_mmf(idx):
	filename=mmf_cat["FILENAME"][idx]
	mask=gtp.return_ps_mask(filename)
	data=gtp.return_data(filename)
	op.get_data_ft(data*mask*emask,smwin=5)
	
	# Optimize theta500 for T500=0.
	filename=gset.mmfset.paths["result_data"] + "otheta500_" + mmf_cat["NAME"][idx][5:] + ".fits"
	err,snr_max0,yc,otheta500,ans0=op.return_optimal_theta500(Tc=0.,write_data=True,filename=filename)

	# Optimize T500 fot optimal theta500
	filename=gset.mmfset.paths["result_data"] + "oT500_" + mmf_cat["NAME"][idx][5:] + ".fits"
	err,snr_maxT,yc,oT500,ansT=op.return_optimal_T500(thetac=round(otheta500,0),write_data=True,filename=filename)

	# Evaluate filter at optimized theta500 & T500
	filename=gset.mmfset.paths["result_data"] + "otheta500_mmf3T500_" + mmf_cat["NAME"][idx][5:] + ".fits"
	fdata,err,snrT,yc=op.return_snr(thetac=round(otheta500,0),Tc=round(mmf_cat["T500"][idx],0),write_data=True,filename=filename)

	print idx,mmf_cat["SNR"][idx],snr_max0,snrT
	return snr_max0
