import numpy as np
import multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.io import fits
from settings import mmf_settings as mmfset
from filters import modular_multi_matched_filter as mmf
from data_preprocess import preprocess_planck_data as ppd
from simulate import cluster_templates as cltemp
from masking import gen_masks as gm

mmfset.init()
tmplt=cltemp.cluster_spectro_spatial_templates(T_step=1.)
tmplt.setup_templates()
mmf_cat=ppd.get_tangent_plane_fnames()
mmf_cat=ppd.eval_M500_T500_theta500(mmf_cat)
idx_list=np.arange(np.size(mmf_cat["SNR"]))
emask=gm.return_edge_apodized_mask()
op=mmf.multi_matched_filter(tmplt.sp_ft_bank,tmplt.sz_spec_bank,tmplt.chfiltr,tmplt.fn_yerr_norm)


def run_mmf_with_mmf3param_in_parallel(numprocs):
    pool=mp.Pool(processes=numprocs)
    snr=pool.map(run_mmf_with_mmf3param,idx_list)
    pool.close()
    pool.join()

def run_mmf_with_mmf3param(idx):
	filename=mmf_cat["FILENAME"][idx]
	mask=fits.getdata(filename,2)
	data=fits.getdata(filename)
	op.get_data_ft(data*mask*emask,smwin=5)
	
	theta500=mmf_cat["theta500"][idx]
	T500=mmf_cat["T500"][idx]
	norm=tmplt.fn_yerr_norm(theta500)
	
	fdata0,err0,snr0,yc0=op.return_snr(thetac=round(theta500,0),Tc=0)
	fdataT,errT,snrT,ycT=op.return_snr(thetac=round(theta500,0),Tc=round(T500,0))

	hdu0=fits.PrimaryHDU()

	hdu1=fits.ImageHDU()
	hdu1.header["EXTNAME"]="RESULT 0"
	hdu1.header["COMMENT"]="theta500, T500, y_c err, SNR, y_c, Conv. to Y err"
	hdu1.data=[theta500,0.,err0,snr0,yc0,norm]

	hdu2=fits.ImageHDU()
	hdu2.header["EXTNAME"]="IMG T0"
	hdu2.data=fdata0

	hdu3=fits.ImageHDU()
	hdu3.header["EXTNAME"]="RESULT T"
	hdu3.data=[theta500,T500,errT,snrT,ycT,norm]

	hdu4=fits.ImageHDU()
	hdu4.header["EXTNAME"]="IMG Tc"
	hdu4.data=fdataT

	hdu=fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4])
	filename=mmfset.paths["result_data"] + "mmf3par_" + mmf_cat["NAME"][idx][5:] + ".fits"
	hdu.writeto(filename,overwrite=True)
	
	print mmf_cat["SNR"][idx],snr0,snrT
	return snr0
