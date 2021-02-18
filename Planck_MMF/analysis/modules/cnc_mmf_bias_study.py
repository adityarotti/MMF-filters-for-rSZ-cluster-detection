##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  11 February 2021		     				 		                             #
# Date modified: 11 February 2021								 								 #
##################################################################################################
# Here we are studying the biases in the Matched Filtering analysis.

import numpy as np
import collections

import sys,os
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../modules/"))
import healpy as h
import collections
from astropy.io import fits

from modules.masking import gen_masks as gm
from modules.settings import global_mmf_settings as gset
from filters import modular_multi_matched_filter_v1 as mmf
from simulate import simulate_multi_channel_cluster as smcc
from modules.flat_sky_codes import flat_sky_analysis as fsa
from modules.data_preprocess import get_tangent_planes as gtp
from modules.flat_sky_codes import tangent_plane_analysis as tpa
from modules.data_preprocess import preprocess_planck_psm_data_esz_cat as ppd
from simulate import cluster_templates as cltemp

class mmf_anasim():
	def __init__(self,theta500_min=2,theta500_max=20,theta500_step=2,nrlz=20):
		self.construct_template_bank(theta500_min,theta500_max,theta500_step)
		self.extract_simulated_maps(nrlz)
		self.emask=gm.return_edge_apodized_mask(edge_width=30,fwhm=30)
	
	def extract_tangent_planes(self):
		ppd.extract_tangent_planes(gen_mask=True,verbose=False,do_data=True,do_mask=True)
	
	def construct_template_bank(self,theta500_min=2,theta500_max=20,theta500_step=2):
		self.theta500_max=theta500_max
		self.theta500_min=theta500_min
		self.theta500_step=theta500_step
		self.tmplt=cltemp.cluster_spectro_spatial_templates(T_step=1.,theta500_min=theta500_min,theta500_max=theta500_max,theta_step=theta500_step)
		self.tmplt.setup_templates()
		self.op=mmf.multi_matched_filter(self.tmplt.sp_ft_bank,self.tmplt.sz_spec_bank,self.tmplt.chfiltr,self.tmplt.fn_yerr_norm)

	def extract_simulated_maps(self,nrlz=20):
		self.nrlz=nrlz
		Nch=len(gset.mmfset.channels)
		self.noise=np.zeros((self.nrlz,Nch,gset.mmfset.npix,gset.mmfset.npix),np.float)
		xsz_cat=ppd.get_tangent_plane_fnames()
		for i in range(20):
			self.noise[i,]=gtp.return_data(xsz_cat["FILENAME"][i])

	def simulate_mf_cluster(self,M500,z,ymin=-5,ymax=-4,sampling=15,cold=True):
		Nch=len(gset.mmfset.channels)
		self.yc_inp=np.logspace(ymin,ymax,sampling)
		self.ymap=np.zeros((sampling,Nch,gset.mmfset.npix,gset.mmfset.npix),np.float)
		self.yc_true=np.zeros(sampling,np.float)

		for iy, yc in enumerate(self.yc_inp):
			self.ymap[iy,],self.Tc,self.theta500,self.yc_true[iy]=smcc.return_mc_cluster(M500,z,self.tmplt,yc,cold)

	def analyse(self,data,thetac,mask_fdata=False):
		soln=collections.OrderedDict()
		self.op.get_data_ft(data*self.emask,emask=self.emask)
		# All known
		fdata,err,snr,yc=self.op.return_snr_lk(thetac,Tc=0.,maskthr=2.,mask_fdata=mask_fdata)
		soln["AK"]=[yc,snr,thetac,err]
		# Size known
		fdata,err,snr,yc=self.op.return_snr(thetac,Tc=0.,maskthr=2.,mask_fdata=mask_fdata)
		soln["SK"]=[yc,snr,thetac,err]
		# Location known
		err,snr,yc,otheta500,ans=self.op.return_optimal_theta500_lk(Tc=0.,maskthr=2.,mask_fdata=mask_fdata)
		soln["LK"]=[yc,snr,otheta500,err]
		# No known
		err,snr,yc,otheta500,ans=self.op.return_optimal_theta500(Tc=0.,maskthr=2.,mask_fdata=mask_fdata)
		soln["NK"]=[yc,snr,otheta500,err]
		return soln
