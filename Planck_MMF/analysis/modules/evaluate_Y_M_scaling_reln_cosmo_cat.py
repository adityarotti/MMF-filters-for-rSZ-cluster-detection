import os
import numpy as np
from astropy.io import fits
from cosmology import cosmo_fn
from masking import gen_masks as gm
import multiprocessing as mp
from simulate import cluster_templates as cltemp
from flat_sky_codes import flat_sky_analysis as fsa
from data_preprocess import get_tangent_planes as gtp
from flat_sky_codes import tangent_plane_analysis as tpa
from filters import modular_multi_matched_filter as mmf
from modules.settings import global_mmf_settings as gset
from data_preprocess import preprocess_planck_data_cosmo_cat as ppd
from modules.simulate.spatial_template import sz_pressure_profile as szp

class Y_M_scaling(object):
	def __init__(self):
		self.xsz_cat=ppd.get_tangent_plane_fnames()
		#ppd.extract_tangent_planes()
		self.conv_Y5R500_SPHR500=szp.convert_Ycyl_xR500_Ysph_xR500()
		self.tmplt=cltemp.cluster_spectro_spatial_templates(T_min=0.,T_max=40.,T_step=0.1,theta500_min=2.,theta500_max=55.,theta_step=1.)
		self.tmplt.setup_templates()
		self.cmask=gm.return_center_mask()
		self.emask=gm.return_edge_apodized_mask(15.,20.)
		self.szspecT0=self.return_sz_spec(Tc=0.)
		self.idx_list=np.arange(np.size(self.xsz_cat["z"]))
	
	def eval_Y500_xray_prior(self,idx):
		filename=self.xsz_cat["FILENAME"][idx]
		theta500=self.xsz_cat["theta500"][idx]
		T500=self.xsz_cat["TX"][idx]
		glon=self.xsz_cat["GLON"][idx]
		glat=self.xsz_cat["GLAT"][idx]
		redshift=self.xsz_cat["z"][idx]
		projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
		ix,iy=projop.ang2ij(glon,glat)

		data=gtp.return_data(filename)
		ps_mask=gtp.return_ps_mask(filename)
		op=mmf.multi_matched_filter(self.tmplt.sp_ft_bank,self.tmplt.sz_spec_bank,self.tmplt.chfiltr,self.tmplt.fn_yerr_norm)
		op.get_data_ft(data*ps_mask*self.emask,smwin=5)

		template=self.tmplt.gen_template(thetac=theta500)
		template_ft=fsa.map2alm(np.fft.fftshift(template),gset.mmfset.reso)

		fdata,err=op.evaluate_mmf(template_ft,self.szspecT0)
		yc=max((fdata*self.cmask).ravel())
		cluster=cltemp.sc.gen_field_cluster_template(ix,iy,theta500,npix=gset.mmfset.npix,pixel_size=gset.mmfset.reso,y0=yc,cutoff=5.)
		Y500_T0=np.sum(cluster)*(gset.mmfset.reso**2.)*self.conv_Y5R500_SPHR500*((cosmo_fn.dA(redshift)*(np.pi/180./60.))**2.)
		Y500_err_T0=err*Y500_T0/yc

		szspecTc=self.return_sz_spec(Tc=T500)
		fdata,err=op.evaluate_mmf(template_ft,szspecTc)
		yc=max((fdata*self.cmask).ravel())
		cluster=cltemp.sc.gen_field_cluster_template(ix,iy,theta500,npix=gset.mmfset.npix,pixel_size=gset.mmfset.reso,y0=yc,cutoff=5.)
		Y500_TT=np.sum(cluster)*(gset.mmfset.reso**2.)*self.conv_Y5R500_SPHR500*((cosmo_fn.dA(redshift)*(np.pi/180./60.))**2.)
		Y500_err_TT=err*Y500_TT/yc
		
		return idx,theta500,T500,Y500_T0,Y500_err_T0,Y500_TT,Y500_err_TT

	def eval_Y500_blind(self,idx):
		filename=self.xsz_cat["FILENAME"][idx]
		T500=self.xsz_cat["TX"][idx]
		glon=self.xsz_cat["GLON"][idx]
		glat=self.xsz_cat["GLAT"][idx]
		redshift=self.xsz_cat["z"][idx]
		projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
		ix,iy=projop.ang2ij(glon,glat)

		data=gtp.return_data(filename)
		ps_mask=gtp.return_ps_mask(filename)
		op=mmf.multi_matched_filter(self.tmplt.sp_ft_bank,self.tmplt.sz_spec_bank,self.tmplt.chfiltr,self.tmplt.fn_yerr_norm)
		op.get_data_ft(data*ps_mask*self.emask,smwin=5)

		err,snr_max0,yc,otheta500T0,ans0=op.return_optimal_theta500(Tc=0.,mask_fdata=False)
		cluster=cltemp.sc.gen_field_cluster_template(ix,iy,otheta500T0,npix=gset.mmfset.npix,pixel_size=gset.mmfset.reso,y0=yc,cutoff=5.)
		Y500_T0=np.sum(cluster)*(gset.mmfset.reso**2.)*self.conv_Y5R500_SPHR500*((cosmo_fn.dA(redshift)*(np.pi/180./60.))**2.)
		Y500_err_T0=err*Y500_T0/yc

		# The code below optimzes theta500 for the xray temperature.
		#idxT500=np.where(abs(T500-self.tmplt.T500)==min(abs(T500-self.tmplt.T500)))[0][0]
		#err,snr_max0,yc,otheta500TT,ans0=self.op.return_optimal_theta500(Tc=self.tmplt.T500[idxT500],mask_fdata=False)
		#cluster=cltemp.sc.gen_field_cluster_template(ix,iy,otheta500TT,npix=gset.mmfset.npix,pixel_size=gset.mmfset.reso,y0=yc,cutoff=5.)
		#Y500_TT=np.sum(cluster)*(gset.mmfset.reso**2.)*self.conv_Y5R500_SPHR500*((cosmo_fn.dA(redshift)*(np.pi/180./60.))**2.)
		#Y500_err_TT=err*Y500_TT/yc
		
		template=self.tmplt.gen_template(thetac=otheta500T0)
		template_ft=fsa.map2alm(np.fft.fftshift(template),gset.mmfset.reso)
		szspecTc=self.return_sz_spec(Tc=T500)
		fdata,err=op.evaluate_mmf(template_ft,szspecTc)
		yc=max((fdata*self.cmask).ravel())
		cluster=cltemp.sc.gen_field_cluster_template(ix,iy,otheta500T0,npix=gset.mmfset.npix,pixel_size=gset.mmfset.reso,y0=yc,cutoff=5.)
		Y500_TT=np.sum(cluster)*(gset.mmfset.reso**2.)*self.conv_Y5R500_SPHR500*((cosmo_fn.dA(redshift)*(np.pi/180./60.))**2.)
		Y500_err_TT=err*Y500_TT/yc

		return idx,otheta500T0,T500,Y500_T0,Y500_err_T0,Y500_TT,Y500_err_TT


	def return_sz_spec(self,Tc=0.):
		temp=self.tmplt.sz_op.fn_sz_2d_T(Tc,gset.mmfset.channels)[:,0]
		szspec={}
		for i,ch in enumerate(gset.mmfset.channels):
			szspec[ch]=temp[i]
		return szspec

	def return_log_err(self,gauss_mean,gauss_err,num_samples=20000,ignore_negatives=True):
		logerr=np.zeros_like(gauss_mean)
		for idx, mu in enumerate(gauss_mean):
			x=np.random.normal(mu,gauss_err[idx],num_samples)
			if ignore_negatives:
				while np.any(x<0):
					neg_idx=np.where(x<0)[0]
					x[neg_idx]=np.random.normal(mu,gauss_err[idx],np.size(neg_idx))
			logerr[idx]=np.std(log10(x))
		return logerr
