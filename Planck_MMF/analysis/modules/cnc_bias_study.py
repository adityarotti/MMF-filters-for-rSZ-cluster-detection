##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  11 February 2021		     				 		                             #
# Date modified: 11 February 2021								 								 #
##################################################################################################
# Here we are studying the biases in the Matched Filtering analysis.

import numpy as np
from scipy.interpolate import interp1d
import collections
import healpy as h
from modules.flat_sky_codes import tangent_plane_analysis as tpa
from modules.flat_sky_codes import flat_sky_analysis as fsa
from simulate.spatial_template import sim_cluster as sc

class matched_filter_anasim(object):
	def __init__(self,npix,nside,fwhm=0.,seed=0,smwin=15,theta500_min=2,theta500_max=40,theta500_steps=20,white=False,apowid=0.03):
		self.npix=npix
		self.nside=nside
		self.reso=h.nside2resol(self.nside,arcmin=True)
		self.fwhm=fwhm
		self.white=white
		self.smwin=smwin
		self.d2x=((self.reso/60.)*np.pi/180.)**2.
		self.d2k=1./(self.npix*self.npix*self.d2x)
		self.theta500_min=theta500_min
		self.theta500_max=theta500_max
		self.theta500_steps=theta500_steps
		self.theta500_fine=np.linspace(self.theta500_min,self.theta500_max,1000)
		self.apowid=apowid
		
		y1=np.loadtxt("/Users/adityarotti/Documents/Work/Data/Planck/maps/bolliet2018.txt")
		fn=interp1d(np.log10(y1[:,0]),np.log10(y1[:,2]),fill_value="extrapolate",kind="linear")
		def return_planck_noise(ell):
			temp=10.**fn(np.log10(ell))
			temp=temp/(ell*(ell+1)*1e12/(2.*np.pi))
			return temp
		
		self.ell=np.linspace(2,int(max(y1[:,0])),int(max(y1[:,0]))-1)
		self.planck_ymap_noise=return_planck_noise(self.ell)
		# You need factors of pixel area here???
		self.pix_var=np.sqrt(np.sum(((2.*self.ell+1)/(4.*np.pi))*self.planck_ymap_noise))
		self.emask=self.return_edge_apodized_mask(self.apowid*self.npix*self.reso,self.apowid*self.npix*self.reso)
		self.construct_template_bank(self.theta500_min,self.theta500_max,self.theta500_steps)
	
		# patch noise power spectrum
		for i in range(10):
#			noise=fsa.gensim(self.planck_ymap_noise,self.npix,self.reso,self.ell)
			noise=self.simulate_noise()
			kell,pk=fsa.map2cl(noise*self.emask,pixel_size=self.reso,smwin=self.smwin)
			if i==0:
				self.clnoise=np.copy(pk)
				self.kell=kell
			else:
				self.clnoise=self.clnoise + pk
				
		self.clnoise=self.clnoise/10.
		self.Pk_noise=fsa.get_fourier_filter(self.clnoise,self.npix,pixel_size=self.reso,ell=self.kell)
		
	def construct_template_bank(self,theta500_min=2,theta500_max=40,theta500_steps=20):
		self.theta500_min=theta500_min
		self.theta500_max=theta500_max
		self.theta500_steps=theta500_steps
		self.theta_array=np.linspace(self.theta500_min,self.theta500_max,self.theta500_steps)
		self.tmplt_bank=collections.OrderedDict()
		self.tmplt_bank_ft=collections.OrderedDict()
		for theta500 in self.theta_array:
			tmplt=sc.gen_cluster_template(self.npix,theta500,self.reso,y0=1.)
			self.tmplt_bank[theta500]=tmplt
			self.tmplt_bank_ft[theta500]=fsa.map2alm(np.fft.fftshift(tmplt),self.reso)
	

	def simulate_noise(self):
		if self.white:
			noise=np.random.normal(size=(self.npix,self.npix),loc=0.,scale=self.pix_var)
		else:
			noise=fsa.gensim(self.planck_ymap_noise,self.npix,self.reso,self.ell)
		# You could add a smoothing step here.
		# But then the templates for MF will also need to be smoothed
		return noise*self.emask


	def simulate_ymap(self,theta500,yc,xy=[]):
		if xy!=[]:
			ymap=sc.gen_field_cluster_template(xy[0],xy[1],theta500,yc,self.npix,self.reso,cutoff=10.,profile="GNFW")
			yc_true=ymap[xy[0],xy[1]]
		else:
			ymap=sc.gen_cluster_template(self.npix,theta500,self.reso,y0=yc)
			yc_true=ymap[self.npix/2,self.npix/2]
		
#		noise=fsa.gensim(self.planck_ymap_noise,self.npix,self.reso,self.ell)
		noise=self.simulate_noise()
		data=ymap + noise
		# You could add a smoothing step here.
		# But then the templates for MF will also need to be smoothed
		return data*self.emask,noise*self.emask,yc_true

	def return_edge_apodized_mask(self,edge_width=30.,fwhm=20.):
		mask=np.ones((self.npix,self.npix),np.float)
		epix=np.int(np.ceil(edge_width/self.reso))
		mask[:epix,:]=0 ; mask[self.npix-epix:,:]=0
		mask[:,:epix]=0 ; mask[:,self.npix-epix:]=0
		ell,bl=fsa.get_gauss_beam(fwhm,20000)
		mask=fsa.filter_map(mask,self.reso,bl,ell)
		return mask

	def mf_ideal_all_known(self,data,theta500,xy,noise=[],data_ft=[]):
		'''
		Assumes noise is known
		Assumes the size is known
		Assumes the location is known
		'''
		if data_ft==[]:
			data_ft=fsa.map2alm(data,self.reso)
		
		if noise==[]:
			Pk=self.Pk_noise
		else:
			noise_ft=fsa.map2alm(noise,self.reso)
			avg_ell,avg_Cl=fsa.alm2cl(noise_ft,pixel_size=self.reso,smwin=self.smwin)
			Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)
		
		tmplt=sc.gen_cluster_template(self.npix,theta500,self.reso,y0=1.)
		template_ft=fsa.map2alm(np.fft.fftshift(tmplt),self.reso)
		norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err=np.sqrt(1./abs(norm))
		mf=(np.conj(template_ft)/Pk)/norm
		mf_data=fsa.alm2map(mf*data_ft,self.reso)
		yc_est=mf_data[xy[0],xy[1]]
		snr=yc_est/err
		
		return snr,yc_est,err,theta500,xy

	def mf_ideal_size_known(self,data,theta500,noise=[],data_ft=[]):
		'''
		Assumes noise is known
		Assume the size is known
		DOES NOT assume the location is known
		'''
		if data_ft==[]:
			data_ft=fsa.map2alm(data,self.reso)
		
		if noise==[]:
			Pk=self.Pk_noise
		else:
			noise_ft=fsa.map2alm(noise,self.reso)
			avg_ell,avg_Cl=fsa.alm2cl(noise_ft,pixel_size=self.reso,smwin=self.smwin)
			Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)
		
		tmplt=sc.gen_cluster_template(self.npix,theta500,self.reso,y0=1.)
		template_ft=fsa.map2alm(np.fft.fftshift(tmplt),self.reso)
		norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err=np.sqrt(1./abs(norm))
		mf=(np.conj(template_ft)/Pk)/norm
		mf_data=fsa.alm2map(mf*data_ft,self.reso)
		
		snr=mf_data/err ; bf_snr=max(snr.ravel())
		loc=np.where(snr==bf_snr) ; bf_xy=[loc[0][0],loc[1][0]]
		bf_yc=mf_data[bf_xy[0],bf_xy[1]]
		bf_soln=[bf_snr,bf_yc,err,theta500,bf_xy]
		
#		# Corrected for degrees of freedom.
#		snr=mf_data/err ; bf_snr=np.sqrt(max(snr.ravel())**2. - 2)
#		loc=np.where(snr==bf_snr) ; bf_xy=[loc[0][0],loc[1][0]]
#		bf_yc=mf_data[bf_xy[0],bf_xy[1]]
#		bf_soln_cdof=[bf_snr,bf_yc,err,theta500,bf_xy]

		return bf_soln #,bf_soln_cdof

	def mf_ideal_loc_known(self,data,xy,noise=[],data_ft=[]):
		'''
		Assumes noise is known
		DOES NOT  assume the size is known
		Assumes the location is known
		'''
		if data_ft==[]:
			data_ft=fsa.map2alm(data,self.reso)
		
		if noise==[]:
			Pk=self.Pk_noise
		else:
			noise_ft=fsa.map2alm(noise,self.reso)
			avg_ell,avg_Cl=fsa.alm2cl(noise_ft,pixel_size=self.reso,smwin=self.smwin)
			Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)
		
		snr=collections.OrderedDict()
		err=collections.OrderedDict()
		yc_est=collections.OrderedDict()
		for theta500 in self.theta_array:
			template_ft=self.tmplt_bank_ft[theta500]
			norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err[theta500]=np.sqrt(1./abs(norm))
			mf=(np.conj(template_ft)/Pk)/norm
			mf_data=fsa.alm2map(mf*data_ft,self.reso)
			yc_est[theta500]=mf_data[xy[0],xy[1]]
			snr[theta500]=(mf_data/err[theta500])[xy[0],xy[1]]
		
		bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln(snr,yc_est,err,xy)
		bf_soln=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]
#		bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln_cdof(snr,yc_est,err,xy,dof=1)
#		bf_soln_cdof=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]
		return bf_soln,[snr,yc_est,err,self.theta_array,xy]

	def mf_ideal(self,data,noise=[],data_ft=[]):
		'''
		Assumes noise is known
		DOES NOT assume the size is known
		DOES NOT assume the location is known
		'''
		if data_ft==[]:
			data_ft=fsa.map2alm(data,self.reso)
		
		if noise==[]:
			Pk=self.Pk_noise
		else:
			noise_ft=fsa.map2alm(noise,self.reso)
			avg_ell,avg_Cl=fsa.alm2cl(noise_ft,pixel_size=self.reso,smwin=self.smwin)
			Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)
		
		snr=collections.OrderedDict()
		err=collections.OrderedDict()
		yc_est=collections.OrderedDict()
		xy=collections.OrderedDict()
		for theta500 in self.theta_array:
			template_ft=self.tmplt_bank_ft[theta500]
			norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err[theta500]=np.sqrt(1./abs(norm))
			mf=(np.conj(template_ft)/Pk)/norm
			mf_data=fsa.alm2map(mf*data_ft,self.reso)
			snr_map=mf_data/err[theta500]
			loc=np.where(snr_map==max(snr_map.ravel())) ; xy[theta500]=[loc[0][0],loc[1][0]]
			yc_est[theta500]=mf_data[xy[theta500][0],xy[theta500][1]]
			snr[theta500]=(mf_data/err[theta500])[xy[theta500][0],xy[theta500][1]]
	
		bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln(snr,yc_est,err,xy)
		bf_soln=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]
		
		return bf_soln,[snr,yc_est,err,self.theta_array,xy]

	def mf_real(self,data,data_ft=[]):
		'''
		Fully agnostic to cluster characteristics and also to noise properties
		'''
		if data_ft==[]:
			data_ft=fsa.map2alm(data,self.reso)
		
		avg_ell,avg_Cl=fsa.alm2cl(data_ft,pixel_size=self.reso,smwin=self.smwin)
		Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)
		snr=collections.OrderedDict()
		err=collections.OrderedDict()
		yc_est=collections.OrderedDict()
		xy=collections.OrderedDict()
		for theta500 in self.theta_array:
			template_ft=self.tmplt_bank_ft[theta500]
			norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err[theta500]=np.sqrt(1./abs(norm))
			mf=(np.conj(template_ft)/Pk)/norm
			mf_data=fsa.alm2map(mf*data_ft,self.reso)
			snr_map=mf_data/err[theta500]
			loc=np.where(snr_map==max(snr_map.ravel())) ; xy[theta500]=[loc[0][0],loc[1][0]]
			yc_est[theta500]=mf_data[xy[theta500][0],xy[theta500][1]]
			snr[theta500]=(mf_data/err[theta500])[xy[theta500][0],xy[theta500][1]]
	
		bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln(snr,yc_est,err,xy)
		bf_soln=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]
		
		return bf_soln,[snr,yc_est,err,self.theta_array,xy]

	def mf_real_iterative(self,data,snr_thr=5,data_ft=[]):
		'''
		Fully agnostic to cluster characteristics and also to noise properties
		'''
		if data_ft==[]:
			data_ft=fsa.map2alm(data,self.reso)

		avg_ell,avg_Cl=fsa.alm2cl(data_ft,pixel_size=self.reso,smwin=self.smwin)
		Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)

		snr=collections.OrderedDict()
		err=collections.OrderedDict()
		yc_est=collections.OrderedDict()
		xy=collections.OrderedDict()
		for theta500 in self.theta_array:
			template_ft=self.tmplt_bank_ft[theta500]
			norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err[theta500]=np.sqrt(1./abs(norm))
			mf=(np.conj(template_ft)/Pk)/norm
			mf_data=fsa.alm2map(mf*data_ft,self.reso)
			snr_map=mf_data/err[theta500]
			loc=np.where(snr_map==max(snr_map.ravel())) ; xy[theta500]=[loc[0][0],loc[1][0]]
			yc_est[theta500]=mf_data[xy[theta500][0],xy[theta500][1]]
			snr[theta500]=(mf_data/err[theta500])[xy[theta500][0],xy[theta500][1]]

#		idx=np.where(max(snr.values())==snr.values())[0][0]
#		theta500_est=snr.keys()[idx]
#		xy_est=xy[theta500_est]

		bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln(snr,yc_est,err,xy)
		bf_soln=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]

		if bf_snr>=snr_thr:
			tmplt=sc.gen_field_cluster_template(bf_xy[0],bf_xy[1],bf_theta500,bf_yc,self.npix,self.reso,cutoff=10.,profile="GNFW")
			tmplt_ft=fsa.map2alm(tmplt,pixel_size=self.reso)
			rev_data_ft=data_ft-tmplt_ft
			#			rev_data_ft=fsa.map2alm(data-tmplt,pixel_size=self.reso)
			avg_ell,avg_Cl=fsa.alm2cl(rev_data_ft,pixel_size=self.reso,smwin=self.smwin)
			Pk=fsa.get_fourier_filter(avg_Cl,self.npix,pixel_size=self.reso,ell=avg_ell)

			snr=collections.OrderedDict()
			err=collections.OrderedDict()
			yc_est=collections.OrderedDict()
			xy=collections.OrderedDict()
			for theta500 in self.theta_array:
				template_ft=self.tmplt_bank_ft[theta500]
				norm=np.sum(np.conj(template_ft)*template_ft/Pk)*self.d2k ; err[theta500]=np.sqrt(1./abs(norm))
				mf=(np.conj(template_ft)/Pk)/norm
				mf_data=fsa.alm2map(mf*data_ft,self.reso)
				snr_map=mf_data/err[theta500]
				loc=np.where(snr_map==max(snr_map.ravel())) ; xy[theta500]=[loc[0][0],loc[1][0]]
				yc_est[theta500]=mf_data[xy[theta500][0],xy[theta500][1]]
				snr[theta500]=(mf_data/err[theta500])[xy[theta500][0],xy[theta500][1]]

			bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln(snr,yc_est,err,xy)
			bf_soln=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]
			#			bf_snr,bf_yc,bf_theta500,bf_err,bf_xy=self.return_best_soln_cdof(snr,yc_est,err,xy,dof=3)
			#			bf_soln_cdof=[bf_snr,bf_yc,bf_err,bf_theta500,bf_xy]
		return bf_soln,[snr,yc_est,err,self.theta_array,xy]


	def return_best_soln(self,snr,yc_est,err,xy):
		fn=interp1d(snr.keys(),snr.values(),kind="linear")
		temp_fine=fn(self.theta500_fine) ; max_snr=max(temp_fine) ; idx=np.where(max_snr==temp_fine)[0][0]
		theta500_est=self.theta500_fine[idx]
		
		#yc_estimate
		# Interpolating in y_c is not a good idea, since it is not a smooth function.
		#fn=interp1d(yc_est.keys(),yc_est.values(),kind="linear")
		#yc=fn(theta500_est)
		theta_loc=np.where(abs(theta500_est-self.theta_array)==min(abs(theta500_est-self.theta_array)))[0][0]
		yc=yc_est.values()[theta_loc]
		
		#error estimate
		fn=interp1d(err.keys(),err.values(),kind="linear")
		err_est=fn(theta500_est)

#		## In principle would require a 2D interpolation.
		if isinstance(xy,dict):
			idx=np.where(max(snr.values())==snr.values())[0][0]
			xy_est=xy[snr.keys()[idx]]
		else:
			xy_est=xy

		return max_snr,yc,theta500_est,err_est,xy_est

	def return_best_soln_cdof(self,snr,yc_est,err,xy,dof):
		fn=interp1d(snr.keys(),snr.values()**2. - dof,kind="linear")
		temp_fine=fn(self.theta500_fine) ; max_snr=np.sqrt(max(temp_fine)) ; idx=np.where(max_snr==temp_fine)[0][0]
		theta500_est=self.theta500_fine[idx]
		
		#yc_estimate
		fn=interp1d(yc_est.keys(),yc_est.values(),kind="linear")
		yc=fn(theta500_est)
		
		#error estimate
		fn=interp1d(err.keys(),err.values(),kind="linear")
		err_est=fn(theta500_est)

#		## In principle would require a 2D interpolation.
		if isinstance(xy,dict):
			idx=np.where(max(snr.values())==snr.values())[0][0]
			xy_est=xy[snr.keys()[idx]]
		else:
			xy_est=xy

		return max_snr,yc,theta500_est,err_est,xy_est
