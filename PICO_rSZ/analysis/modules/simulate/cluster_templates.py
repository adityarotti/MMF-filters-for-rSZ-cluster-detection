import os
import numpy as np
import healpy as h
from astropy.io import fits
from scipy.interpolate import interp1d

from settings import mmf_settings as mmfset
from spectral_template import sz_spec as szsed
from spatial_template import sim_cluster as sc
from flat_sky_codes  import flat_sky_analysis as fsa

class cluster_spectro_spatial_templates(object):

	def __init__(self,theta500_min=2.,theta500_max=55.,theta_step=1.,T_min=0.,T_max=15.,T_step=1.,profile="GNFW",cutoff=10.):
		self.theta500_min=theta500_min
		self.theta500_max=theta500_max
		self.theta_step=theta_step
		self.T_max=T_max
		self.T_min=T_min
		self.T_step=T_step
		self.profile=profile
		self.cutoff=cutoff
		self.theta500=np.arange(self.theta500_min,self.theta500_max+self.theta_step,self.theta_step,dtype="float")
		self.T500=np.arange(self.T_min,self.T_max+self.T_step,self.T_step,dtype="float")

	def setup_templates(self,gen_template=False):
		if gen_template:
			self.gen_template()
		self.gen_template_ft_bank()
		self.sz_op=szsed.sz_spectrum()
		self.sz_spec_bank=self.sz_op.return_sz_sed_template_bank(mmfset.channels,self.T_min,self.T_max,self.T_step)
		self.setup_fn_yerr_norm()
		self.setup_channel_beam_filters()

	def gen_template_bank(self):
		for thetac in self.theta500:
			template=self.gen_template(thetac)
			hdu = fits.ImageHDU()
			hdu.header["Comments"]="Template"
			hdu.header["R500"]=str(thetac) + " arcminutes"
			hdu.header["profile"]=profile
			hdu.data=template
			filename=mmfset.paths["templates"] + "cluster_" +str(thetac) + ".fits"
			hdu.writeto(filename,overwrite=True)

	def gen_template(self,thetac):
		template=sc.gen_cluster_template(mmfset.npix,thetac,pixel_size=mmfset.reso,y0=1.,cutoff=self.cutoff,dorandom=False,profile=self.profile)
		return template

	def gen_template_ft_bank(self):
		self.sp_ft_bank={}
		for thetac in self.theta500:
			template=self.get_template(thetac)
			self.sp_ft_bank[thetac]=fsa.map2alm(np.fft.fftshift(template),mmfset.reso)

	def get_template(self,thetac,getheader=False):
		filename=mmfset.paths["templates"] + "cluster_" +str(thetac) + ".fits"
		file_exists = os.path.isfile(filename)

		if file_exists:
			template=fits.getdata(filename)
		
			if getheader:
				header=fits.getheader(filename,1)
				return template, header
			else:
				return template
		else:
			print filename, " does not exist."
			print "generating the template to return"
			
			template=sc.gen_cluster_template(mmfset.npix,thetac,pixel_size=mmfset.reso,y0=1.,cutoff=10.,dorandom=False,profile="GNFW")
			hdu = fits.ImageHDU()
			hdu.header["Comments"]="Template"
			hdu.header["R500"]=str(thetac) + " arcminutes"
			hdu.header["profile"]=self.profile
			hdu.data=template
			hdu.writeto(filename,overwrite=True)
			
			return template

	def setup_fn_yerr_norm(self):
		noise_norm=np.zeros(np.size(self.theta500))
		for idx,thetac in enumerate(self.theta500):
			template=self.get_template(thetac)
			noise_norm[idx]=np.sum(template)*mmfset.reso*mmfset.reso
		self.fn_yerr_norm=interp1d(self.theta500,noise_norm,bounds_error=False,fill_value=0.)


	def setup_channel_beam_filters(self):
		lmax=4*mmfset.nside
		
		pwc=h.pixwin(mmfset.nside,pol=False)
		ellpwc=np.arange(np.size(pwc),dtype="float")
		if not(mmfset.pwc):
			pwc[:]=1.
		
		blp={}
		for ch in mmfset.channels:
			ellp,temp_bl=fsa.get_gauss_beam(mmfset.fwhm[ch],lmax=lmax)
			chpwc=np.interp(ellp,ellpwc,pwc)
			blp[ch]=temp_bl*chpwc

		self.chfiltr={}
		for ch in mmfset.channels:
			self.chfiltr[ch]=fsa.get_fourier_filter(blp[ch],mmfset.npix,mmfset.reso,ell=ellp)
