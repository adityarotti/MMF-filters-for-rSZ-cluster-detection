##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################
import os
import healpy as h
import numpy as np
import socket
hostname = socket.gethostname()

mmfset=None

def setup_mmf_config(outpath,tempdatapath,dataset="planck_pr3",nside=2048,xsize=10.,pwc=True,channels=[],chmin=[],do_band_pass=True,gen_paths=True,use_psf_data=True):
	global mmfset
	mmfset=setup_mmf_analysis(dataset=dataset,nside=nside,xsize=xsize,pwc=pwc,channels=channels,chmin=chmin,do_band_pass=do_band_pass,outpath=outpath,tempdatapath=tempdatapath,use_psf_data=use_psf_data)
	if gen_paths:
		mmfset.init_paths()

class setup_mmf_analysis(object):
	def __init__(self,dataset="planck_pr3",nside=2048,xsize=10.,pwc=True,channels=[],chmin=[],do_band_pass=True,outpath="",tempdatapath="",use_psf_data=True):
		if dataset=="planck_pr3":
			from experiments import planck_pr3 as planck
			self.__dict__=planck.__dict__.copy()
		elif dataset=="planck_pr1":
			from experiments import planck_pr1 as planck
			self.__dict__=planck.__dict__.copy()
		elif dataset=="planck_psm_sim":
			from experiments import planck_psm_sim
			self.__dict__=planck_psm_sim.__dict__.copy()
		elif dataset=="pico":
			from experiments import pico_sims
			self.__dict__=pico_sims.__dict__.copy()

		if (channels==[]) & (chmin==[]):
			self.channels=np.copy(self.all_channels)
		elif (channels==[]):
			chmin_idx=np.where(np.array(self.all_channels)==chmin)[0][0]
			self.channels=np.copy(self.all_channels[chmin_idx:])
		else:
			self.channels=channels

		self.dataset=dataset
		self.nside=nside
		if dataset=="pico":
			self.nside=4096
		
		self.use_psf_data=use_psf_data
		self.xsize=xsize
		self.pwc=pwc
		self.outpath=outpath
		self.tempdatapath=tempdatapath
		self.do_band_pass=do_band_pass

		from flat_sky_codes import tangent_plane_analysis as tpa
		projection_operator=tpa.tangent_plane_setup(self.nside,self.xsize,0.,0.,rescale=1.)
		self.reso=projection_operator.pixel_size # arcminutes
		self.npix=projection_operator.npix

		self.mask_planck_maps=True
		self.mask_tangent_planes=True
		self.paths["sz_spec"]=os.path.abspath("../modules/simulate/spectral_template/sz_spectra/")

		globaloutpath="/Users/adityarotti/Documents/Work/Projects/Relativistic-SZ/MMF-filters-for-rSZ-cluster-detection/Planck_MMF/"
		if hostname=="sirius.jb.man.ac.uk" or hostname=="sirius3.jb.man.ac.uk":
			globaloutpath="/nvme/arotti/mmf_dataout/"

		# Setting tempdataout paths
		self.tempdatapath=globaloutpath + self.tempdatapath + str(int(xsize)) + "deg_patches/"
		self.paths["templates"]=self.tempdatapath + "/template_bank/"
		self.paths["tplanes"]=self.tempdatapath + "/tangent_planes/"

		# Setting result paths
		self.result_path=globaloutpath + self.outpath + str(int(xsize)) + "deg_patches/"
		self.paths["result_data"]=self.result_path + "/data/"
		self.paths["result_figs"]=self.result_path + "/figs/"

		# This ensures thet the point source masks from the low frequency channels are ignored.
		self.ps_mask_weights={}
		for ch in self.all_channels:
			self.ps_mask_weights[ch]=1.
			if ch<100.:
				self.ps_mask_weights[ch]=0.

	def init_paths(self):
		self.ensure_dir(self.paths["templates"])
		self.ensure_dir(self.paths["tplanes"])
		self.ensure_dir(self.paths["result_data"])
		self.ensure_dir(self.paths["result_figs"])

	def ensure_dir(self,file_path):
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
