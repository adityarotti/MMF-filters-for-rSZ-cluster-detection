import os
import healpy as h
import numpy as np

mmfset=None

def setup_mmf_config(dataset="planck",outpath="",nside=2048,xsize=10.,pwc=True,channels=[],chmin=[],result_midfix="",gen_paths=True):
	global mmfset
	mmfset=setup_mmf_analysis(dataset=dataset,outpath=outpath,nside=nside,xsize=xsize,pwc=pwc,channels=channels,chmin=chmin,result_midfix=result_midfix)
	if gen_paths:
		mmfset.init_paths()

class setup_mmf_analysis(object):
	def __init__(self,dataset="planck",outpath="",nside=2048,xsize=10.,pwc=True,channels=[],chmin=[],result_midfix=""):
		
		if dataset=="planck":
			from experiments import planck
			self.__dict__=planck.__dict__.copy()
		if dataset=="planck_psm_sim":
			from experiments import planck_psm_sim
			self.__dict__=planck_psm_sim.__dict__.copy()

		if (channels==[]) & (chmin==[]):
			self.channels=np.copy(self.planck_channels)
		elif (channels==[]):
			chmin_idx=np.where(np.array(self.planck_channels)==chmin)[0][0]
			self.channels=np.copy(self.planck_channels[chmin_idx:])
		else:
			self.channels=channels

		self.dataset=dataset
		self.nside=nside
		self.xsize=xsize
		self.pwc=pwc
		self.outpath=outpath
		self.result_midfix=result_midfix

		from flat_sky_codes import tangent_plane_analysis as tpa
		projection_operator=tpa.tangent_plane_setup(self.nside,self.xsize,0.,0.,rescale=1.)
		self.reso=projection_operator.pixel_size # arcminutes
		self.npix=projection_operator.npix

		self.mask_planck_maps=True
		self.mask_tangent_planes=True

		# Setting dataout paths
		self.paths["templates"]=self.outpath + "/data/template_bank/" + str(int(xsize)) + "deg_patches/"
		self.paths["tplanes"]=self.outpath + "/data/tangent_planes/" + str(int(xsize)) + "deg_patches/"

		# Setting result paths
		if self.result_midfix=="":
			self.result_path=self.outpath + "/results/" + str(int(xsize)) + "deg_patches/"
		else:
			self.result_path=self.outpath + "/results/" + "/" + self.result_midfix + "/" + str(int(xsize)) + "deg_patches/"

		self.paths["result_data"]=self.result_path + "/data/"
		self.paths["result_figs"]=self.result_path + "/figs/"


		# This ensures thet the point source masks from the low frequency channels are ignored.
		self.ps_mask_weights={}
		for ch in self.planck_channels:
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

