##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created: 27 September 2018      															 #
# Date modified: 14 November 2018																 #
##################################################################################################
import numpy as np
import sympy as sp
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from settings import constants as cnst
from modules.settings import global_mmf_settings as gset
import sz_spec as szspec

class sz_spectrum(object):

	def __init__(self,datapath="",szspecpath="",channels=[30,44,70,100,143,217,353,545,857]):
		if datapath=="":
			self.datapath=gset.mmfset.paths["planck_bp"]
		else:
			self.datapath=datapath
		
		if szspecpath=="":
			self.szspecpath=gset.mmfset.paths["sz_spec"]
		else:
			self.szspecpath=szspecpath
		self.plbp={}
		self.channels=np.array(channels)
		self.wn2hz=1e-7*cnst.c_sol
		self.sz_op=szspec.sz_spectrum(szspecpath=self.szspecpath)
		self.conv_I2Tcmb=self.sz_op.fn["bb_diffT"](self.channels)
		self.return_band_pass()
		self.setup_fn_sz_2d_T()

	def return_band_pass(self,thr=1.e-6):
		hfi_im=fits.open(self.datapath + "/HFI_RIMO_R2.00.fits")
		lfi_im=fits.open(self.datapath + "/LFI_RIMO_R2.50.fits")
		
		plbp_raw={}

		# Extracting the LFI band passes
		for ch in self.channels[:3]:
			#print ch
			whichbp="BANDPASS_0" + str(ch)
			plbp_raw[ch]=[lfi_im[whichbp].data["WAVENUMBER"],lfi_im[whichbp].data["TRANSMISSION"]]

		# Extracting the HFI band passes
		for ch in self.channels[3:]:
			#print ch
			whichbp="BANDPASS_F" + str(ch)
			plbp_raw[ch]=[hfi_im[whichbp].data["WAVENUMBER"]*self.wn2hz,hfi_im[whichbp].data["TRANSMISSION"]]

		# Impose filter threshold and return band passes uniformly sampled in frequency
		for ch in self.channels:
			plbp_raw[ch][0]=plbp_raw[ch][0][plbp_raw[ch][1]>max(plbp_raw[ch][1])*thr]
			plbp_raw[ch][1]=plbp_raw[ch][1][plbp_raw[ch][1]>max(plbp_raw[ch][1])*thr]
			fn=interp1d(plbp_raw[ch][0],plbp_raw[ch][1],kind="linear")
			rf=np.linspace(min(plbp_raw[ch][0]),max(plbp_raw[ch][0]),100*len(plbp_raw[ch][0]))
			self.plbp[ch]=[rf,fn(rf)]

	def setup_fn_sz_2d_T(self,intkind="quintic"):
		# Evaluating the channel normalization for the band pass integrations.
		self.bp_norm=np.zeros(np.size(self.channels),float)
		for i,ch in enumerate(self.channels):
			self.bp_norm[i]=np.sum(self.plbp[ch][1]*self.sz_op.fn["bb_diffT"](self.plbp[ch][0]))
			
		# Evaluating the band passed spectra.
		self.YTeT=np.zeros((np.size(self.channels),np.size(self.sz_op.T)),float)
		for j,T in enumerate(self.sz_op.T):
			for i,ch in enumerate(self.channels):
				self.YTeT[i,j]=np.sum(self.plbp[ch][1]*self.sz_op.fn_sz_2d_I(T,self.plbp[ch][0]).flatten())/self.bp_norm[i]
		
		self.fn_sz_2d_T=interp2d(self.sz_op.T,self.channels,self.YTeT,kind=intkind,bounds_error=False,fill_value=0.)

	def return_sz_sed_template_bank(self,nu,Tmin,Tmax,Tstep):
		Te=np.arange(Tmin,Tmax+Tstep,Tstep,dtype="float")
		bank={}
		for T in Te:
			temp=self.fn_sz_2d_T(T,nu)[:,0]
			bank[T]={}
			for i,ch in enumerate(nu):
				bank[T][ch]=temp[i]
		return bank
