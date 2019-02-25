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
from settings import mmf_settings as mmfset

class band_pass_filtering(object):

	def __init__(self,datapath="",szspecpath="",channels=[30,44,70,100,143,217,353,545,857]):
		if datapath=="":
			self.datapath=mmfset.paths["planck_bp"]
		else:
			self.datapath=datapath
		
		if szspecpath=="":
			self.szspecpath=mmfset.paths["sz_spec"]
		else:
			self.szspecpath=szspecpath
		self.plbp={}
		self.channels=np.array(channels)
		
		# Defining the necessary constants
		self.T_cmb=2.726
		self.h=6.62607004e-34
		self.k=1.38064852e-23
		self.c=299792458. # m/s
		self.wn2hz=1e-7*self.c
		
		self.setup_bb_fns()
		self.return_band_pass()
		self.setup_num_sz_spec()

	def setup_bb_fns(self):
		nu,T,c0,c1=sp.symbols("nu T c0 c1")
		self.fn_symb={}
		self.fn_symb["bb"]=c0*(nu**3.)/(sp.exp(c1*nu/T)-1.)
		self.fn_symb["bb_diffT"]=self.fn_symb["bb"].diff(T,1)
		self.fn_symb["SZ"]=self.T_cmb*self.fn_symb["bb_diffT"]*(((c1*(nu/T))*(sp.exp(c1*(nu/T))+1.)/(sp.exp(c1*(nu/T))-1.))-4.)
		
		self.fntype=self.fn_symb.keys()
		self.fn={}
		for ft in self.fntype:
			tempfn=self.fn_symb[ft].subs(c0,(2.*self.h/(self.c**2.)))
			tempfn=tempfn.subs(c1,(self.h/self.k))
			tempfn=tempfn.subs(nu,nu*1e9)
			tempfn=tempfn.subs(T,self.T_cmb)
			self.fn[ft]=sp.lambdify([nu],tempfn,modules="numpy")

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

	def setup_num_sz_spec(self,intkind="quintic"):
		T_cmb_SZpack=self.T_cmb #2.726 ; print T_cmb_SZpack
		filename=self.szspecpath + "/SZ_CNSN_basis_2KeV.dat"
		data=np.loadtxt(filename)
		self.sz_nu=data[:,0]*self.k*T_cmb_SZpack/self.h/1e9
		
		self.YTe=np.zeros((np.size(self.sz_nu),50),float)
	
		# Evaluating the 0KeV case
		self.T=[0]
		self.YTe[:,0]=self.fn["SZ"](self.sz_nu)
		
		# Remember that 1KeV file is missing.
		for i in range(49):
			Temperature=i+2 # KeV
			self.T=np.append(self.T,Temperature)
			norm=1e4*0.01*Temperature/511.
			filename=self.szspecpath + "/SZ_CNSN_basis_" + str(Temperature) + "KeV.dat"
			data=np.loadtxt(filename)
			self.YTe[:,i+1]=data[:,2]*((self.T_cmb/T_cmb_SZpack)**4.)/norm/1.e16
		
		self.fn_sz_2d=interp2d(self.T,self.sz_nu,self.YTe,kind=intkind,bounds_error=False,fill_value=0.)
		
	def fn_sz_2d_bp(self,T=0.):
		do_flatten=False
		if not isinstance(T,(list,np.ndarray)):
			T=[T]
			do_flatten=True
		bp_ysz=np.zeros((np.size(T),np.size(self.channels)),float)
		for j,temp in enumerate(T):
			for i,ch in enumerate(self.channels):
				bp_ysz[j,i]=np.sum(self.plbp[ch][1]*self.fn_sz_2d(temp,self.plbp[ch][0]).flatten())
				bp_ysz[j,i]=bp_ysz[j,i]/np.sum(self.plbp[ch][1]*self.fn["bb_diffT"](self.plbp[ch][0]))
		if do_flatten:
			bp_ysz=bp_ysz.flatten()
		return np.transpose(bp_ysz) # So that the return format is same as fn_sz_2d

	def write_sz_spec(self,T=0.,datapath=""):
		if datapath=="":
			datapath="./dataout/"

		filename=open(datapath + "/planck_sz_spectrum_T=" + str(T) +"keV.txt","wb")
		wbp=self.fn_sz_2d_bp(T=T) ; wobp=self.fn_sz_2d_bp(T=T)
		filename.write("%-15s %-15s %-15s \n \n" % ("# Freq.","With BP","Without BP"))
		for i,ch in enumerate(self.channels):
			filename.write("%-15.8f %-15.8f %-15.8f\n" % (ch,wbp[i],wobp[i]))
		filename.close()
