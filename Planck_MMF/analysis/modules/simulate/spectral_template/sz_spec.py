##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
import sympy as sp
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from settings import constants as cnst
from modules.settings import global_mmf_settings as gset


class sz_spectrum(object):
	def __init__(self,szspecpath=""):
		if szspecpath=="":
			self.szspecpath=gset.mmfset.paths["sz_spec"]
		else:
			self.szspecpath=szspecpath

		self.setup_bb_fns()
		self.setup_fn_2d_sz_spec()

	def setup_bb_fns(self):
		nu,T,c0,c1=sp.symbols("nu T c0 c1")
		self.fn_symb={}
		self.fn_symb["bb"]=c0*(nu**3.)/(sp.exp(c1*nu/T)-1.)
		self.fn_symb["bb_diffT"]=self.fn_symb["bb"].diff(T,1)
		self.fn_symb["SZ"]=cnst.T_cmb*self.fn_symb["bb_diffT"]*(((c1*(nu/T))*(sp.exp(c1*(nu/T))+1.)/(sp.exp(c1*(nu/T))-1.))-4.)
		
		self.fntype=self.fn_symb.keys()
		self.fn={}
		for ft in self.fntype:
			tempfn=self.fn_symb[ft].subs(c0,(2.*cnst.h_planck/(cnst.c_sol**2.)))
			tempfn=tempfn.subs(c1,(cnst.h_planck/cnst.k_boltzmann))
			tempfn=tempfn.subs(nu,nu*1e9)
			tempfn=tempfn.subs(T,cnst.T_cmb)
			self.fn[ft]=sp.lambdify([nu],tempfn,modules="numpy")

	def setup_fn_2d_sz_spec(self,intkind="quintic"):
		filename=self.szspecpath + "/SZ_CNSN_basis_2KeV.dat"
		data=np.loadtxt(filename)
		self.sz_nu=data[:,0]*cnst.k_boltzmann*cnst.T_cmb/cnst.h_planck/1e9
		conv_I2Tcmb=self.fn["bb_diffT"](self.sz_nu)
		self.YTeI=np.zeros((np.size(self.sz_nu),50),float)
		self.YTeT=np.zeros((np.size(self.sz_nu),50),float)
		
		
		# Evaluating the 0KeV case
		self.T=[0]
		self.YTeI[:,0]=self.fn["SZ"](self.sz_nu)
		self.YTeT[:,0]=self.YTeI[:,0]/conv_I2Tcmb
		
		# Remember that 1KeV file is missing.
		for i in range(49):
			Temperature=i+2 # KeV
			self.T=np.append(self.T,Temperature)
			norm=1e4*0.01*Temperature/511.
			filename=self.szspecpath + "/SZ_CNSN_basis_" + str(Temperature) + "KeV.dat"
			data=np.loadtxt(filename)
			self.YTeI[:,i+1]=data[:,2]/norm/1.e16
			self.YTeT[:,i+1]=self.YTeI[:,i+1]/conv_I2Tcmb
		
		self.fn_sz_2d_I=interp2d(self.T,self.sz_nu,self.YTeI,kind=intkind,bounds_error=False,fill_value=0.)
		self.fn_sz_2d_T=interp2d(self.T,self.sz_nu,self.YTeT,kind=intkind,bounds_error=False,fill_value=0.)

	def return_sz_sed_template_bank(self,nu,Tmin,Tmax,Tstep):
		Te=np.arange(Tmin,Tmax+Tstep,Tstep,dtype="float")
		bank={}
		for T in Te:
			temp=self.fn_sz_2d_T(T,nu)[:,0]
			bank[T]={}
			for i,ch in enumerate(nu):
				bank[T][ch]=temp[i]
		return bank
