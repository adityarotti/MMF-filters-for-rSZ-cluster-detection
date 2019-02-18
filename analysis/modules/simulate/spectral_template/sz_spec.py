import numpy as np
import planck_band_pass_sz as pbpsz

def sz_spec(nu,T=0.):
	'''
	nu : Observing frequency in GHz.
	'''
	Tcmb=2.7255       # Kelvin
	kb=1.38064852e-23 # Boltzmann constant
	h=6.62607004e-34  # Planck constant
	x=h*nu*1e9/(kb*Tcmb)
	y=(x*(np.exp(x)+1.)/(np.exp(x)-1.))-4.
	return y

def return_planck_bp_sz_spec(T=0.,datapath=""):
	bp=pbpsz.band_pass_filtering(datapath=datapath)
	temp=bp.fn_sz_2d_bp(T)
	sz_spec={}
	for i,ch in enumerate(bp.channels):
		sz_spec[ch]=temp[i]
	return sz_spec

def return_pbp_sz_sed_template_bank(Tmin,Tmax,Tstep,datapath=""):
	bp=pbpsz.band_pass_filtering(datapath=datapath)
	Te=np.arange(Tmin,Tmax+Tstep,Tstep,dtype="float")
	bank={}
	for T in Te:
		temp=bp.fn_sz_2d_bp(T)
		bank[T]={}
		for i,ch in enumerate(bp.channels):
			bank[T][ch]=temp[i]
	return bank

def return_sz_sed_template_bank(nu,Tmin,Tmax,Tstep,datapath=""):
	bp=pbpsz.band_pass_filtering(datapath=datapath)
	fn=bp.fn["bb_diffT"]
	Te=np.arange(Tmin,Tmax+Tstep,Tstep,dtype="float")
	bank={}
	for T in Te:
		temp=bp.fn_sz_2d(T,nu)[:,0]/fn(nu)
		bank[T]={}
		for i,ch in enumerate(nu):
			bank[T][ch]=temp[i]
	return bank
