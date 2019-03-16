##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
import sympy as sp

T_cmb=2.726
h=6.62607004e-34
k=1.38064852e-23
c=299792458. # m/s

nu,T,c0,c1,c2=sp.symbols("nu T c0 c1 c2")
fn_symb={}
fn_symb["bb"]=c0*(nu**3.)/(sp.exp(c1*nu/T)-1.)
fn_symb["bb_diffT"]=fn_symb["bb"].diff(T,1)
fn_symb["Tcmb_TRJ"]=fn_symb["bb_diffT"]*c2/(nu**2.)

tempfn=fn_symb["Tcmb_TRJ"].subs(c0,(2.*h/(c**2.)))
tempfn=tempfn.subs(c1,(h/k))
tempfn=tempfn.subs(c2,(c**2./(2.*k)))
tempfn=tempfn.subs(nu,nu*1e9)
tempfn=tempfn.subs(T,T_cmb)
fn=sp.lambdify([nu],tempfn,modules="numpy")

def conv_KTBB_KTRJ(nu):
	return fn(nu)

def conv_KTRJ_KTBB(nu):
	return 1./fn(nu)

def conv_uKTRJ_KTBB(nu):
	return 1.e-6/fn(nu)

