import sys, os
from cosmology import cosmo_fn
from scipy.special import erfc
import bces.bces as bces
import numpy as np
from contextlib import contextmanager
import sys, os

def return_Y_M_fit(YSZ_500,YSZ_500_err,M500,M500_err,redshift,qcut=6.,mbias=[],use_approx_err=False,fidx=0,verbose=False):
	np.random.seed(0)

	Ezgamma=(cosmo_fn.Ez(redshift)**(-2./3.))/1e-4
	xdata=np.log10(M500/6.)

	if use_approx_err:
		qM500=M500/M500_err
		xerr=np.log10(1. + 1./qM500)
	else:
		xerr=return_log_err(M500/6.,M500_err/6.)

	N=np.float64(np.size(YSZ_500))

	if mbias==[]:
		mbias=np.ones_like(YSZ_500,dtype=np.float64)
		ombias=np.copy(mbias)

		iteration=0
		while (np.sum(abs(mbias-ombias)/ombias/N>1.e-3) or (iteration<1)) :
			ombias=np.copy(mbias)
			ydata=np.log10(YSZ_500*Ezgamma/ombias)
			if use_approx_err:
				qY500=YSZ_500/YSZ_500_err
				yerr=np.log10(1. + 1./qY500)
			else:
				yerr=return_log_err(YSZ_500*Ezgamma/ombias,YSZ_500_err*Ezgamma/ombias)
			with suppress_stdout():
				alpha,A,alpha_err,A_err,cov_alphaA=bces.bcesp(xdata,xerr,ydata,yerr,np.zeros_like(xdata),10000)
			dvar=yerr**2. + (alpha[fidx]**2.)*(xerr**2.)
			w=(N/dvar)/np.sum(1./dvar)
			dmm=ydata - alpha[fidx]*xdata - A[fidx]
			var_raw=np.sum(w*(dmm**2.))/(N-2.)
			var_stat=np.sum(yerr**2.)/N
			var_int=var_raw-var_stat
			var=np.float64(yerr**2. + var_int)

			q=YSZ_500/YSZ_500_err
			x= -np.log10(q/qcut)
			numerator=np.exp(-x**2./(2.*var))*np.sqrt(var)
			denominator=np.sqrt(np.pi/2.)*erfc(x/np.sqrt(2.*var))
			mbias=10.**(numerator/denominator)
			iteration=iteration+1
			if verbose:
				print alpha[3],alpha_err[3],A[3],A_err[3],np.sqrt(var_raw),np.sqrt(var_int)
	else:
		ydata=np.log10(YSZ_500*Ezgamma/mbias)
		if use_approx_err:
			qY500=YSZ_500/YSZ_500_err
			yerr=np.log10(1. + 1./qY500)
		else:
			yerr=return_log_err(YSZ_500*Ezgamma/mbias,YSZ_500_err*Ezgamma/mbias)
		with suppress_stdout():
			alpha,A,alpha_err,A_err,cov_alphaA=bces.bcesp(xdata,xerr,ydata,yerr,np.zeros_like(xdata),10000)
		dvar=yerr**2. + (alpha[fidx]**2.)*(xerr**2.)
		w=(N/dvar)/np.sum(1./dvar)
		dmm=ydata - alpha[fidx]*xdata - A[fidx]
		var_raw=np.sum(w*(dmm**2.))/(N-2.)
		var_stat=np.sum(yerr**2.)/N
		var_int=var_raw-var_stat
		if verbose:
			print alpha[3],alpha_err[3],A[3],A_err[3],np.sqrt(var_raw),np.sqrt(var_int)

	result={}
	result["alpha"]=alpha[3]
	result["alpha_err"]=alpha_err[3]
	result["A"]=A[3]
	result["A_err"]=A_err[3]
	result["sigma_raw"]=np.sqrt(var_raw)
	result["sigma_int"]=np.sqrt(var_int)
	result["sigma_stat"]=np.sqrt(var_stat)
	result["BIAS"]=mbias
	return result

def return_log_err(gauss_mean,gauss_err,num_samples=100000,ignore_negatives=True):
    logerr=np.zeros_like(gauss_mean)
    for idx, mu in enumerate(gauss_mean):
        x=np.random.normal(mu,gauss_err[idx],num_samples)
        if ignore_negatives:
            while np.any(x<0):
                neg_idx=np.where(x<0)[0]
                x[neg_idx]=np.random.normal(mu,gauss_err[idx],np.size(neg_idx))
        logerr[idx]=np.std(np.log10(x))
    return logerr


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
