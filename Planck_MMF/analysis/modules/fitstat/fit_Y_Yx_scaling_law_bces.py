import sys, os
import numpy as np
import bces.bces as bces
from scipy.special import erfc
from contextlib import contextmanager

# How do you compute the error on the raw scatter?
# How do you compute the error on the mean measurement error?
# Propogating errors to the error on the intrinsic scatter.

def return_Y_Yx_fit(YSZ_500,YSZ_500_err,Yx,Yx_err,mbias=[],qcut=6.,fidx=3,verbose=False,use_approx_err=False):
	np.random.seed(0)

	solve_mbias=False
	if mbias==[]:
		mbias=np.ones_like(YSZ_500,dtype=np.float64)
		ombias=np.copy(mbias)
		solve_mbias=True

	xdata=np.log10(Yx/1.e-4)
	ydata=np.log10(YSZ_500/mbias)
	if use_approx_err:
		xerr=np.log10(1. + Yx_err/Yx)
		yerr=np.log10(1. + YSZ_500_err/YSZ_500)
	else:
		xerr=return_log_err(Yx/1.e-4,Yx_err/1.e-4)
		yerr=return_log_err(YSZ_500,YSZ_500_err)

	if solve_mbias:
		iteration=0
		while((np.sum(abs(mbias-ombias)/ombias/np.size(mbias))>1.e-3) or (iteration<1)):
			#print np.sum(abs(mbias-ombias)/ombias/np.size(mbias))
			mbias=np.copy(ombias)
			ydata=np.log10(YSZ_500/mbias)
			result=return_bces_fit(xdata,ydata,xerr,yerr,fidx=fidx)
			result=return_logy_int_scat(xdata,ydata,xerr,yerr,result)
			ombias=return_malmquist_bias(YSZ_500/YSZ_500_err,yerr,result["param"][2],qcut=qcut)
			iteration=iteration+1
		result["BIAS"]=ombias
	else:
		ydata=np.log10(YSZ_500/mbias)
		result=return_bces_fit(xdata,ydata,xerr,yerr,fidx=fidx)
		result=return_logy_int_scat(xdata,ydata,xerr,yerr,result)
		result["BIAS"]=mbias
	return result

def return_logy_int_scat(xdata,ydata,xerr,yerr,result):
	alpha=result["param"][0] ; A=result["param"][1]
	N=np.size(xdata)
	dvar=yerr**2. + (alpha**2.)*(xerr**2.)
	w=(N/dvar)/np.sum(1./dvar)
	dmm=ydata - alpha*xdata - A
	var_raw=np.sum(w*(dmm**2.))/(N-2.) ; var_raw_var=(1./np.sum(1./dvar))
	var_stat=np.mean(yerr**2.) ; var_stat_var=np.std(yerr**2.)**2.
	var_int=var_raw-var_stat ; var_int_var=(var_raw_var+var_stat_var)#/(4.*var_int)
	result["param"][2]=np.sqrt(var_int)
	result["cov_mat"][2,2]=var_int_var
	result["raw_var"]=np.array([np.sqrt(var_raw),np.sqrt(var_raw_var)/(2.*np.sqrt(var_raw))])
	result["stat_var"]=np.array([np.sqrt(var_stat),np.sqrt(var_stat)/(2.*np.sqrt(var_stat))])
	return result

def return_bces_fit(xdata,ydata,xerr,yerr,fidx=3):
	with suppress_stdout():
		alpha,A,alpha_err,A_err,cov_alphaA=bces.bcesp(xdata,xerr,ydata,yerr,np.zeros_like(xdata),10000)
	
	cov_mat=np.zeros((3,3),float)
	cov_mat[0,0]=alpha_err[fidx]**2.
	cov_mat[1,1]=A_err[fidx]**2.
	cov_mat[0,1]=cov_alphaA[fidx] ; cov_mat[1,0]=cov_mat[0,1]
	best_fit_param=np.zeros(3,float)
	best_fit_param[0] = alpha[fidx] ; best_fit_param[1]=A[fidx]
	result={}
	result["param"]=best_fit_param
	result["cov_mat"]=cov_mat
	return result

def return_malmquist_bias(qYSZ,log_yerr,log_yint,qcut=6):
	var=log_yerr**2. + log_yint**2.
	x= -np.log10(qYSZ/qcut)
	numerator=np.exp(-x**2./(2.*var))*np.sqrt(var)
	denominator=np.sqrt(np.pi/2.)*erfc(x/np.sqrt(2.*var))
	mbias=10.**(numerator/denominator)
	return mbias

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
