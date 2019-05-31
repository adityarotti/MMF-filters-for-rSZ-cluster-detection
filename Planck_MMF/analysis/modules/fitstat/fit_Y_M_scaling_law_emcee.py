import emcee
import corner
import numpy as np
from cosmology import cosmo_fn
from scipy.special import erfc
from matplotlib import pyplot as plt

def return_Y_M_fit(YSZ_500,YSZ_500_err,M500,M500_err,redshift,mbias=[],qcut=6.,min_type="SLS",nwalkers=50,nsamples=2000,burnin=500,use_approx_err=False,ana_corr=False,verbose=False):
	np.random.seed(0)
	
	solve_mbias=False
	if mbias==[]:
		mbias=np.ones_like(YSZ_500,dtype=np.float64)
		ombias=np.copy(mbias)
		solve_mbias=True

	Ezgamma=(cosmo_fn.Ez(redshift)**(-2./3.))/1e-4
	xdata=np.log10(M500/6.)
	if ana_corr:
		Tx=cosmo_fn.convert_M500_T500(M500,redshift)
		xdata=xdata+np.log10(1.-0.07*Tx/5.)
	ydata=np.log10(YSZ_500*Ezgamma/mbias)
	if use_approx_err:
		xerr=np.log10(1. + M500_err/M500)
		yerr=np.log10(1. + YSZ_500_err/YSZ_500)
	else:
		xerr=return_log_err(M500/6.,M500_err/6.)
		yerr=return_log_err(YSZ_500*Ezgamma,YSZ_500_err*Ezgamma)

	if solve_mbias:
		if verbose:
			print "Solving for Malmquist bias"
		iteration=0
		while((np.sum(abs(mbias-ombias)/ombias/np.size(mbias))>1.e-3) or (iteration<1)):
			#print np.sum(abs(mbias-ombias)/ombias/np.size(mbias))
			mbias=np.copy(ombias)
			ydata=np.log10(YSZ_500*Ezgamma/mbias)
			result=return_emcee_fit(xdata,ydata,xerr,yerr,min_type=min_type,nwalkers=nwalkers,nsamples=nsamples,burnin=burnin)
			ombias=return_malmquist_bias(YSZ_500/YSZ_500_err,yerr,result["param"][2],qcut=qcut)
			iteration=iteration+1
		result["BIAS"]=ombias
	else:
		ydata=np.log10(YSZ_500*Ezgamma/mbias)
		result=return_emcee_fit(xdata,ydata,xerr,yerr,min_type=min_type,nwalkers=nwalkers,nsamples=nsamples,burnin=burnin)
		result["BIAS"]=mbias
	return result

def return_emcee_fit(xdata,ydata,xerr,yerr,min_type="SLS",nwalkers=50,nsamples=2000,burnin=500):
	'''
	This is currently setup to carry out a linear fit and solve for the intrinsic scatter in log_10(Y).
	min_type: minimization type. SLS -- > Simple least squares. ; ORTH --> Minimizes orthogonal distance.
	'''
	
	if min_type=="SLS":
		def lnlike(param, x, y, xerr,yerr):
			alpha, A, log10_log_yint = param
			model = alpha * x + A
			inv_sigma2 = 1.0/(yerr**2 + alpha**2*xerr**2. + 10.**(2.*log10_log_yint))
			dmm=(y-model)
			return -0.5*(np.sum(dmm**2.*inv_sigma2 - np.log(inv_sigma2)))
	elif min_type=="ORTH":
		def lnlike(param, x, y, xerr,yerr):
			alpha, A, log10_log_yint = param
			model = alpha * x + A
			inv_sigma2 = 1.0/((yerr**2 + alpha**2*xerr**2.)/(1. + alpha**2.) + (10.**(2.*log10_log_yint))*np.cos(np.arctan(alpha))**2.)
			dmm=(y-model)/np.sqrt(1. + alpha**2.)
			return -0.5*(np.sum(dmm**2.*inv_sigma2 - np.log(inv_sigma2)))

	def lnprior(param):
		alpha, A, log10_log_yint = param
		if 0. < alpha < 4. and -2.0 < A < 2.0 and -2. < log10_log_yint < 0.:
			return 0.0
		return -np.inf

	def lnprob(param, x, y, xerr, yerr):
		lp = lnprior(param)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlike(param, x, y, xerr, yerr)

	ndim=3
	pos = [[0.5,0.2,0.1]*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata,xerr, yerr))
	temp=sampler.run_mcmc(pos, nsamples)
	samples = sampler.chain[:, burnin:, :].reshape((-1, ndim)) ; samples[:, 2] = 10.**(samples[:, 2])

	alpha, A, log_yint = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
	zip(*np.percentile(samples, [15.87, 50., 84.13],
	axis=0)))

	best_fit_param=np.zeros(ndim,float)
	cov_mat=np.zeros((ndim,ndim),float)
	corr_mat=np.zeros((ndim,ndim),float)

	for i in range(ndim):
		best_fit_param[i]=np.mean(samples[:,i])
		erri=np.std(samples[:,i])
		for j in range(ndim):
			errj=np.std(samples[:,j])
			cov_mat[i,j]=np.mean((samples[:,i]-np.mean(samples[:,i]))*(samples[:,j]-np.mean(samples[:,j])))
			corr_mat[i,j]=cov_mat[i,j]/(erri*errj)

	result={}
	result["param"]=best_fit_param
	result["cov_mat"]=cov_mat
	result["corr_mat"]=corr_mat
	result["samples"]=samples

	return result

def return_malmquist_bias(qYSZ,log_yerr,log_yint,qcut=6):
	var=log_yerr**2. + log_yint**2.
	x= -np.log10(qYSZ/qcut)
	numerator=np.exp(-x**2./(2.*var))*np.sqrt(var)
	denominator=np.sqrt(np.pi/2.)*erfc(x/np.sqrt(2.*var))
	mbias=10.**(numerator/denominator)
	return mbias

def gen_corner_plot(samples,labels=[r"$\alpha$", "$A$", r"$\sigma_{{\rm Log}Y|M}$"],figttl="",figname="",numdec=3):
	temp,ndim=np.shape(samples)
	param=np.zeros(ndim,float) ; param_err=np.zeros(ndim,float)
	corr_mat=np.zeros((ndim,ndim),float)
	
	for i in range(ndim):
		param[i]=np.mean(samples[:,i])
		param_err[i]=np.std(samples[:,i])
		for j in range(ndim):
			param_err[j]=np.std(samples[:,j])
			corr_mat[i,j]=np.mean((samples[:,i]-np.mean(samples[:,i]))*(samples[:,j]-np.mean(samples[:,j])))/(param_err[i]*param_err[j])

	fig=corner.corner(samples, labels=[r"$\alpha$", "$A$", r"$\sigma_{{\rm Log}Y|M}$"])
	fig.text(0.71,0.63,figttl)
	fig.text(0.71,0.6,r"$\alpha=$" + str(round(param[0],numdec)) + "$\pm$" + str(round(param_err[0],numdec)))
	fig.text(0.71,0.57,r"$A=$" + str(round(param[1],numdec)) + "$\pm$" + str(round(param_err[1],numdec)))
	fig.text(0.71,0.54,r"$\sigma_{{\rm Log}Y|M}=$" + str(round(param[2],numdec)) + "$\pm$" + str(round(param_err[2],numdec)))
	fig.text(0.71,0.51,r"$ \rm{Corr}(\alpha,A)=$" + str(round(corr_mat[0,1],numdec)))
	fig.text(0.71,0.48,r"$ \rm{Corr}(\alpha,\sigma_{{\rm Log}Y|M})=$" + str(round(corr_mat[0,2],numdec)))
	fig.text(0.71,0.45,r"$ \rm{Corr}(A,\sigma_{{\rm Log}Y|M})=$" + str(round(corr_mat[1,2],numdec)))
	fig.text(0.53,0.86,r"$\rm{log}_{10}\left(E_z^{-2/3} \frac{y^{rSZ}}{10^{-4}}\right) = A + \alpha \rm{log}_{10}\left(\frac{M_x}{6}\right)$",fontsize=12)
	if figname!="":
		plt.savefig(figname,bbox_inches="tight")

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

