import numpy as np
import healpy as h

from simulate.spectral_template import sz_spec as szsed
from simulate.spatial_template import sim_cluster as sc
from flat_sky_codes import flat_sky_analysis as fsa

def run_multi_matched_filter(data,R500,channels,pixel_size,fwhm={},T=0.,smwin=5,return_Pk=False,pwc_nside=[],cutoff=10.,profile="GNFW",datapath=""):
	lmax=180./(pixel_size/60.)
	nxpix=data.shape[1] ; nypix=data.shape[2] ; totnpix=nxpix*nypix
	d2x=((pixel_size/60.)*np.pi/180.)**2. ; d2k= 1./(nxpix*nypix*d2x)
	numch=np.size(channels)
	
	# Getting the Planck band passed SZ spectrum
	plbp_sz_spec=szsed.return_planck_bp_sz_spec(T,datapath=datapath)
	
	template=sc.gen_cluster_template(nxpix,R500,pixel_size,cutoff=cutoff,profile=profile)
	
	data_ft=np.zeros((numch,nxpix,nxpix),complex)
	cross_Pk=np.zeros((totnpix,numch,numch),np.float64)
	template_ft=np.zeros((totnpix,numch),np.complex)
	cluster_ft=fsa.map2alm(np.fft.fftshift(template),pixel_size)
	
	if fwhm!={}:
		if pwc_nside!=[]:
			pwc=h.pixwin(pwc_nside,pol=False)
			ellpwc=np.arange(np.size(pwc),dtype="float")
		else:
			pwc=np.ones(1000,float)
			ellpwc=np.arange(np.size(pwc),dtype="float")
		
		for i,ch in enumerate(channels):
			data_ft[i,]=fsa.map2alm(data[i,],pixel_size)
			for j in range(i+1):
				ell,cl=fsa.alm2cl(alm=data_ft[i,],almp=data_ft[j,],pixel_size=pixel_size,smwin=smwin)
				filtr=fsa.get_fourier_filter(cl,nxpix,pixel_size,ell=ell)
				cross_Pk[:,i,j]=filtr.reshape(totnpix)
				cross_Pk[:,j,i]=cross_Pk[:,i,j]
			ellp,bl=fsa.get_gauss_beam(fwhm[ch],2*lmax)
			chpwc=np.interp(ellp,ellpwc,pwc) ; bl=bl*chpwc
			cluster_ft=fsa.map2alm(np.fft.fftshift(template),pixel_size)
			template_ft[:,i]=(fsa.filter_alm(cluster_ft,pixel_size,bl=bl,ell=ellp)*plbp_sz_spec[ch]).reshape(totnpix)
	else:
		for i,ch in enumerate(channels):
			data_ft[i,]=fsa.map2alm(data[i,],pixel_size)
			for j in range(i+1):
				ell,cl=fsa.alm2cl(alm=data_ft[i,],almp=data_ft[j,],pixel_size=pixel_size,smwin=smwin)
				filtr=fsa.get_fourier_filter(cl,nxpix,pixel_size,ell=ell)
				cross_Pk[:,i,j]=filtr.reshape(totnpix)
				cross_Pk[:,j,i]=cross_Pk[:,i,j]
			template_ft[:,i]=(cluster_ft*plbp_sz_spec[ch]).reshape(totnpix)

	cross_Pk_inv=np.linalg.inv(cross_Pk)

	normk=np.einsum("ki,kij,kj->k",template_ft,cross_Pk_inv,np.conj(template_ft))
	norm=1./(np.sum(normk)*d2k) ; rec_err=np.sqrt(abs(norm))
	mmf=norm*np.einsum("kij,kj->ki",cross_Pk_inv,template_ft)
	mmf=mmf.reshape(nxpix,nxpix,numch)

	result_ft=np.zeros((nxpix,nxpix),complex)
	for i in range(numch):
		result_ft += mmf[:,:,i]*data_ft[i,:,:]

	result=fsa.alm2map(result_ft,pixel_size)

	if return_Pk:
		return result,rec_err,cross_Pk
	else:
		return result,rec_err



