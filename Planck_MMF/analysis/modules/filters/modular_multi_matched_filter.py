##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from modules.settings import global_mmf_settings as gset
from flat_sky_codes import flat_sky_analysis as fsa
from masking import gen_masks as gm
from cosmology import cosmo_fn as cosmo_fn

class multi_matched_filter(object):

	def __init__(self,sp_ft_bank,sz_spec_bank,chfiltr,fn_yerr_norm,lmax_cutoff=20000):
		self.sp_ft_bank=sp_ft_bank
		self.sz_spec_bank=sz_spec_bank
		self.chfiltr=chfiltr
		self.fn_yerr_norm=fn_yerr_norm
		
		self.numch=np.size(gset.mmfset.channels)
		self.nxpix=gset.mmfset.npix ; self.totnpix=self.nxpix*self.nxpix
		self.d2x=((gset.mmfset.reso/60.)*np.pi/180.)**2. ; self.d2k= 1./(self.totnpix*self.d2x)
		
		# Low pass filtr
		ell=np.arange(20000) ; bl=np.ones(np.size(ell),float) ; bl[ell>=lmax_cutoff]=0.
		self.lpfiltr=fsa.get_fourier_filter(bl,nxpix=gset.mmfset.npix,pixel_size=gset.mmfset.reso,ell=ell)
	
		self.cmask=gm.return_center_mask()
	
	def get_data_ft(self,data,smwin=5):
		self.data_ft=np.zeros((self.numch,self.nxpix,self.nxpix),complex)
		self.cross_Pk=np.zeros((self.totnpix,self.numch,self.numch),np.float64)
		
		for i,ch in enumerate(gset.mmfset.channels):
			self.data_ft[i,]=fsa.map2alm(data[i,],gset.mmfset.reso)
			for j in range(i+1):
				ell,cl=fsa.alm2cl(alm=self.data_ft[i,],almp=self.data_ft[j,],pixel_size=gset.mmfset.reso,smwin=smwin)
				filtr=fsa.get_fourier_filter(cl,self.nxpix,gset.mmfset.reso,ell=ell)
				self.cross_Pk[:,i,j]=filtr.reshape(self.totnpix)
				self.cross_Pk[:,j,i]=self.cross_Pk[:,i,j]
				
		self.cross_Pk_inv=np.linalg.inv(self.cross_Pk)


	def evaluate_mmf(self,template_alm,sz_spec,optimize=False):
		template_ft=np.zeros((self.totnpix,self.numch),np.complex)
		for i,ch in enumerate(gset.mmfset.channels):
			template_ft[:,i]=(template_alm*self.chfiltr[ch]*sz_spec[ch]).reshape(self.totnpix)

		normk=np.einsum("ki,kij,kj->k",template_ft,self.cross_Pk_inv,np.conj(template_ft),optimize=optimize)
		norm=1./(np.sum(normk)*self.d2k) ; rec_err=np.sqrt(abs(norm))
		mmf=norm*np.einsum("kij,kj->ki",self.cross_Pk_inv,template_ft,optimize=optimize)
		mmf=mmf.reshape(self.nxpix,self.nxpix,self.numch)
		result_ft=np.zeros((self.nxpix,self.nxpix),complex)

		for i in range(self.numch):
			result_ft += mmf[:,:,i]*self.data_ft[i,:,:]*self.lpfiltr
			
		result=fsa.alm2map(result_ft,gset.mmfset.reso)
		return result,rec_err

	def return_snr(self,thetac,Tc,maskthr=3.,mask_fdata=True,psmask=[],write_data=False,filename=[]):
		fdata,err=self.evaluate_mmf(self.sp_ft_bank[thetac],self.sz_spec_bank[Tc])
		mask=np.ones((gset.mmfset.npix,gset.mmfset.npix),float)
		psmask=np.ones((gset.mmfset.npix,gset.mmfset.npix),float)
		
		if mask_fdata:
			# This will mask the clusters in the field
			mask[(fdata/err)>=maskthr]=0.
			# This mask picks up the point source contribution to the filtered data
			if psmask==[]:
				psmask[(fdata/err)<=-maskthr]=0.
			# We find the std. dev. of the filtered data after removing the sources.
			err=np.std(fdata[mask*psmask==1.])
			
		yc=np.max((fdata*self.cmask).ravel())
		snr=yc/err
		norm=self.fn_yerr_norm(thetac)
		
		if write_data:
			hdu0=fits.PrimaryHDU()
			
			hdu1 = fits.ImageHDU()
			hdu1.header["EXTNAME"]="Result"
			hdu1.header["COMMENT"]="theta500, T, y_c err, SNR, y_c, Conv. to Y err"
			hdu1.data=[thetac,Tc,err,snr,yc,norm]
			
			hdu2 = fits.ImageHDU()
			hdu2.header["EXTNAME"]="Fdata"
			hdu2.data=fdata
			
			hdu=fits.HDUList([hdu0,hdu1,hdu2])
			hdu.writeto(filename,overwrite=True)

		return fdata,err,snr,yc

	def return_optimal_theta500(self,Tc,maskthr=3.,mask_fdata=True,write_data=False,filename=[]):
		ans=np.zeros((4,np.size(self.sp_ft_bank.keys())),float)
		#fdata=np.zeros((np.size(template_ft_bank.keys()),gset.mmfset.npix,gset.mmfset.npix),float)
		theta500=sorted(self.sp_ft_bank.keys())
		for idx,thetac in enumerate(theta500):
			template_ft=self.sp_ft_bank[thetac]
			fdata,ans[0,idx],ans[1,idx],ans[2,idx]=self.return_snr(thetac,Tc,mask_fdata=mask_fdata)
			ans[3,idx]=self.fn_yerr_norm(thetac)
		
		thetac=np.linspace(min(theta500),max(theta500),1000.)
		fn=interp1d(theta500,ans[1,:],kind="cubic") ; snr=fn(thetac) ; snr_max=max(snr)
		otheta500=thetac[np.where(snr==snr_max)[0][0]]
		fn=interp1d(theta500,ans[0,:],kind="cubic") ; err=fn(otheta500) ; norm=self.fn_yerr_norm(otheta500)
		fn=interp1d(theta500,ans[2,:],kind="cubic") ; yc=fn(otheta500)
		
		if write_data:
			hdu0=fits.PrimaryHDU()
			
			hdu1 = fits.ImageHDU()
			hdu1.header["EXTNAME"]="Result"
			hdu1.header["COMMENT"]="Opt_theta500, T, y_c err, SNR, y_c, Conv. to Y err"
			hdu1.data=[otheta500,Tc,err,snr_max,yc, norm]
			
			hdu2 = fits.ImageHDU()
			hdu2.header["EXTNAME"]="Theta500"
			hdu2.header["COMMENT"]="arcminutes"
			hdu2.data=theta500
			
			hdu3 = fits.ImageHDU()
			hdu3.header["EXTNAME"]="Raw"
			hdu3.header["COL-0"]="y_c err"
			hdu3.header["COL-1"]="SNR"
			hdu3.header["COL-2"]="y_c"
			hdu3.header["COL-3"]="Convert to Y err"
			hdu3.data=ans
			
			hdu=fits.HDUList([hdu0,hdu1,hdu2,hdu3])
			hdu.writeto(filename,overwrite=True)
			
		return err,snr_max,yc,otheta500,ans

	def return_optimal_T500(self,thetac,maskthr=3.,mask_fdata=True,write_data=False,filename=[]):
		ans=np.zeros((4,np.size(self.sz_spec_bank.keys())),float)
		norm=self.fn_yerr_norm(thetac)
		#fdata=np.zeros((np.size(sz_spec_bank.keys()),gset.mmfset.npix,gset.mmfset.npix),float)
		T500=sorted(self.sz_spec_bank.keys())
		for idx,Tc in enumerate(T500):
			fdata,ans[0,idx],ans[1,idx],ans[2,idx]=self.return_snr(thetac,Tc,mask_fdata=mask_fdata)
			ans[3,idx]=norm
		
		Tc=np.linspace(min(T500),max(T500),1000.)
		fn=interp1d(T500,ans[1,:],kind="cubic") ; snr=fn(Tc) ; snr_max=max(snr)
		oT500=Tc[np.where(snr==snr_max)[0][0]]
		fn=interp1d(T500,ans[0,:],kind="cubic") ; err=fn(oT500)
		fn=interp1d(T500,ans[2,:],kind="cubic") ; yc=fn(oT500)
		
		if write_data:
			hdu0=fits.PrimaryHDU()
			
			hdu1 = fits.ImageHDU()
			hdu1.header["EXTNAME"]="Result"
			hdu1.header["COMMENT"]="opt_theta500, opt_T500, y_c err, SNR, y_c, Conv. to Y err"
			hdu1.data=[thetac,oT500,err,snr_max,yc,norm]
			
			hdu2 = fits.ImageHDU()
			hdu2.header["EXTNAME"]="T500"
			hdu2.header["COMMENT"]="keV"
			hdu2.data=T500
			
			hdu3 = fits.ImageHDU()
			hdu3.header["EXTNAME"]="Raw"
			hdu3.header["COL-0"]="y_c err"
			hdu3.header["COL-1"]="SNR"
			hdu3.header["COL-2"]="y_c"
			hdu3.header["COL-3"]="Convert to Y err"
			hdu3.data=ans
			
			hdu=fits.HDUList([hdu0,hdu1,hdu2,hdu3])
			hdu.writeto(filename,overwrite=True)
		
		return err,snr_max,yc,oT500,ans

	def eval_mmf_theta500_T500(self,maskthr=3.,mask_fdata=True,write_data=False,filename=[]):
		ans=np.zeros((4,np.size(self.sp_ft_bank.keys()),np.size(self.sz_spec_bank.keys())),float)
		
		T500=sorted(self.sz_spec_bank.keys())
		theta500=sorted(self.sp_ft_bank.keys())
		norm=np.zeros(np.size(self.sp_ft_bank.keys()),float)
		norm=self.fn_yerr_norm(theta500)
		for idx,Tc in enumerate(T500):
			for jdx, thetac in enumerate(theta500):
				fdata,ans[0,jdx,idx],ans[1,jdx,idx],ans[2,jdx,idx]=self.return_snr(thetac,Tc,mask_fdata=mask_fdata)
				ans[3,jdx,idx]=norm[jdx]
	
		if write_data:
			hdu0=fits.PrimaryHDU()
			
			hdu1 = fits.ImageHDU()
			hdu1.header["EXTNAME"]="Theta500"
			hdu1.header["COMMENT"]="arcminutes"
			hdu1.data=theta500
			
			hdu2 = fits.ImageHDU()
			hdu2.header["EXTNAME"]="T500"
			hdu2.header["COMMENT"]="keV"
			hdu2.data=T500
			
			hdu3 = fits.ImageHDU()
			hdu3.header["EXTNAME"]="Raw"
			hdu3.header["COL-0"]="y_c err"
			hdu3.header["COL-1"]="SNR"
			hdu3.header["COL-2"]="y_c"
			hdu3.header["COL-3"]="Convert to Y err"
			hdu3.data=ans
			
			hdu=fits.HDUList([hdu0,hdu1,hdu2,hdu3])
			hdu.writeto(filename,overwrite=True)
		
		return theta500,T500,ans


	def eval_mmf_theta500_T500_constrained(self,redshift,maskthr=3.,mask_fdata=True,write_data=False,filename=[]):
		theta500=np.array(sorted(self.sp_ft_bank.keys()))
		T500=cosmo_fn.convert_theta500_T500(theta500,redshift)
		T500_max=max(np.array(self.sz_spec_bank.keys()))
		norm=self.fn_yerr_norm(theta500)
		ans=np.zeros((4,np.size(theta500)),float)
		
		for idx,thetac in enumerate(theta500):
			Tc=min(T500_max,T500[idx])
			T500[idx]=Tc
			fdata,ans[0,idx],ans[1,idx],ans[2,idx]=self.return_snr(thetac,np.round(Tc,0),mask_fdata=mask_fdata)
			ans[3,idx]=norm[idx]
			
		thetac=np.linspace(min(theta500),max(theta500),1000.)
		Tc=np.linspace(min(T500),max(T500),1000.)

		fn=interp1d(theta500,ans[1,:],kind="cubic")
		snr=fn(thetac) ; snr_max=max(snr)
		otheta500=thetac[np.where(snr==snr_max)[0][0]]
		oT500=Tc[np.where(snr==snr_max)[0][0]]
		fn=interp1d(theta500,ans[0,:],kind="cubic") ; err=fn(otheta500)
		fn=interp1d(theta500,ans[2,:],kind="cubic") ; yc=fn(otheta500)
		onorm=self.fn_yerr_norm(otheta500)

		if write_data:
			hdu0=fits.PrimaryHDU()
	
			hdu1= fits.ImageHDU()
			hdu1.header["EXTNAME"]="Result"
			hdu1.header["REDSHIFT"]=redshift
			hdu1.header["COMMENT"]="This is run on a constrained T500-theta500 grid"
			hdu1.header["COMMENT"]="opt_theta500, opt_T, y_c err, SNR, y_c, Conv. to Y err"
			hdu1.data=[otheta500,oT500,err,snr_max,yc,onorm]

			hdu2 = fits.ImageHDU()
			hdu2.header["EXTNAME"]="Theta500"
			hdu2.header["COMMENT"]="arcminutes"
			hdu2.data=theta500

			hdu3 = fits.ImageHDU()
			hdu3.header["EXTNAME"]="T500"
			hdu3.header["COMMENT"]="keV"
			hdu3.data=T500

			hdu4 = fits.ImageHDU()
			hdu4.header["EXTNAME"]="Raw"
			hdu4.header["COL-0"]="y_c err"
			hdu4.header["COL-1"]="SNR"
			hdu4.header["COL-2"]="y_c"
			hdu4.header["COL-3"]="Convert to Y err"
			hdu4.data=ans

			hdu=fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4])
			hdu.writeto(filename,overwrite=True)
		return err,snr_max,yc,oT500,ans
