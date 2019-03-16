##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
import healpy as h
from scipy.interpolate import interp1d
from astropy.io import fits

from modules.settings import global_mmf_settings as gset
from spectral_template import sz_spec as szsed
from spatial_template import sim_cluster as sc
from spatial_template import sz_pressure_profile as szp
from flat_sky_codes import flat_sky_analysis as fsa
from flat_sky_codes import tangent_plane_analysis as tpa

def sim_cluster(clus_prop={},write_alm=True,cutoff=10.,profile="GNFW"):
	if clus_prop=={}:
		clus_prop={}
		clus_prop[0]={}
		clus_prop[0]["injy"]=0.0002 ; clus_prop[0]["theta500"]=8. # arcminutes
		clus_prop[0]["lon"]=-4. ; clus_prop[0]["lat"]=60.

		clus_prop[2]={}
		clus_prop[2]["injy"]=0.0004 ; clus_prop[2]["theta500"]=12. # arcminutes
		clus_prop[2]["lon"]=4. ; clus_prop[2]["lat"]=62.

	clusters=np.zeros(h.nside2npix(gset.mmfset.nside),float)

	for k in clus_prop.keys():
		#print clus_prop[k].keys()
		inj_y=clus_prop[k]['injy']
		theta500=clus_prop[k]['theta500']
		lon=clus_prop[k]['lon']
		lat=clus_prop[k]['lat']
		
		rhop=np.linspace(0.001*theta500,1.2*cutoff*theta500,500.)
		yprofile=np.zeros(np.size(rhop),float)
		if profile=="GNFW":
			yprofile=szp.gnfw_2D_pressure_profile(rhop,theta500)
		elif profile=="beta":
			yprofile=szp.analytical_beta_2D_profile_profile(rhop,theta500)
		fn_yprofile=interp1d(rhop,yprofile,kind="cubic",bounds_error=False,fill_value=(yprofile[0],yprofile[-1]))

		vec=h.ang2vec(lon,lat,lonlat=True)
		cpix=h.vec2pix(gset.mmfset.nside,vec[0],vec[1],vec[2])
		theta0,phi0=h.pix2ang(gset.mmfset.nside,cpix)
		spix=h.query_disc(gset.mmfset.nside,vec,(theta500*cutoff/60.)*np.pi/180.,inclusive=True,fact=2)
		theta1,phi1=h.pix2ang(gset.mmfset.nside,spix)
		cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
		beta=np.arccos(cosbeta)*180./np.pi*60
		clusters[cpix]=fn_yprofile(0.)*inj_y
		
		for i,pix in enumerate(spix):
		    clusters[pix]=fn_yprofile(beta[i])*inj_y

	filename=gset.mmfset.paths["clusters"] + "clusters.fits"
	h.write_map(filename,clusters,overwrite=True)
		
	if write_alm:
	    filename=gset.mmfset.paths["clusters"] + "clusters_alm.fits"
	    cluster_alm=h.map2alm(clusters,lmax=3*gset.mmfset.nside)
	    h.write_alm(filename,cluster_alm,overwrite=True)

def gen_multi_freq_cluster_map(T=0.):
	plbp_sz_spec=szsed.return_planck_bp_sz_spec(T=T)
	
	filename=gset.mmfset.paths["clusters"] + "clusters_alm.fits"
	cluster_alm=h.read_alm(filename)
	
	for ch in gset.mmfset.channels:
		fwhm=(gset.mmfset.fwhm[ch]/60.)*np.pi/180.
		bl=h.gauss_beam(fwhm,lmax=3*gset.mmfset.nside)
		if gset.mmfset.pwc:
			pwc=h.pixwin(gset.mmfset.nside)[:np.size(bl)]
			bl=bl*pwc
		almp=h.almxfl(cluster_alm,bl)*plbp_sz_spec[ch]

		cluster=h.alm2map(almp,gset.mmfset.nside,verbose=False)

		filename=gset.mmfset.paths["clusters"] + "cluster_" + str(int(ch)) + "GHz.fits"
		h.write_map(filename,cluster,overwrite=True)
		print ch

def sim_multi_frequency_cmb_map():
	cl=h.read_cl(gset.mmfset.cmb_spectra)[0]
	ell=np.arange(np.size(cl),dtype="float")
	cmbalm=h.synalm(cl,lmax=3*gset.mmfset.nside)

	for ch in gset.mmfset.channels:
		fwhm=(gset.mmfset.fwhm[ch]/60.)*np.pi/180.
		bl=h.gauss_beam(fwhm,lmax=3*gset.mmfset.nside)
		if gset.mmfset.pwc:
			pwc=h.pixwin(gset.mmfset.nside)[:np.size(bl)]
			bl=bl*pwc
		almp=h.almxfl(cmbalm,bl)
		cmb=h.alm2map(almp,gset.mmfset.nside,verbose=False)*1e-6
		filename=gset.mmfset.paths["cmb"] + "cmb_" + str(int(ch)) + "GHz.fits"
		h.write_map(filename,cmb,overwrite=True)
		print ch

def extract_tangent_planes(latlon=[],rescale_y=1.):
	pc=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,60.,-4.,rescale=1.)
	if latlon==[]:
		p1=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,60.,51.,rescale=1.)
	else:
		p1=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,latlon[0],latlon[1],rescale=1.)

	cmb=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)
	noise=np.zeros((np.size(gset.mmfset.channels),mmfset.npix,gset.mmfset.npix),float)
	cmbfrg=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)
	injclus=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)
	for i,ch in enumerate(gset.mmfset.channels):
		# These maps are in uK_RJ units and we are converting them to K_CMB.
		filename=gset.mmfset.paths["psm_sims"] + "group2_map_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)*gset.mmfset.conv_KRJ_KCMB[ch]*1e-6
		cmbfrg[i,:,:]=p1.get_tangent_plane(tempmap)

		# These maps are in uK_RJ units and we are converting them to K_CMB.
		filename=gset.mmfset.paths["psm_sims"] + "group8_map_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)*gset.mmfset.conv_KRJ_KCMB[ch]*1e-6
		noise[i,:,:]=p1.get_tangent_plane(tempmap)

		
		filename=gset.mmfset.paths["clusters"] + "cluster_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		injclus[i,:,:]=pc.get_tangent_plane(tempmap)*rescale_y

		#filename=gset.mmfset.paths["cmb"] + "cmb_" + str(int(ch)) + "GHz.fits"
		filename=gset.mmfset.paths["psm_sims"] + "group3_map_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)*gset.mmfset.conv_KRJ_KCMB[ch]*1e-6
		cmb[i,:,:]=p1.get_tangent_plane(tempmap)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB"
	hdu.data=injclus + cmb + noise/1000.
	filename=gset.mmfset.paths["tplanes"] + "sz_cmb.fits"
	hdu.writeto(filename,overwrite=True)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB + NOISE"
	hdu.data=injclus + cmb + noise
	filename=gset.mmfset.paths["tplanes"] + "sz_cmb_noise.fits"
	hdu.writeto(filename,overwrite=True)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB + FRG"
	hdu.data=injclus + cmbfrg + noise/1000.
	filename=gset.mmfset.paths["tplanes"] + "sz_cmb_frg.fits"
	hdu.writeto(filename,overwrite=True)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB + FRG + NOISE"
	hdu.data=injclus + cmbfrg + noise
	filename=gset.mmfset.paths["tplanes"] + "sz_cmb_frg_noise.fits"
	hdu.writeto(filename,overwrite=True)
