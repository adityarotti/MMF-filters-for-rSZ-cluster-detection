import numpy as np
import healpy as h
from scipy.interpolate import interp1d
from astropy.io import fits

from spectral_template import sz_spec as szsed
from spatial_template import sim_cluster as sc
from spatial_template import sz_pressure_profile as szp
from settings import mmf_settings as mmfset
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

	clusters=np.zeros(h.nside2npix(mmfset.nside),float)

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
		cpix=h.vec2pix(mmfset.nside,vec[0],vec[1],vec[2])
		theta0,phi0=h.pix2ang(mmfset.nside,cpix)
		spix=h.query_disc(mmfset.nside,vec,(theta500*cutoff/60.)*np.pi/180.,inclusive=True,fact=2)
		theta1,phi1=h.pix2ang(mmfset.nside,spix)
		cosbeta=np.sin(theta0)*np.sin(theta1)*np.cos(phi1-phi0)+np.cos(theta0)*np.cos(theta1)
		beta=np.arccos(cosbeta)*180./np.pi*60
		clusters[cpix]=fn_yprofile(0.)*inj_y
		
		for i,pix in enumerate(spix):
		    clusters[pix]=fn_yprofile(beta[i])*inj_y

	filename=mmfset.paths["clusters"] + "clusters.fits"
	h.write_map(filename,clusters,overwrite=True)
		
	if write_alm:
	    filename=mmfset.paths["clusters"] + "clusters_alm.fits"
	    cluster_alm=h.map2alm(clusters,lmax=3*mmfset.nside)
	    h.write_alm(filename,cluster_alm,overwrite=True)

def gen_multi_freq_cluster_map(T=0.):
	plbp_sz_spec=szsed.return_planck_bp_sz_spec(T=T)
	
	filename=mmfset.paths["clusters"] + "clusters_alm.fits"
	cluster_alm=h.read_alm(filename)
	
	for ch in mmfset.channels:
		fwhm=(mmfset.fwhm[ch]/60.)*np.pi/180.
		bl=h.gauss_beam(fwhm,lmax=3*mmfset.nside)
		if mmfset.pwc:
			pwc=h.pixwin(mmfset.nside)[:np.size(bl)]
			bl=bl*pwc
		almp=h.almxfl(cluster_alm,bl)*plbp_sz_spec[ch]

		cluster=h.alm2map(almp,mmfset.nside,verbose=False)

		filename=mmfset.paths["clusters"] + "cluster_" + str(int(ch)) + "GHz.fits"
		h.write_map(filename,cluster,overwrite=True)
		print ch

def sim_multi_frequency_cmb_map():
	cl=h.read_cl(mmfset.cmb_spectra)[0]
	ell=np.arange(np.size(cl),dtype="float")
	cmbalm=h.synalm(cl,lmax=3*mmfset.nside)

	for ch in mmfset.channels:
		fwhm=(mmfset.fwhm[ch]/60.)*np.pi/180.
		bl=h.gauss_beam(fwhm,lmax=3*mmfset.nside)
		if mmfset.pwc:
			pwc=h.pixwin(mmfset.nside)[:np.size(bl)]
			bl=bl*pwc
		almp=h.almxfl(cmbalm,bl)
		cmb=h.alm2map(almp,mmfset.nside,verbose=False)*1e-6
		filename=mmfset.paths["cmb"] + "cmb_" + str(int(ch)) + "GHz.fits"
		h.write_map(filename,cmb,overwrite=True)
		print ch

def extract_tangent_planes(latlon=[],rescale_y=1.):
	pc=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,60.,-4.,rescale=1.)
	if latlon==[]:
		p1=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,60.,51.,rescale=1.)
	else:
		p1=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,latlon[0],latlon[1],rescale=1.)

	cmb=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),float)
	noise=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),float)
	cmbfrg=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),float)
	injclus=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),float)
	for i,ch in enumerate(mmfset.channels):
		# These maps are in uK_RJ units and we are converting them to K_CMB.
		filename=mmfset.paths["psm_sims"] + "group2_map_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)*mmfset.conv_KRJ_KCMB[ch]*1e-6
		cmbfrg[i,:,:]=p1.get_tangent_plane(tempmap)

		# These maps are in uK_RJ units and we are converting them to K_CMB.
		filename=mmfset.paths["psm_sims"] + "group8_map_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)*mmfset.conv_KRJ_KCMB[ch]*1e-6
		noise[i,:,:]=p1.get_tangent_plane(tempmap)

		
		filename=mmfset.paths["clusters"] + "cluster_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		injclus[i,:,:]=pc.get_tangent_plane(tempmap)*rescale_y

		#filename=mmfset.paths["cmb"] + "cmb_" + str(int(ch)) + "GHz.fits"
		filename=mmfset.paths["psm_sims"] + "group3_map_" + str(int(ch)) + "GHz.fits"
		tempmap=h.read_map(filename,verbose=False)*mmfset.conv_KRJ_KCMB[ch]*1e-6
		cmb[i,:,:]=p1.get_tangent_plane(tempmap)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB"
	hdu.data=injclus + cmb + noise/1000.
	filename=mmfset.paths["tplanes"] + "sz_cmb.fits"
	hdu.writeto(filename,overwrite=True)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB + NOISE"
	hdu.data=injclus + cmb + noise
	filename=mmfset.paths["tplanes"] + "sz_cmb_noise.fits"
	hdu.writeto(filename,overwrite=True)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB + FRG"
	hdu.data=injclus + cmbfrg + noise/1000.
	filename=mmfset.paths["tplanes"] + "sz_cmb_frg.fits"
	hdu.writeto(filename,overwrite=True)

	hdu = fits.ImageHDU()
	hdu.header["EXTNAME"]="SZ + CMB + FRG + NOISE"
	hdu.data=injclus + cmbfrg + noise
	filename=mmfset.paths["tplanes"] + "sz_cmb_frg_noise.fits"
	hdu.writeto(filename,overwrite=True)
