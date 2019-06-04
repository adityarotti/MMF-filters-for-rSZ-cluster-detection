##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import os
import numpy as np
import healpy as h
from astropy.io import fits
from astropy.coordinates import SkyCoord

from flat_sky_codes import tangent_plane_analysis as tpa
from modules.settings import global_mmf_settings as gset
from settings import constants as cnst
from cosmology import cosmo_fn
from astropy.io import fits
import data_inpaint_ps as paint

def extract_tangent_planes(gen_mask=True,verbose=False,do_data=True,do_mask=True):
	
	xsz_cat=get_tangent_plane_fnames()

	if do_data:
		if verbose:
			print "Catalogue size: ", np.size(xsz_cat["z"])
		tfname=[None]*np.size(xsz_cat["z"])

		for ich,ch in enumerate(gset.mmfset.all_channels):
			if verbose:
				print "Working on extraction tangent planes from " + str(ch) + " GHz maps"
			chmap=h.read_map(gset.mmfset.map_fnames[ch],0,verbose=False)/gset.mmfset.conv_KCMB2MJY[ch]
			for idx,filename in enumerate(xsz_cat["FILENAME"]):
				glon=xsz_cat["GLON"][idx]
				glat=xsz_cat["GLAT"][idx]
				projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
				timage=projop.get_tangent_plane(chmap)

				if os.path.isfile(filename):
					f=fits.open(filename)
					f[2].data[ich,:,:]=timage
					fits.update(filename,f[2].data,f[2].header,"Data tangent plane")
					f.close()
				else:
					hdu0=fits.PrimaryHDU()
					hdu_ch = fits.ImageHDU()
					hdu_ch.header["EXTNAME"]="Channels"
					hdu_ch.data=gset.mmfset.all_channels
					hdu_map = fits.ImageHDU()
					hdu_map.header["EXTNAME"]="Data tangent plane"
					hdu_map.header["XYsize"]=str(gset.mmfset.xsize) + " degrees"
					hdu_map.header["Reso"]=str(gset.mmfset.reso) + " arcminutes"
					hdu_map.header["GLON"]=str(round(glon,4)) + " degrees"
					hdu_map.header["GLAT"]=str(round(glat,4)) + " degrees"
					null_data=np.zeros((np.size(gset.mmfset.all_channels),gset.mmfset.npix,gset.mmfset.npix),float)
					null_data[ich,:,:]=timage
					hdu_map.data=null_data
					hdu=fits.HDUList([hdu0,hdu_ch,hdu_map])
					hdu.writeto(filename)
				if ich==0:
					tfname[idx]=filename

	if do_mask:
		# Point source mask
		if verbose:
			print "Appending point source masks"
		chmap=gen_ps_mask(ps_cutoff=3.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Point Source Mask"
		for idx,filename in enumerate(xsz_cat["FILENAME"]):
			glon=xsz_cat["GLON"][idx]
			glat=xsz_cat["GLAT"][idx]
			projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			f=fits.open(filename)
			if len(f)>3:
				f[3].data=timage
				fits.update(filename,f[3].data,f[3].header,"Point Source Mask")
			else:
				hdu_mask.data=timage
				fits.append(filename,hdu_mask.data,hdu_mask.header)
			f.close()
		
		# Extended point source mask
		if verbose:
			print "Appending extended point source masks"
		chmap=gen_ps_mask(ps_cutoff=5.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Extended point Source Mask"
		for idx,filename in enumerate(xsz_cat["FILENAME"]):
			glon=xsz_cat["GLON"][idx]
			glat=xsz_cat["GLAT"][idx]
			projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			
			f=fits.open(filename)
			if len(f)>4:
				f[4].data=timage
				fits.update(filename,f[4].data,f[4].header,"Extended point Source Mask")
			else:
				hdu_mask.data=timage
				fits.append(filename,hdu_mask.data,hdu_mask.header)
			f.close()

def gen_ps_mask(snrthr=10.,ps_cutoff=3.,verbose=False,gen_mask=True):
	filename=gset.mmfset.paths["planck_masks"] + "mmf3_ps_snr" + str(int(ps_cutoff)) + "_mask.fits"
	if (os.path.isfile(filename) and not(gen_mask)):
		mask=h.read_map(filename,verbose=verbose)
	else:
		mask=np.ones(h.nside2npix(gset.mmfset.nside),float)
		#chmask=np.ones((np.size(gset.mmfset.channels),h.nside2npix(gset.mmfset.nside)),float)
		chmask=np.ones(h.nside2npix(gset.mmfset.nside),float)
		for ich,ch in enumerate(gset.mmfset.all_channels):
			chmask[:]=1.
			f=fits.open(gset.mmfset.ps_cat_fname[ch])
			radius=gset.mmfset.ps_mask_weights[ch]*(ps_cutoff/np.sqrt(8.*np.log(2.)))*(gset.mmfset.fwhm[ch]/60.)*np.pi/180.
			#print ch,radius

			detflux=f[f[1].header["EXTNAME"]].data.field("DETFLUX")
			err=f[f[1].header["EXTNAME"]].data.field("DETFLUX_ERR")
			snr_mask=[(detflux/err) >= snrthr]

			glon=detflux=f[f[1].header["EXTNAME"]].data.field("GLON")[snr_mask]
			glat=detflux=f[f[1].header["EXTNAME"]].data.field("GLAT")[snr_mask]
			pixcs=h.ang2pix(gset.mmfset.nside,glon,glat,lonlat=True)

			if verbose:
				print ch,np.size(pixcs)

			for pix in pixcs:
				vec=h.pix2vec(gset.mmfset.nside,pix)
				disc_pix=h.query_disc(gset.mmfset.nside,vec,radius=radius,fact=4,inclusive=True)
				chmask[disc_pix]=0.0
			mask=mask*chmask

		h.write_map(filename,mask,overwrite=True)

	return mask

def get_tangent_plane_fnames():
    xsz_cat=get_esz_catalogue()

    tfname=[None]*np.size(xsz_cat["z"])
    for idx,clstrname in enumerate(xsz_cat["z"]):
    	glon=xsz_cat["GLON"][idx]
    	glat=xsz_cat["GLAT"][idx]
        filename=gset.mmfset.paths["tplanes"] + "cluster_G" + str(round(glon,2)) + ["", "+"][glat >= 0] + str(round(glat,2)) + ".fits"
        tfname[idx]=filename

	xsz_cat["FILENAME"]=tfname
    return xsz_cat

def get_esz_catalogue():
	dtype=["T500","T500_err","z","Mg500","Mg500_err","M500","M500_err","RA","DEC","R500","YSZ_500","YSZ_500_err","YX_500","YX_500_err"]
	cat=np.loadtxt(gset.mmfset.esz_cat_2011_file)
	xsz_cat={}
	for idx,d in enumerate(dtype):
		xsz_cat[d]=cat[:,idx]
	
	c = SkyCoord(xsz_cat["RA"], xsz_cat["DEC"] , frame='icrs', unit='deg')
	xsz_cat["GLON"]=c.galactic.l.degree
	xsz_cat["GLAT"]=c.galactic.b.degree

	# Get the angular size for clusters
	xsz_cat["theta500"]=(xsz_cat["R500"]/(cosmo_fn.dA(xsz_cat["z"])*1000.))*180.*60./np.pi

	# Normalizing the X-ray Compton y parameter
	# This returns Y_x in Mpc^2 units
	norm=(cnst.thomson_cc/cnst.e_rme)*(1./(cnst.mu_e*cnst.m_proton))*cnst.m_sun*1e14/cnst.mpc**2.
	xsz_cat["YX_500"]=xsz_cat["YX_500"]*norm
	xsz_cat["YX_500_err"]=xsz_cat["YX_500_err"]*norm
	
	xsz_cat["YSZ_500"]=xsz_cat["YSZ_500"]/1e4
	xsz_cat["YSZ_500_err"]=xsz_cat["YSZ_500_err"]/1e4
	
	return xsz_cat

def return_tangent_planes(glon,glat,gen_mask=True):
	projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
	data=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)
	for ich,ch in enumerate(gset.mmfset.channels):
		chmap=h.read_map(gset.mmfset.map_fnames[ch],0,verbose=False)/gset.mmfset.conv_KCMB2MJY[ch]
		data[ich,]=projop.get_tangent_plane(chmap)
	
	chmap=gen_ps_mask(ps_cutoff=3.,gen_mask=gen_mask)
	ps_mask=projop.get_tangent_plane(chmap)
	chmap=gen_ps_mask(ps_cutoff=5.,gen_mask=gen_mask)
	ext_ps_mask=projop.get_tangent_plane(chmap)
	return data,ps_mask,ext_ps_mask

# This is obsolete as the point source mask map provided
# on the legacy archive also mask some of the clusters.
def return_ps_mask(return_gal_mask=False,idx=0):
	mask=np.ones(h.nside2npix(gset.mmfset.nside),float)
	for i in range(6):
		mask=mask*h.read_map(gset.mmfset.ps_mask_name,i,verbose=False)
	return mask

def gen_ps_inpainted_data(idx):
	xsz_cat=get_tangent_plane_fnames()
	filename=xsz_cat["FILENAME"][idx]
	f=fits.open(filename)
	
	if len(f)==5:
		hdu = fits.ImageHDU()
		hdu.header["EXTNAME"]="PS inpainted data tangent plane"
		hdu.data=np.zeros_like(f[2].data)
		for i in range(f[2].data.shape[0]):
			hdu.data[i,]=paint.return_ps_filled_data(f[2].data[i,],f[3].data,pixel_size=gset.mmfset.reso,diffthr=1e-3,itermax=20)
		fits.append(filename,hdu.data,hdu.header)
	elif len(f)==6:
		for i in range(f[2].data.shape[0]):
			f[5].data[i,]=paint.return_ps_filled_data(f[2].data[i,],f[3].data,pixel_size=gset.mmfset.reso,diffthr=1e-3,itermax=20)
		fits.update(filename,f[5].data,f[5].header,"PS inpainted data tangent plane")
	f.close()
