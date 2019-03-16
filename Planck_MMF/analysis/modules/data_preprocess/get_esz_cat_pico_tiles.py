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
import unit_conv as uc


def extract_tangent_planes(dryrun=False,verbose=False):
	
	xsz_cat=get_tangent_plane_fnames()
	
	if dryrun:
		for mtype in gset.mmfset.map_fnames.keys():
			for ich, ch in enumerate(gset.mmfset.all_channels):
				#print gset.mmfset.map_fnames[mtype][ch]
				print ch,uc.conv_uKTRJ_KTBB(ch)
				for idx,fname_suffix in enumerate(xsz_cat["FILENAME"]):
					filename=gset.mmfset.paths["tplanes"] + mtype + fname_suffix
					print filename
	else:
		for mtype in gset.mmfset.map_fnames.keys():
			for ich, ch in enumerate(gset.mmfset.all_channels):
				#print gset.mmfset.map_fnames[mtype][ch]
				chmap=h.read_map(gset.mmfset.map_fnames[mtype][ch],0,verbose=False)*uc.conv_uKTRJ_KTBB(ch)
				for idx,fname_suffix in enumerate(xsz_cat["FILENAME"]):
					filename=gset.mmfset.paths["tplanes"] + mtype + fname_suffix
					glon=xsz_cat["GLON"][idx]
					glat=xsz_cat["GLAT"][idx]
					projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
					timage=projop.get_tangent_plane(chmap)

					if os.path.isfile(filename):
						f=fits.open(filename)
						f[2].data[ich,:,:]=timage
						fits.update(filename,f[2].data,f[2].header,"Data tangent plane")
					else:
						hdu0=fits.PrimaryHDU()
						hdu_ch = fits.ImageHDU()
						hdu_ch.header["EXTNAME"]="Channels"
						hdu_ch.data=gset.mmfset.channels
						hdu_map = fits.ImageHDU()
						hdu_map.header["EXTNAME"]="Data tangent plane"
						hdu_map.header["XYsize"]=str(gset.mmfset.xsize) + " degrees"
						hdu_map.header["Reso"]=str(gset.mmfset.reso) + " arcminutes"
						hdu_map.header["GLON"]=str(round(glon,4)) + " degrees"
						hdu_map.header["GLAT"]=str(round(glat,4)) + " degrees"
						null_data=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)
						null_data[ich,:,:]=timage
						hdu_map.data=null_data
						hdu=fits.HDUList([hdu0,hdu_ch,hdu_map])
						hdu.writeto(filename)


def get_tangent_plane_fnames():
    xsz_cat=get_esz_catalogue()

    tfname=[None]*np.size(xsz_cat["z"])
    for idx,clstrname in enumerate(xsz_cat["z"]):
    	glon=xsz_cat["GLON"][idx]
    	glat=xsz_cat["GLAT"][idx]
        filename="_G" + str(round(glon,2)) + ["", "+"][glat >= 0] + str(round(glat,2)) + ".fits"
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

