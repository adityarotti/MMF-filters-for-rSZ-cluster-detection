##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  12 September 2020		     				 		                             #
# Date modified: 12 September 2020																 #
##################################################################################################

import os
import numpy as np
import healpy as h
from astropy.io import fits

from flat_sky_codes import tangent_plane_analysis as tpa
from flat_sky_codes import flat_sky_analysis as fsa
from masking import gen_masks as gm

from modules.settings import global_mmf_settings as gset
	
def return_zone_mask(nside=[]):
	zone_mask=h.read_map(gset.mmfset.gal_mask_name,verbose=False)
	mask=(2.-zone_mask)/2.
	
	# Masking the Large and Small Magellanic clouds.
	lmc_glon=280.4652
	lmc_glat=-32.8884
	lmc_pix=h.ang2pix(gset.mmfset.nside,lmc_glon,lmc_glat,lonlat=True)
	lmc_vec=h.pix2vec(gset.mmfset.nside,lmc_pix)
	lmc_pix=h.query_disc(gset.mmfset.nside,lmc_vec,radius=(2*60./60.)*np.pi/180.)
	mask[lmc_pix]=0.

	smc_glon=302.8084
	smc_glat=-44.3277
	smc_pix=h.ang2pix(gset.mmfset.nside,smc_glon,smc_glat,lonlat=True)
	smc_vec=h.pix2vec(gset.mmfset.nside,smc_pix)
	smc_pix=h.query_disc(gset.mmfset.nside,smc_vec,radius=(1.5*60./60.)*np.pi/180.)
	mask[smc_pix]=0.
	
	if nside!=[]:
		mask=h.ud_grade(mask,nside)
	
	return mask
	
		
def return_tile_definition(tilenside=8,fsky_map=[],fsky_thr=0.):
	if fsky_map==[]:
		pixnum=np.arange(h.nside2npix(tilenside))
		fsky_map=np.ones(h.nside2npix(tilenside),dtype=np.float64)
	else:
		pixnum=np.arange(len(fsky_map))
		tilenside=h.get_nside(fsky_map)

	pix_glon,pix_glat=h.pix2ang(tilenside,pixnum,lonlat=True)
	temp_pixnum=pixnum[fsky_map>=fsky_thr]
#	temp_pixnum=temp_pixnum[fsky_map[temp_pixnum]<fsky_thr]

	tiledef={}
	for px in temp_pixnum[:10]:
		tiledef[px]={}
		tiledef[px]["FSKY"]=fsky_map[px]
		tiledef[px]["GLON"]=pix_glon[px]
		tiledef[px]["GLAT"]=pix_glat[px]
		tiledef[px]["TILENAME"]="tile_G" + str(round(pix_glon[px],2)) + ["", "+"][pix_glat[px] >= 0] + str(round(pix_glat[px],2))
		tiledef[px]["FILENAME"]=gset.mmfset.paths["tplanes"] + tiledef[px]["TILENAME"] + ".fits"
		tiledef[px]["CATNAME"]=gset.mmfset.paths["result_data"] + tiledef[px]["TILENAME"] + ".dict"
	return tiledef

def return_sky_tile_map(tilenside=8,dummy_nside=512,rescale=0.5,edge_width=60.,fwhm=60.):
	pixnum=np.arange(h.nside2npix(tilenside))
	pix_glon,pix_glat=h.pix2ang(tilenside,pixnum,lonlat=True)
	gal_mask=return_zone_mask(nside=dummy_nside)
	
	projop=tpa.tangent_plane_setup(dummy_nside,gset.mmfset.xsize,0.,0.,rescale=rescale)
	x,y=np.indices((projop.npix,projop.npix))
	apo_mask=return_edge_apodized_mask(projop.npix,projop.pixel_size,edge_width=edge_width,fwhm=fwhm)
	bin_mask=np.ones_like(apo_mask)
	bin_mask[apo_mask<=0.99]=0.
	
	temp_map=np.zeros(h.nside2npix(dummy_nside),np.float64)
	tile_map=np.zeros(h.nside2npix(dummy_nside),np.float64)
	fsky_map=np.zeros(np.size(pixnum),np.float64)
	
	for idx in pixnum:
		projop=tpa.tangent_plane_setup(dummy_nside,gset.mmfset.xsize,pix_glat[idx],pix_glon[idx],rescale=0.5)
		glon,glat=projop.ij2ang(x.ravel(),y.ravel())
		mapixs=h.ang2pix(dummy_nside,glon,glat,lonlat=True)
		tmask=projop.get_tangent_plane(gal_mask)
		fsky_map[idx]=np.sum(tmask)/np.size(tmask)
		temp_map[:]=0.
		temp_map[mapixs]=(bin_mask*tmask).ravel()
		tile_map=tile_map+temp_map
	return tile_map,fsky_map,apo_mask

def return_edge_apodized_mask(npix,reso,edge_width=60.,fwhm=60.):
	mask=np.ones((npix,npix),np.float64)
	epix=np.int(np.ceil(edge_width/reso))
	mask[:epix,:]=0 ; mask[npix-epix:,:]=0
	mask[:,:epix]=0 ; mask[:,npix-epix:]=0
	ell,bl=fsa.get_gauss_beam(fwhm,20000)
	mask=fsa.filter_map(mask,reso,bl,ell)
	return mask




