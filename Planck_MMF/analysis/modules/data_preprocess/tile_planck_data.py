##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January 2019     				 		                             #
# Date modified: 13 September 2020								 								     #
##################################################################################################

import os
import numpy as np
import healpy as h
from astropy.io import fits

from flat_sky_codes import tangent_plane_analysis as tpa
from modules.settings import global_mmf_settings as gset
from data_preprocess import tiling_the_sphere as tts
from masking import gen_masks as gm
import data_inpaint_ps as paint


def extract_data_tiles(tiledef,do_data=True,do_mask=True,gen_mask=True,verbose=False):
	
	gal_mask=tts.return_zone_mask()
	
	if do_data:
		for ich,ch in enumerate(gset.mmfset.all_channels):
			if verbose:
				print "Working on extraction tangent planes from " + str(ch) + " GHz maps"
			chmap=h.read_map(gset.mmfset.map_fnames[ch],0,verbose=False)/gset.mmfset.conv_KCMB2MJY[ch]
			
			for pix in tiledef.keys():
				filename=tiledef[pix]["FILENAME"]
				glat=tiledef[pix]["GLAT"]
				glon=tiledef[pix]["GLON"]
				projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
				timage=projop.get_tangent_plane(chmap)
				tmask=projop.get_tangent_plane(gal_mask)
				tmask=gm.return_apodized_mask(tmask,apowidth=40.)

				if os.path.isfile(filename):
					f=fits.open(filename)
					extname="Data tangent plane"
					f[extname].data[ich,:,:]=timage
					fits.update(filename,f[extname].data,f[extname].header,extname)
					
					extname="Galactic mask"
					f[extname].data=tmask
					fits.update(filename,f[extname].data,f[extname].header,extname)
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
					hdu_map.header["GLON"]=str(round(glon,10)) + " degrees"
					hdu_map.header["GLAT"]=str(round(glat,10)) + " degrees"
					hdu_gmask = fits.ImageHDU()
					hdu_gmask.header["EXTNAME"]="Galactic mask"
					hdu_gmask.header["XYsize"]=str(gset.mmfset.xsize) + " degrees"
					hdu_gmask.header["Reso"]=str(gset.mmfset.reso) + " arcminutes"
					hdu_gmask.header["GLON"]=str(round(glon,10)) + " degrees"
					hdu_gmask.header["GLAT"]=str(round(glat,10)) + " degrees"
					null_data=np.zeros((np.size(gset.mmfset.all_channels),gset.mmfset.npix,gset.mmfset.npix),float)
					null_data[ich,:,:]=timage
					hdu_map.data=null_data
					
					hdu_gmask.data=tmask
					
					hdu=fits.HDUList([hdu0,hdu_ch,hdu_map,hdu_gmask])
					hdu.writeto(filename)
					
	if do_mask:
		# Point source mask
		if verbose:
			print "Appending point source masks"
		chmap=gen_ps_mask(ps_cutoff=3.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Point Source Mask"
		for pix in tiledef.keys():
			filename=tiledef[pix]["FILENAME"]
			glat=tiledef[pix]["GLAT"]
			glon=tiledef[pix]["GLON"]
			projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			hdu_mask.data=timage
			f=fits.open(filename)
			extnames=[f[i+1].header["EXTNAME"] for i in range(len(f)-1)]
			if "Point Source Mask" not in extnames:
				fits.append(filename,hdu_mask.data,hdu_mask.header)
			else:
				fits.update(filename,hdu_mask.data,hdu_mask.header,"Point Source Mask")
			f.close()

		# Extended point source mask
		if verbose:
			print "Appending extended point source masks"
		chmap=gen_ps_mask(ps_cutoff=5.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Extended point Source Mask"
		for pix in tiledef.keys():
			filename=tiledef[pix]["FILENAME"]
			glat=tiledef[pix]["GLAT"]
			glon=tiledef[pix]["GLON"]
			projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			hdu_mask.data=timage
			f=fits.open(filename)
			extnames=[f[i+1].header["EXTNAME"] for i in range(len(f)-1)]
			if "Extended point Source Mask" not in extnames:
				fits.append(filename,hdu_mask.data,hdu_mask.header)
			else:
				fits.update(filename,hdu_mask.data,hdu_mask.header,"Extended point Source Mask")
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

def get_tangent_plane_fnames(tilenside=8,fsky_map=[],fsky_thr=0.):
    tiledef=tts.return_tile_definition(tilenside=tilenside,fsky_map=fsky_map,fsky_thr=fsky_thr)
    return tiledef

def gen_ps_inpainted_data(pix,tiledef):
	filename=tiledef[pix]["FILENAME"]
	f=fits.open(filename)
	extnames=[f[i+1].header["EXTNAME"] for i in range(len(f)-1)]
	
	data_extname="Data tangent plane"
	mask_extname="Point Source Mask"
	work_extname="PS inpainted data tangent plane"
	if work_extname not in extnames:
		hdu = fits.ImageHDU()
		hdu.header["EXTNAME"]=work_extname
		hdu.data=np.zeros_like(f[data_extname].data)
		if (1-np.sum(f[mask_extname].data)/np.size(f[mask_extname].data))>1.e-4:
			for i in range(f[data_extname].data.shape[0]):
				hdu.data[i,]=paint.return_ps_filled_data(f[data_extname].data[i,],f[mask_extname].data,pixel_size=gset.mmfset.reso,diffthr=1e-3,itermax=20)
		else:
			hdu.data=f[data_extname].data
		fits.append(filename,hdu.data,hdu.header)
	else:
		if (1-np.sum(f[mask_extname].data)/np.size(f[mask_extname].data))>1.e-4:
			for i in range(f[data_extname].data.shape[0]):
				f[work_extname].data[i,]=paint.return_ps_filled_data(f[data_extname].data[i,],f[mask_extname].data,pixel_size=gset.mmfset.reso,diffthr=1e-3,itermax=20)
		else:
			f[work_extname].data=f[data_extname].data
		fits.update(filename,f[work_extname].data,f[work_extname].header,"PS inpainted data tangent plane")
	f.close()
