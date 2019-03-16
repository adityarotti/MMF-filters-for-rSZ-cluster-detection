import os
import numpy as np
import healpy as h
from astropy.io import fits

from flat_sky_codes import tangent_plane_analysis as tpa
from modules.settings import global_mmf_settings as gset
from masking import gen_masks as gm

def extract_data_tiles(do_data=True,do_mask=True,gen_mask=True,verbose=False,glat_thr=0.):
	mask=gm.return_gal_mask()
	
	tilenside=8
	pixnum=np.arange(h.nside2npix(tilenside))
	fsky=np.zeros(np.size(pixnum),float)
	cglon,cglat=h.pix2ang(tilenside,pixnum,lonlat=True)
	
	for pix in pixnum:
		projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,cglat[pix],cglon[pix],rescale=1)
		tmask=projop.get_tangent_plane(mask)
		fsky[pix]=np.sum(tmask)/np.size(tmask)
		
	if do_data:
		for ich,ch in enumerate(gset.mmfset.all_channels):
			if verbose:
				print "Working on extraction tangent planes from " + str(ch) + " GHz maps"
			chmap=h.read_map(gset.mmfset.map_fnames[ch],0,verbose=False)/gset.mmfset.conv_KCMB2MJY[ch]
			
			for pix in pixnum[(fsky>0.3) & (abs(cglat)>glat_thr)]:
				filename=gset.mmfset.paths["tplanes"] + "tile_G" + str(round(cglon[pix],2)) + ["", "+"][cglat[pix] >= 0] + str(round(cglat[pix],2)) + ".fits"
				projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,cglat[pix],cglon[pix],rescale=1.)
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
					hdu_map.header["GLON"]=str(round(cglon[pix],10)) + " degrees"
					hdu_map.header["GLAT"]=str(round(cglat[pix],10)) + " degrees"
					null_data=np.zeros((np.size(gset.mmfset.all_channels),gset.mmfset.npix,gset.mmfset.npix),float)
					null_data[ich,:,:]=timage
					hdu_map.data=null_data
					hdu=fits.HDUList([hdu0,hdu_ch,hdu_map])
					hdu.writeto(filename)
					
	if do_mask:
		# Point source mask
		if verbose:
			print "Appending point source masks"
		chmap=gen_ps_mask(ps_cutoff=3.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Point Source Mask"
		for pix in pixnum[(fsky>0.3) & (abs(cglat)>glat_thr)]:
			filename=gset.mmfset.paths["tplanes"] + "tile_G" + str(round(cglon[pix],2)) + ["", "+"][cglat[pix] >= 0] + str(round(cglat[pix],2)) + ".fits"
			projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,cglat[pix],cglon[pix],rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			hdu_mask.data=timage
			fits.append(filename,hdu_mask.data,hdu_mask.header)
		
		# Extended point source mask
		if verbose:
			print "Appending extended point source masks"
		chmap=gen_ps_mask(ps_cutoff=5.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Extended point Source Mask"
		for pix in pixnum[(fsky>0.3) & (abs(cglat)>glat_thr)]:
			filename=gset.mmfset.paths["tplanes"] + "tile_G" + str(round(cglon[pix],2)) + ["", "+"][cglat[pix] >= 0] + str(round(cglat[pix],2)) + ".fits"
			projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,cglat[pix],cglon[pix],rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			hdu_mask.data=timage
			fits.append(filename,hdu_mask.data,hdu_mask.header)

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

#def get_data_tile_fnames(glat_thr=0.):
#    mmf3=get_mmf3_catalogue(snrthr=snrthr,cosmo_flag=cosmo_flag,zknown=True)
#
#    tfname=[None]*np.size(mmf3["SNR"])
#    for idx,clstrname in enumerate(mmf3["NAME"]):
#        filename=gset.mmfset.paths["tplanes"] + "cluster_" + clstrname[5:] + ".fits"
#        tfname[idx]=filename
#
#	mmf3["FILENAME"]=tfname
#    return mmf3
