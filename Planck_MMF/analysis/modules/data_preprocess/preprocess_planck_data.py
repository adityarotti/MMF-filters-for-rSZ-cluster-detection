import os
import numpy as np
import healpy as h
from astropy.io import fits

from flat_sky_codes import tangent_plane_analysis as tpa
from settings import mmf_settings as mmfset
from cosmology import cosmo_fn

def extract_tangent_planes(snrthr=6.,cosmo_flag=True,zknown=True,gen_mask=True,verbose=False,do_data=True,do_mask=True):
	mmf3=get_mmf3_catalogue(snrthr=snrthr,cosmo_flag=cosmo_flag,zknown=True)

	if do_data:
		if verbose:
			print "Catalogue size: ", np.size(mmf3["SNR"])
		tfname=[None]*np.size(mmf3["SNR"])

		for ich,ch in enumerate(mmfset.planck_channels):
			if verbose:
				print "Working on extraction tangent planes from " + str(ch) + " GHz maps"
			chmap=h.read_map(mmfset.map_fnames[ch],0,verbose=False)/mmfset.conv_KCMB2MJY[ch]
			for idx,clstrname in enumerate(mmf3["NAME"]):
				glon=mmf3["GLON"][idx]
				glat=mmf3["GLAT"][idx]
				snr=mmf3["SNR"][idx]
				filename=mmfset.paths["tplanes"] + "cluster_" + clstrname[5:] + ".fits"
				projop=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,glat,glon,rescale=1.)
				timage=projop.get_tangent_plane(chmap)

				if os.path.isfile(filename):
					f=fits.open(filename)
					f[2].data[ich,:,:]=timage
					fits.update(filename,f[2].data,f[2].header,"Data tangent plane")
				else:
					hdu0=fits.PrimaryHDU()
					hdu_ch = fits.ImageHDU()
					hdu_ch.header["EXTNAME"]="Channels"
					hdu_ch.data=mmfset.planck_channels
					hdu_map = fits.ImageHDU()
					hdu_map.header["EXTNAME"]="Data tangent plane"
					hdu_map.header["XYsize"]=str(mmfset.xsize) + " degrees"
					hdu_map.header["Reso"]=str(mmfset.reso) + " arcminutes"
					hdu_map.header["GLON"]=str(round(glon,4)) + " degrees"
					hdu_map.header["GLAT"]=str(round(glat,4)) + " degrees"
					hdu_map.header["SNR"]=str(snr)
					null_data=np.zeros((np.size(mmfset.planck_channels),mmfset.npix,mmfset.npix),float)
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
		for idx,clstrname in enumerate(mmf3["NAME"]):
			glon=mmf3["GLON"][idx]
			glat=mmf3["GLAT"][idx]
			filename=mmfset.paths["tplanes"] + "cluster_" + clstrname[5:] + ".fits"
			projop=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,glat,glon,rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			hdu_mask.data=timage
			fits.append(filename,hdu_mask.data,hdu_mask.header)
		
		# Extended point source mask
		if verbose:
			print "Appending extended point source masks"
		chmap=gen_ps_mask(ps_cutoff=5.,gen_mask=gen_mask)
		hdu_mask = fits.ImageHDU()
		hdu_mask.header["EXTNAME"]="Extended point Source Mask"
		for idx,clstrname in enumerate(mmf3["NAME"]):
			glon=mmf3["GLON"][idx]
			glat=mmf3["GLAT"][idx]
			filename=mmfset.paths["tplanes"] + "cluster_" + clstrname[5:] + ".fits"
			projop=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,glat,glon,rescale=1.)
			timage=projop.get_tangent_plane(chmap)
			hdu_mask.data=timage
			fits.append(filename,hdu_mask.data,hdu_mask.header)
	#return mmf3

def gen_ps_mask(snrthr=10.,ps_cutoff=3.,verbose=False,gen_mask=True):
	filename=mmfset.paths["planck_masks"] + "mmf3_ps_snr" + str(int(ps_cutoff)) + "_mask.fits"
	if (os.path.isfile(filename) and not(gen_mask)):
		mask=h.read_map(filename,verbose=verbose)
	else:
		mask=np.ones(h.nside2npix(mmfset.nside),float)
		#chmask=np.ones((np.size(mmfset.channels),h.nside2npix(mmfset.nside)),float)
		chmask=np.ones(h.nside2npix(mmfset.nside),float)
		for ich,ch in enumerate(mmfset.planck_channels):
			chmask[:]=1.
			f=fits.open(mmfset.ps_cat_fname[ch])
			radius=mmfset.ps_mask_weights[ch]*(ps_cutoff/np.sqrt(8.*np.log(2.)))*(mmfset.fwhm[ch]/60.)*np.pi/180.
			#print ch,radius

			detflux=f[f[1].header["EXTNAME"]].data.field("DETFLUX")
			err=f[f[1].header["EXTNAME"]].data.field("DETFLUX_ERR")
			snr_mask=[(detflux/err) >= snrthr]

			glon=detflux=f[f[1].header["EXTNAME"]].data.field("GLON")[snr_mask]
			glat=detflux=f[f[1].header["EXTNAME"]].data.field("GLAT")[snr_mask]
			pixcs=h.ang2pix(mmfset.nside,glon,glat,lonlat=True)

			if verbose:
				print ch,np.size(pixcs)

			for pix in pixcs:
				vec=h.pix2vec(mmfset.nside,pix)
				disc_pix=h.query_disc(mmfset.nside,vec,radius=radius,fact=4,inclusive=True)
				chmask[disc_pix]=0.0
			mask=mask*chmask

		h.write_map(filename,mask,overwrite=True)

	return mask

# This is obsolete as the point source mask map provided
# on the legacy archive also mask some of the clusters.
def return_ps_mask(return_gal_mask=False,idx=0):
	mask=np.ones(h.nside2npix(mmfset.nside),float)
	for i in range(6):
		mask=mask*h.read_map(mmfset.ps_mask_name,i,verbose=False)
	return mask

def get_tangent_plane_fnames(snrthr=6.,cosmo_flag=True,zknown=True):
    mmf3=get_mmf3_catalogue(snrthr=snrthr,cosmo_flag=cosmo_flag,zknown=True)

    tfname=[None]*np.size(mmf3["SNR"])
    for idx,clstrname in enumerate(mmf3["NAME"]):
        filename=mmfset.paths["tplanes"] + "cluster_" + clstrname[5:] + ".fits"
        tfname[idx]=filename

	mmf3["FILENAME"]=tfname
    return mmf3


def get_mmf3_catalogue(snrthr=6.,cosmo_flag=True,zknown=True):
    ints_sample = fits.open(mmfset.union_cat_file)
    mmf3_sample = fits.open(mmfset.mmf3_cat_file)

    # Getting the keys for the union catalogue
    keys_ints=ints_sample['PSZ2_UNION'].header.keys()
    fkeys_ints=[k for k in keys_ints if "TTYPE" in k]
    fields_ints=[ints_sample['PSZ2_UNION'].header[tkeys] for tkeys in fkeys_ints]
    #print fields_ints

    # Getting the keys for the MMF3 catalogue
    keys_mmf3=mmf3_sample['PSZ2_INDIVIDUAL'].header.keys()
    fkeys_mmf3=[k for k in keys_mmf3 if "TTYPE" in k]
    fields_mmf3=[mmf3_sample['PSZ2_INDIVIDUAL'].header[tkeys] for tkeys in fkeys_mmf3]

    # Getting the keys which are in UNION but not in MMF3 catalogue.
    fields=[f for f in fields_ints if f not in fields_mmf3]
	
    ints_raw={}
    for n in fields_ints:
	if n=="NAME":
            ints_raw[n]=list(ints_sample['PSZ2_UNION'].data.field(n).flatten())
    	else:
            ints_raw[n]=ints_sample['PSZ2_UNION'].data.field(n).flatten()

    mmf3_raw={}
    for n in fields_mmf3:
        if n=="NAME":
            mmf3_raw[n]=list(mmf3_sample['PSZ2_INDIVIDUAL'].data.field(n).flatten())
        else:
            mmf3_raw[n]=mmf3_sample['PSZ2_INDIVIDUAL'].data.field(n).flatten()

    for f in fields:
        mmf3_raw[f]=[]
        
    for i,n in enumerate(mmf3_raw["NAME"]):
        idx=ints_raw["NAME"].index(n)
        for f in fields:
            mmf3_raw[f]=np.append(mmf3_raw[f],ints_raw[f][idx])

	mask=(mmf3_raw["SNR"]>=snrthr)
    if cosmo_flag:
        mask=mask & (mmf3_raw["COSMO"]==1)
	if zknown:
		mask=mask & (mmf3_raw["REDSHIFT"]>=0.)

    mmf3={}
    for f in fields_ints:
        if f=="NAME":
            mmf3[f]=list(np.array(mmf3_raw[f])[mask])
        else:
            mmf3[f]=mmf3_raw[f][mask]

    return mmf3


def eval_M500_T500_theta500(clcat):
	Y500=np.zeros(np.size(clcat["Y5R500"]),float)
	M500=np.zeros(np.size(clcat["Y5R500"]),float)
	theta500=np.zeros(np.size(clcat["Y5R500"]),float)
	T500=np.zeros(np.size(clcat["Y5R500"]),float)
	
	Y500[:]=clcat["Y5R500"][:]*cosmo_fn.conv_y2y500
	M500[:]=cosmo_fn.convert_y500_M500(Y500,clcat["REDSHIFT"])
	T500[:]=cosmo_fn.convert_M500_T500(M500,clcat["REDSHIFT"])
	theta500[:]=cosmo_fn.convert_M500_theta500(M500,clcat["REDSHIFT"])
	
	clcat["M500"]=M500
	clcat["T500"]=T500
	clcat["theta500"]=theta500
	return clcat

