import os
import numpy as np
import healpy as h
from astropy.io import fits
from modules.settings import global_mmf_settings as gset
from flat_sky_codes import tangent_plane_analysis as tpa
import unit_conv as uc

def get_reduced_pico_sims():
	for ch in gset.mmfset.planck_channels:
		# Noise
		filename=gset.mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group13_map_" + str(int(ch))+"GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		h.write_map(gset.mmfset.map_fnames["noise"][ch],tempmap,overwrite=True)
		
		#All sky
		filename=gset.mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group2_map_" + str(int(ch))+"GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		
		# Thermal SZ
		filename=gset.mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group4_map_" + str(int(ch))+"GHz.fits"
		tempmap=tempmap-h.read_map(filename,verbose=False)

		# Kinetic SZ
		filename=gset.mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group5_map_" + str(int(ch))+"GHz.fits"
		tempmap=tempmap-h.read_map(filename,verbose=False)
		
		# CMB
		filename=gset.mmfset.paths["pico_sims"] +  str(int(ch))+"GHz/" + "group3_map_" + str(int(ch))+"GHz.fits"
		cmbmap=h.read_map(filename,verbose=False)
		
		h.write_map(gset.mmfset.map_fnames["frg"][ch],tempmap-cmbmap,overwrite=True)
		h.write_map(gset.mmfset.map_fnames["cmb"][ch],cmbmap,overwrite=True)



def get_random_plane_centers(numplanes=100,minlat=20.,nsideout=8,seed=0):
	glon,glat=h.pix2ang(nsideout,np.arange(h.nside2npix(nsideout)),lonlat=True)
	bandmask=np.zeros(np.size(glat),float)
	bandmask[abs(glat)>minlat]=1.
	pixels=np.where(bandmask==1.)[0]
	np.random.seed(seed)
	rpixels=np.random.choice(pixels,numplanes)
	bandmask[rpixels]=10
	return rpixels,bandmask

def extract_tangent_planes(numplanes=100,minlat=20.,nsideout=8,seed=0,dryrun=False):
	rpixels,bandmask=get_random_plane_centers(numplanes=numplanes,minlat=minlat,nsideout=nsideout,seed=seed)
	glon,glat=h.pix2ang(nsideout,rpixels,lonlat=True)

	if dryrun:
		for mtype in gset.mmfset.map_fnames.keys():
			for ich, ch in enumerate(gset.mmfset.channels):
				#print gset.mmfset.map_fnames[mtype][ch]
				print ch,uc.conv_uKTRJ_KTBB(ch)
				for ipix,pix in enumerate(rpixels):
					filename=gset.mmfset.paths["tplanes"] + mtype + "_pico_sim_tplane_" + str(ipix) + ".fits"
					#print filename
	else:
		for mtype in gset.mmfset.map_fnames.keys():
			for ich, ch in enumerate(gset.mmfset.channels):
				#print gset.mmfset.map_fnames[mtype][ch]
				chmap=h.read_map(gset.mmfset.map_fnames[mtype][ch],0,verbose=False)*uc.conv_uKTRJ_KTBB(ch)
				for ipix,pix in enumerate(rpixels):
					filename=gset.mmfset.paths["tplanes"] + mtype + "_pico_sim_tplane_" + str(ipix) + ".fits"
					projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat[ipix],glon[ipix],rescale=1.)
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
						hdu_map.header["GLON"]=str(round(glon[ipix],4)) + " degrees"
						hdu_map.header["GLAT"]=str(round(glat[ipix],4)) + " degrees"
						null_data=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)
						null_data[ich,:,:]=timage
						hdu_map.data=null_data
						hdu=fits.HDUList([hdu0,hdu_ch,hdu_map])
						hdu.writeto(filename)
