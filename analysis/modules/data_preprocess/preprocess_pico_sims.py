import os
import numpy as np
import healpy as h
from astropy.io import fits
from settings import mmf_settings as mmfset
from flat_sky_codes import tangent_plane_analysis as tpa


def get_reduced_pico_sims():
	for ch in mmfset.channels:
		# Noise
		filename=mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group13_map_" + str(int(ch))+"GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		h.write_map(mmfset.map_names["noise"][ch],tempmap,overwrite=True)
		
		# CMB
		filename=mmfset.paths["pico_sims"] +  str(int(ch))+"GHz/" + "group3_map_" + str(int(ch))+"GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		h.write_map(mmfset.map_names["cmb"][ch],tempmap,overwrite=True)

		#All sky
		filename=mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group2_map_" + str(int(ch))+"GHz.fits"
		tempmap=h.read_map(filename,verbose=False)
		
		# Thermal SZ
		filename=mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group4_map_" + str(int(ch))+"GHz.fits"
		tempmap=tempmap-h.read_map(filename,verbose=False)

		# Kinetic SZ
		filename=mmfset.paths["pico_sims"] + str(int(ch))+"GHz/" + "group5_map_" + str(int(ch))+"GHz.fits"
		tempmap=tempmap-h.read_map(filename,verbose=False)
		
		h.write_map(mmfset.map_names["cmbfrg"][ch],tempmap,overwrite=True)

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
		for mtype in mmfset.map_fnames.keys():
			for ich, ch in enumerate(mmfset.channels):
				print mmfset.map_fnames[mtype][ch]
				for ipix,pix in enumerate(rpixels):
					filename=mmfset.paths["tplanes"] + mtype + "_pico_sim_tplane_" + str(ipix) + ".fits"
					print filename
	else:
		for mtype in mmfset.map_fnames.keys():
			for ich, ch in enumerate(mmfset.channels):
				#print mmfset.map_fnames[mtype][ch]
				chmap=h.read_map(mmfset.map_fnames[mtype][ch],0,verbose=False) #/mmfset.conv_KCMB2MJY[ch]
				for ipix,pix in enumerate(rpixels):
					filename=mmfset.paths["tplanes"] + mtype + "_pico_sim_tplane_" + str(ipix) + ".fits"
					projop=tpa.tangent_plane_setup(mmfset.nside,mmfset.xsize,glat[ipix],glon[ipix],rescale=1.)
					timage=projop.get_tangent_plane(chmap)

					if os.path.isfile(filename):
						f=fits.open(filename)
						f[2].data[ich,:,:]=timage
						fits.update(filename,f[2].data,f[2].header,"Data tangent plane")
					else:
						hdu0=fits.PrimaryHDU()
						hdu_ch = fits.ImageHDU()
						hdu_ch.header["EXTNAME"]="Channels"
						hdu_ch.data=mmfset.channels
						hdu_map = fits.ImageHDU()
						hdu_map.header["EXTNAME"]="Data tangent plane"
						hdu_map.header["XYsize"]=str(mmfset.xsize) + " degrees"
						hdu_map.header["Reso"]=str(mmfset.reso) + " arcminutes"
						hdu_map.header["GLON"]=str(round(glon[ipix],4)) + " degrees"
						hdu_map.header["GLAT"]=str(round(glat[ipix],4)) + " degrees"
						null_data=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),float)
						null_data[ich,:,:]=timage
						hdu_map.data=null_data
						hdu=fits.HDUList([hdu0,hdu_ch,hdu_map])
						hdu.writeto(filename)
