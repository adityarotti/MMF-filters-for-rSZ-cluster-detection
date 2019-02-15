import os
import numpy as np
import healpy as h

pico_channels=np.array([21.,25.,30.,36.,43.,52.,62.,75.,90.,110.,130.,155.,185.,225.,270.,320.,385.,460.,555.,665.,800.])
pico_fwhm =np.array([40.9,34.1,28.4,23.7,19.7,16.4,13.7,11.4,9.5,7.9,6.6,5.5,4.6,3.8,3.2,2.7,2.2,1.8,1.5,1.3,1.1])
pico_nstd = np.array([35.3553, 23.3345, 15.8392, 10.6066, 6.43467, 4.94975, 3.53553, 2.82843, 2.26274, 2.05061, 1.90919, 1.83848, 2.54558, 3.74767, 6.36396, 11.3137, 22.6274, 53.0330, 155.563, 777.817, 7071.07])

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

datapath="/global/cscratch1/sd/raditya/PICO/CMB_PROBE_2017/"
datapathout="/global/cscratch1/sd/raditya/PICO/reduced_data/"
ensure_dir(datapathout)

for ch in pico_channels:
	# Noise
	filename=datapath + str(int(ch))+"GHz/" + "group13_map_" + str(int(ch))+"GHz.fits"
	tempmap=h.read_map(filename,verbose=False)
	filename=datapathout + "noise_" + str(int(ch))+"GHz.fits"
	h.write_map(filename,tempmap,overwrite=True)
	
	# CMB
	filename=datapath + str(int(ch))+"GHz/" + "group3_map_" + str(int(ch))+"GHz.fits"
	tempmap=h.read_map(filename,verbose=False)
	filename=datapathout + "cmb_" + str(int(ch))+"GHz.fits"
	h.write_map(filename,tempmap,overwrite=True)

	#All sky
	filename=datapath + str(int(ch))+"GHz/" + "group2_map_" + str(int(ch))+"GHz.fits"
	tempmap=h.read_map(filename,verbose=False)
	
	# Thermal SZ
	filename=datapath + str(int(ch))+"GHz/" + "group4_map_" + str(int(ch))+"GHz.fits"
	tempmap=tempmap-h.read_map(filename,verbose=False)

	# Kinetic SZ
	filename=datapath + str(int(ch))+"GHz/" + "group5_map_" + str(int(ch))+"GHz.fits"
	tempmap=tempmap-h.read_map(filename,verbose=False)
	
#	# Strong point sources
#	filename=datapath + str(int(ch))+"GHz/" + "group12_map_" + str(int(ch))+"GHz.fits"
#	tempmap=tempmap-h.read_map(filename,verbose=False)
#
	filename=datapathout + "cmb_rfrg_" + str(int(ch))+"GHz.fits"
	h.write_map(filename,tempmap,overwrite=True)



# NOTES:
#COMPONENT GROUP 1: all
#COMPONENT GROUP 2: allsky
#COMPONENT GROUP 3: cmb
#COMPONENT GROUP 4: thermalsz
#COMPONENT GROUP 5: kineticsz
#COMPONENT GROUP 6: synchrotron
#COMPONENT GROUP 7: freefree
#COMPONENT GROUP 8: thermaldust
#COMPONENT GROUP 9: spindust
#COMPONENT GROUP 10: firb
#COMPONENT GROUP 11: faintps
#COMPONENT GROUP 12: strongps
#COMPONENT GROUP 13: noise
