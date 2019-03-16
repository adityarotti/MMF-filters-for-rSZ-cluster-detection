all_channels=[21.,25.,30.,36.,43.,52.,62.,75.,90.,110.,130.,155.,185.,225.,270.,320.,385.,460.,555.,665.,800.]

pico_fwhm=[40.9,34.1,28.4,23.7,19.7,16.4,13.7,11.4,9.5,7.9,6.6,5.5,4.6,3.8,3.2,2.7,2.2,1.8,1.5,1.3,1.1]
fwhm={}
for ich,ch in enumerate(all_channels):
	fwhm[ch]=pico_fwhm[ich]

pico_nstd = [35.3553, 23.3345, 15.8392, 10.6066, 6.43467, 4.94975, 3.53553, 2.82843, 2.26274, 2.05061, 1.90919, 1.83848, 2.54558, 3.74767, 6.36396, 11.3137, 22.6274, 53.0330, 155.563, 777.817, 7071.07]

paths={}
planck_data_dir="/Users/adityarotti/Documents/Work/Data/Planck/"
#planck_data_dir="/mirror/arotti/Planck/"
paths["planck_mmf3_cat"]=planck_data_dir + "/COM_PCCS_SZ-Catalogs_vPR2/"
mmf3_cat_file=paths["planck_mmf3_cat"] + "HFI_PCCS_SZ-MMF3_R2.08.fits"
union_cat_file=paths["planck_mmf3_cat"] + "HFI_PCCS_SZ-union_R2.08.fits"
esz_cat_2011_file=paths["planck_mmf3_cat"] + "esz_cat_2011.txt"

datain_dir="/mirror/arotti/simulations/PICO/"
paths["pico_sims"]=datain_dir + "/CMB_PROBE_2017/"
paths["reduced_pico_sims"]=datain_dir + "/reduced_data/"
paths["sz_spec"]="../data/sz_spectra/"

map_fnames={}
map_fnames["cmb"]={}
map_fnames["noise"]={}
map_fnames["frg"]={}

for ch in all_channels:
	map_fnames["noise"][ch]=paths["reduced_pico_sims"] + "noise_" + str(int(ch))+"GHz.fits"
	map_fnames["cmb"][ch]=paths["reduced_pico_sims"] + "cmb_" + str(int(ch))+"GHz.fits"
	map_fnames["frg"][ch]=paths["reduced_pico_sims"] + "frg_" + str(int(ch))+"GHz.fits"
