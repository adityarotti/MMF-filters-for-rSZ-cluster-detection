##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################
import socket
hostname = socket.gethostname()

all_channels=[30.,44.,70.,100.,143.,217.,353.,545.,857.]

#fwhm={} # Arcminutes
#fwhm[30.]=32.29 ; fwhm[44.]=27.94 ; fwhm[70.]=13.08
#fwhm[100.]=9.66 ; fwhm[143.]=7.22 ; fwhm[217.]=4.90
#fwhm[353.]=4.92 ; fwhm[545.]=4.67 ; fwhm[857.]=4.22

## From Planck 2015 XXII :  A map of the thermal Sunyaev-Zeldovich effect
#fwhm={} # Arcminutes
#fwhm[30.]=33.10 ; fwhm[44.]=27.94 ; fwhm[70.]=13.07
#fwhm[100.]=9.66 ; fwhm[143.]=7.27 ; fwhm[217.]=5.01
#fwhm[353.]=4.86 ; fwhm[545.]=4.84 ; fwhm[857.]=4.63

# From Planck 2015 XXVII The Second Planck Catalogue of Sunyaev-Zeldovich Sources
fwhm={} # Arcminutes
fwhm[30.]=33.10 ; fwhm[44.]=27.94 ; fwhm[70.]=13.07
fwhm[100.]=9.659 ; fwhm[143.]=7.220 ; fwhm[217.]=4.900
fwhm[353.]=4.916 ; fwhm[545.]=4.675 ; fwhm[857.]=4.216

datain_dir="/Users/adityarotti/Documents/Work/Data/Planck/"
if hostname=="sirius.jb.man.ac.uk":
	datain_dir="/mirror/arotti/Planck/"

paths={}
paths["planck_maps"]=datain_dir + "/maps/"
paths["planck_masks"]=datain_dir + "/masks/"
paths["planck_bp"]=datain_dir + "/channel_band_passes/"
paths["planck_mmf3_cat"]=datain_dir + "/COM_PCCS_SZ-Catalogs_vPR2/"
paths["pccs"]=datain_dir + "/PCCS/"

# Cluster catalogue
mmf3_cat_file=paths["planck_mmf3_cat"] + "HFI_PCCS_SZ-MMF3_R2.08.fits"
union_cat_file=paths["planck_mmf3_cat"] + "HFI_PCCS_SZ-union_R2.08.fits"
esz_cat_2011_file=paths["planck_mmf3_cat"] + "esz_cat_2011.txt"
cosmo_cat_2013_ysz_file=paths["planck_mmf3_cat"] + "cosmo_cluster_sample_ysz.fits"
cosmo_cat_2013_yxray_file=paths["planck_mmf3_cat"] + "cosmo_cluster_sample_yxray.dat"

# Planck maps
map_fnames={}
map_fnames[30.]=paths["planck_maps"] + "LFI_SkyMap_030-BPassCorrected-field-IQU_2048_R3.00_full.fits"
map_fnames[44.]=paths["planck_maps"] + "LFI_SkyMap_044-BPassCorrected-field-IQU_2048_R3.00_full.fits"
map_fnames[70.]=paths["planck_maps"] + "LFI_SkyMap_070-BPassCorrected-field-IQU_2048_R3.00_full.fits"
map_fnames[100.]=paths["planck_maps"] + "HFI_SkyMap_100-field-IQU_2048_R3.00_full.fits"
map_fnames[143.]=paths["planck_maps"] + "HFI_SkyMap_143-field-IQU_2048_R3.00_full.fits"
map_fnames[217.]=paths["planck_maps"] + "HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits"
map_fnames[353.]=paths["planck_maps"] + "HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full.fits"
map_fnames[545.]=paths["planck_maps"] + "HFI_SkyMap_545-field-Int_2048_R3.00_full.fits"
map_fnames[857.]=paths["planck_maps"] + "HFI_SkyMap_857-field-Int_2048_R3.00_full.fits"

# Masks
gal_mask_name=paths["planck_masks"] + "COM_Mask_PCCS-143-zoneMask_2048_R2.01.fits"
ps_mask_name=paths["planck_masks"] + "HFI_Mask_PointSrc_2048_R2.00.fits"

# Point source catalogue
ps_cat_fname={}
ps_cat_fname[30.] =paths["pccs"] + "COM_PCCS_030_R2.04.fits"
ps_cat_fname[44.] =paths["pccs"] + "COM_PCCS_044_R2.04.fits"
ps_cat_fname[70.] =paths["pccs"] + "COM_PCCS_070_R2.04.fits"
ps_cat_fname[100.]=paths["pccs"] + "COM_PCCS_100_R2.01.fits"
ps_cat_fname[143.]=paths["pccs"] + "COM_PCCS_143_R2.01.fits"
ps_cat_fname[217.]=paths["pccs"] + "COM_PCCS_217_R2.01.fits"
ps_cat_fname[353.]=paths["pccs"] + "COM_PCCS_353_R2.01.fits"
ps_cat_fname[545.]=paths["pccs"] + "COM_PCCS_545_R2.01.fits"
ps_cat_fname[857.]=paths["pccs"] + "COM_PCCS_857_R2.01.fits"

conv_uK2K=1e-6
sigma={} #uK_deg converted to K_arcminute
sigma[30.]=2.5*60.*conv_uK2K
sigma[44.]=2.7*60.*conv_uK2K
sigma[70.]=3.5*60.*conv_uK2K
sigma[100.]=1.29*60.*conv_uK2K
sigma[143.]=0.55*60.*conv_uK2K
sigma[217.]=0.78*60.*conv_uK2K
sigma[353.]=2.56*60.*conv_uK2K
sigma[545.]=0.78*60.*1000.*conv_uK2K/57.1943
sigma[857.]=0.72*60.*1000.*conv_uK2K/1.43907

conv_KCMB2MJY={}
conv_KCMB2MJY[30.]=1.
conv_KCMB2MJY[44.]=1.
conv_KCMB2MJY[70.]=1.
conv_KCMB2MJY[100.]=1.
conv_KCMB2MJY[143.]=1.
conv_KCMB2MJY[217.]=1.
conv_KCMB2MJY[353.]=1.
conv_KCMB2MJY[545.]=58.0356
conv_KCMB2MJY[857.]=2.2681
#conv_KRJ_KCMB={}
#conv_KRJ_KCMB[30.]=1.023468
#conv_KRJ_KCMB[44.]=1.0510271
#conv_KRJ_KCMB[70.]=1.1331761
#conv_KRJ_KCMB[100.]=1.2865727
#conv_KRJ_KCMB[143.]=1.6535001
#conv_KRJ_KCMB[217.]=2.9908492
#conv_KRJ_KCMB[353.]=12.90104
#conv_KRJ_KCMB[545.]=159.67296
#conv_KRJ_KCMB[857.]=15700.02123
