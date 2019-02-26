channels=[30.,44.,70.,100.,143.,217.,353.,545.,857.]

# From Planck SZ paper
fwhm={} # Arcminutes
fwhm[30.]=33.10 ; fwhm[44.]=27.94 ; fwhm[70.]=13.07
fwhm[100.]=9.66 ; fwhm[143.]=7.27 ; fwhm[217.]=5.01
fwhm[353.]=4.86 ; fwhm[545.]=4.84 ; fwhm[857.]=4.63

paths={}
paths["planck_bp"]="../data/Planck/channel_band_passes/"
paths["planck_mmf3_cat"]="../data/Planck/COM_PCCS_SZ-Catalogs_vPR2/"
paths["pccs"]="../data/Planck/PCCS/"
paths["sz_spec"]="../data/sz_spectra/"
paths["psm_sims"]="../data/simulations/psm_sims/"
paths["clusters"]="../data/simulations/clusters/"
paths["cmb"]="../data/simulations/cmb/"

cmb_spectra="../data/Planck/spectra/planck_wp_highL_lensing_param.fits"

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
conv_KCMB2MJY[100.]=1.
conv_KCMB2MJY[143.]=1.
conv_KCMB2MJY[217.]=1.
conv_KCMB2MJY[353.]=1.
conv_KCMB2MJY[545.]=58.0356
conv_KCMB2MJY[857.]=2.2681

conv_KRJ_KCMB={}
conv_KRJ_KCMB[30.]=1.023468
conv_KRJ_KCMB[44.]=1.0510271
conv_KRJ_KCMB[70.]=1.1331761
conv_KRJ_KCMB[100.]=1.2865727
conv_KRJ_KCMB[143.]=1.6535001
conv_KRJ_KCMB[217.]=2.9908492
conv_KRJ_KCMB[353.]=12.90104
conv_KRJ_KCMB[545.]=159.67296
conv_KRJ_KCMB[857.]=15700.02123
