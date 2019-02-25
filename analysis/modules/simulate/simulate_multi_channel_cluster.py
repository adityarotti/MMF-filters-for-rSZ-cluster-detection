import numpy as np
from cosmology import cosmo_fn as clcosmo
from settings import mmf_settings as mmfset
from spatial_template import sim_cluster as sc
from flat_sky_codes  import flat_sky_analysis as fsa


def return_mc_cluster(M500,z,op,y0=1e-4):
	Tc=clcosmo.convert_M500_T500(M500,z)
	thetac=clcosmo.convert_M500_theta500(M500,z)
	template=sc.gen_cluster_template(mmfset.npix,thetac,mmfset.reso,y0=y0)
	temp_ft=fsa.map2alm(template,mmfset.reso)

	template_ft=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),complex)
	cluster=np.zeros((np.size(mmfset.channels),mmfset.npix,mmfset.npix),float)

	for i, ch in enumerate(mmfset.channels):
		template_ft[i,:,:]=temp_ft*op.chfiltr[ch]*op.sz_op.fn_sz_2d_T(Tc,ch)
		cluster[i,:,:]=fsa.alm2map(template_ft[i,:,:],mmfset.reso)

	return cluster,Tc,thetac
