##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################

import numpy as np
from modules.settings import global_mmf_settings as gset
from cosmology import cosmo_fn as clcosmo
from spatial_template import sim_cluster as sc
from flat_sky_codes  import flat_sky_analysis as fsa


def return_mc_cluster(M500,z,op,y0=1e-4):
	Tc=clcosmo.convert_M500_T500(M500,z)
	thetac=clcosmo.convert_M500_theta500(M500,z)
	template=sc.gen_cluster_template(gset.mmfset.npix,thetac,gset.mmfset.reso,y0=y0)
	temp_ft=fsa.map2alm(template,gset.mmfset.reso)

	template_ft=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),complex)
	cluster=np.zeros((np.size(gset.mmfset.channels),gset.mmfset.npix,gset.mmfset.npix),float)

	for i, ch in enumerate(gset.mmfset.channels):
		template_ft[i,:,:]=temp_ft*op.chfiltr[ch]*op.sz_op.fn_sz_2d_T(Tc,ch)
		cluster[i,:,:]=fsa.alm2map(template_ft[i,:,:],gset.mmfset.reso)

	return cluster,Tc,thetac
