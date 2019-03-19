import numpy as np
from flat_sky_codes import flat_sky_analysis as fsa

def return_ps_filled_data(data,mask,pixel_size,diffthr=1e-4,itermax=20):
    npix=data.shape[0]
	
    ellstep=np.arange(500,6500,500)
    bl=np.ones(6000,float) ; elld=np.arange(np.size(bl))
	
    fdata=np.zeros_like(data)
    for lmax in ellstep:
        bl=np.ones(6000,float)
        bl[elld>lmax]=0.
        lpfiltr=fsa.get_fourier_filter(cl=bl,ell=elld,nxpix=npix,pixel_size=pixel_size)
        imp=1. ; diff=1. ; iterations=0
        while diff>diffthr and iterations<itermax:
            ofdata=data+fdata*(1-mask)
            fdatalm=fsa.map2alm(ofdata,pixel_size=pixel_size)
            fdata=fsa.alm2map(fdatalm*lpfiltr,pixel_size=pixel_size)
            temp=np.sum(((fdata-ofdata).ravel())**2.)
            diff = np.abs(imp-temp)/imp
            imp=temp
            iterations=iterations+1
            #print lmax,iterations,diff
    return fdata
