import numpy as np

def gensim(cl,nxpix,pixel_size,ell=[],nypix=[]):
	'''
	cl : Assumed to be from ell=2:size(cl)+1 if ell not given
	nxpix : Number of pixel in the x-direction.
	pixel_size : The size of the pixel side in arcminutes (necessary to set the physical scale).
	ell : The multipoles corresponding to the cl.
	nypix : Number of pixels in the y-direction, if different from nxpix
	'''
	if nypix==[]:
		nypix=nxpix
	
	if ell==[]:
		ell=np.linspace(2,np.size(cl)+1,np.size(cl))

	pixel_area=((pixel_size/60.)*np.pi/180.)**2.
	alm=genalm(cl=cl,ell=ell,pixel_size=pixel_size,nxpix=nxpix,nypix=nypix)
	sim=alm2map(alm=alm,pixel_size=pixel_size)

	return sim

def genalm(cl,nxpix,pixel_size,ell=[],nypix=[]):
	'''
	cl : Assumed to be from ell=2:size(cl)+1 if ell not given
	nxpix : Number of pixel in the x-direction.
	pixel_size : The size of the pixel side in arcminutes (necessary to set the physical scale).
	ell : The multipoles corresponding to the cl.
	nypix : Number of pixels in the y-direction, if different from nxpix
	'''

	if nypix==[]:
		nypix=nxpix
	
	if ell==[]:
		ell=np.linspace(2,np.size(cl)+1,np.size(cl))
	
	pixel_area=((pixel_size/60.)*np.pi/180.)**2.

	alm=np.zeros((nxpix,nypix),dtype=np.complex64)
	
	if np.mod(nxpix,2)==0:
		zadrx=np.int(np.ceil(np.float(nxpix)/2))
		zadry=np.int(np.ceil(np.float(nypix)/2))
		origin=1
	else:
		zadrx=np.int(np.ceil(np.float(nxpix)/2))-1
		zadry=np.int(np.ceil(np.float(nypix)/2))-1
		origin=0
	#print origin,zadry,zadrx

	alm_r=np.random.normal(size=(zadry+1)*nxpix).reshape((zadry+1),nxpix)
	alm_i=np.random.normal(size=(zadry+1)*nxpix).reshape((zadry+1),nxpix)
	
	alm[:(zadry+1),:]=alm_r + 1j*alm_i
	# Here we are imposing the condition of reality of CMB maps.
	alm[zadry+1:,zadrx+1:]=np.conj(alm[origin:zadry,origin:zadrx][::-1,::-1]) # Diagonal mirroring
	alm[zadry+1:,origin:zadrx]=np.conj(alm[origin:zadry,zadrx+1:][::-1,::-1]) # Anti-diagonal mirroring
	alm[zadry,origin:zadrx]=np.conj(alm[zadry,zadrx+1:][::-1]) 		 # X-mirroring
	alm[zadry+1:,zadrx]=np.conj(alm[origin:zadry,zadrx][::-1])		 # Y-mirroring
	alm[zadry,zadrx]=np.sqrt(2.)*np.real(alm[zadry,zadrx])			 # 00 mode
	if origin==1:
		alm[0,zadrx+1:]=np.conj(alm[0,origin:zadrx][::-1])
		alm[0,0]=np.sqrt(2.)*np.real(alm[0,0])
		alm[0,zadrx]=np.sqrt(2.)*np.real(alm[0,zadrx])
		alm[zadry+1:,0]=np.conj(alm[origin:zadry,0][::-1])
		alm[zadry,0]=np.sqrt(2.)*np.real(alm[zadry,0])

	cl=cl*pixel_area*nxpix*nypix/2.
	filtr=get_fourier_filter(cl=np.sqrt(cl),ell=ell,nxpix=nxpix,nypix=nypix,pixel_size=pixel_size)
	alm=alm*filtr
	
	return alm

def map2cl(map,pixel_size,lmax=[],smwin=1,mapp=[]):
	'''
	map : The input sky map.
	pixel_size: Side of the pixel in the map in arcminutes (necessary to set the physical scale).
	lmax : The maximum multipole upto which to calculate the power spectrum.
	'''
	
	if mapp==[]:
		alm=map2alm(map,pixel_size=pixel_size)
		ell,cl=alm2cl(alm=alm,pixel_size=pixel_size,lmax=lmax,smwin=smwin)
	else:
		alm=map2alm(map,pixel_size=pixel_size)
		almp=map2alm(mapp,pixel_size=pixel_size)
		ell,cl=alm2cl(alm=alm,almp=almp,pixel_size=pixel_size,lmax=lmax,smwin=smwin)

	return ell,cl

def alm2map(alm,pixel_size):
	'''
	alm : The harmonics of the map.
	pixel_size: Side of a pixel in the map in arcminutes (necessary to set the physical scale).
	'''
	pixel_area=((pixel_size/60.)*np.pi/180.)**2.
	map=np.real(np.fft.ifft2(np.fft.ifftshift(alm),norm=None))/pixel_area
	#map=np.fft.ifft2(np.fft.fftshift(alm),norm=None)/pixel_area
	return map

def map2alm(map,pixel_size):
	'''
	map: The sky map for wants harmonics.
	pixel_size: Side of a pixel in the map in arcminutes (necessary to set the physical scale).
	'''
	pixel_area=((pixel_size/60.)*np.pi/180.)**2.
	alm=np.fft.fftshift(np.fft.fft2(map,norm=None))*pixel_area
	return alm

def alm2cl(alm,pixel_size,lmax=[],smwin=1,almp=[]):
	'''
	alm : The sky map harmonic coefficients.
	pixel_size: Side of the pixel in the map in arcminutes (necessary to set the physical scale).
	lmax : The maximum multipole upto which to calculate the power spectrum.
	almp : The spherical harmonic coefficients of another map. If given, this code returns the cross power spectrum.
	'''
	
	nyqfreq=180./(pixel_size/60.)
	if lmax==[]:
		lmax=nyqfreq
	else:
		lmax=min(lmax,nyqfreq)
	
	if almp==[]:
		ps=np.real(alm*np.conj(alm))
	else:
		ps=np.real(alm*np.conj(almp))

	nxpix, nypix = ps.shape[0], ps.shape[1]

	YY, XX = np.indices((ps.shape))
	XX=np.fft.fftshift(XX) ; YY=np.fft.fftshift(YY)
	maskY=((YY.ravel() > np.int(np.ceil(np.float(nypix)/2)-1)).astype(np.int)).reshape(nxpix,nypix)
	maskX=((XX.ravel() > np.int(np.ceil(np.float(nxpix)/2)-1)).astype(np.int)).reshape(nxpix,nypix)
	XX=(1.-maskX)*XX + nxpix*maskX-XX*maskX
	YY=(1.-maskY)*YY + nypix*maskY-YY*maskY
	k = 360./(pixel_size/60.)*np.sqrt((XX/nxpix)**2. + (YY/nypix)**2.)

	k = k.ravel() ; ps = ps.ravel()
	freq=np.fft.fftshift(np.fft.fftfreq(nxpix,1./(nyqfreq*2.)))
	freq=freq[(freq>=0) & (freq<=lmax)]
	
	pixel_area=((pixel_size/60.)*np.pi/180.)**2.
	
	ell = np.zeros(len(freq)-smwin,dtype=np.float64)
	cl = np.zeros(len(freq)-smwin, dtype=np.float64)
	for i in np.arange(len(freq)-smwin):
		index = np.where(((k>=freq[i]) & (k<freq[i+smwin])))[0]
		ell[i] = np.mean(k[index])
		cl[i] = np.mean(ps[index])/(pixel_area*nxpix*nypix)

	return ell,cl

def get_fourier_filter(cl,nxpix,pixel_size,ell=[],nypix=[]):
	'''
	cl : Assumed to be from ell=2:size(cl)+1 if ell not given.
	nxpix : Number of pixel in the x-direction.
	pixel_size : The size of the pixel side in arcminutes (necessary to set the physical scale).
	ell : The multipoles corresponding to the cl.
	nypix : Number of pixels in the y-direction, if different from nxpix.
	'''
	
	if nypix==[]:
		nypix=nxpix
	
	if ell==[]:
		ell=np.linspace(2,np.size(cl)+1,np.size(cl))

	filtr=np.zeros((nxpix,nypix),dtype=np.float64)
	YY, XX = np.indices((filtr.shape))
	XX=np.fft.fftshift(XX) ; YY=np.fft.fftshift(YY)
	maskY=((YY.ravel() > np.int(np.ceil(np.float(nypix)/2)-1)).astype(np.int)).reshape(nxpix,nypix)
	maskX=((XX.ravel() > np.int(np.ceil(np.float(nxpix)/2)-1)).astype(np.int)).reshape(nxpix,nypix)
	XX=(1.-maskX)*XX + nxpix*maskX-XX*maskX
	YY=(1.-maskY)*YY + nypix*maskY-YY*maskY
	k = (360./(pixel_size/60.))*np.sqrt((XX/nxpix)**2. + (YY/nypix)**2.)

	k = k.ravel()
	filtr=filtr.ravel()
	filtr=np.interp(k,ell,cl)
	filtr=filtr.reshape(nxpix,nypix)
	
	return filtr

def filter_map(map,pixel_size,bl,ell=[]):
	'''
	map : The map which need filtering.
	pixel_size: Side of a pixel in the map in arcminutes (necessary to set the physical scale).
	bl  : The harmonic space window function, assumed to be from ell=2:size(bl)+1 if ell not given.
	'''
	
	if ell==[]:
		ell=np.linspace(2,np.size(bl)+1,np.size(bl))
	
	falm=filter_alm(alm=map2alm(map,pixel_size),pixel_size=pixel_size,bl=bl,ell=ell)
	fmap=alm2map(falm,pixel_size=pixel_size)

	return fmap

def filter_alm(alm,pixel_size,bl,ell=[]):
	'''
	alm : The harmonic coefficients which need filtering.
	pixel_size: Side of a pixel in the map in arcminutes (necessary to set the physical scale).
	bl  : The harmonic space window function, assumed to be from ell=2:size(bl)+1 if ell not given.
	'''
	
	if ell==[]:
		ell=np.linspace(2,np.size(bl)+1,np.size(bl))

	nxpix, nypix = alm.shape[0], alm.shape[1]
	filtr=get_fourier_filter(cl=bl,ell=ell,nxpix=nxpix,nypix=nypix,pixel_size=pixel_size)
	falm=alm*filtr

	return falm

def get_azi_avg(map,pixel_size,smwin=1):
	'''
	map : The map which need to be averaged.
	pixel_size : The size of the pixel side in arcminutes (necessary to set the physical scale).
	'''
	
	nxpix, nypix = map.shape[0], map.shape[1]

	YY, XX = np.indices((map.shape))
	XX=np.fft.fftshift(XX) ; YY=np.fft.fftshift(YY)
	maskY=((YY.ravel() > np.int(np.ceil(np.float(nypix)/2)-1)).astype(np.int)).reshape(nxpix,nypix)
	maskX=((XX.ravel() > np.int(np.ceil(np.float(nxpix)/2)-1)).astype(np.int)).reshape(nxpix,nypix)
	XX=(1.-maskX)*XX + nxpix*maskX-XX*maskX
	YY=(1.-maskY)*YY + nypix*maskY-YY*maskY
	radius = np.sqrt((XX/nxpix)**2. + (YY/nypix)**2.)*pixel_size

	dradius=(XX[0,:]/nxpix)*pixel_size ; dradius=np.unique(dradius[dradius>=0.])
	radius = radius.ravel() ; map=map.ravel()
	
	ravg = np.zeros(len(dradius)-smwin,dtype=np.float64)
	mapavg = np.zeros(len(dradius)-smwin, dtype=np.float64)
	for i in np.arange(len(dradius)-smwin):
		index = np.where(((radius>=dradius[i]) & (radius<dradius[i+smwin])))[0]
		ravg[i] = np.mean(radius[index])
		mapavg[i] = np.mean(map[index])

	return ravg,mapavg


def get_gauss_beam(fwhm,lmax):
	'''
	fwhm : The FWHM of the Gaussian beam in arcminutes.
	lmax : The maximum multipole upto which the harmonic space beam is to be returned.
	'''
	
	ell=np.arange(lmax+1)
	theta=((fwhm/60.)*np.pi/180.)/np.sqrt(8.*np.log(2.))
	bl=np.exp(-theta*theta*ell*(ell+1)/2.)
	return ell,bl


