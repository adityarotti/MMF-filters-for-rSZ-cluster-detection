##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  15 January September 2019     				 		                             #
# Date modified: 16 March 2019								 								     #
##################################################################################################


import numpy as np
import healpy as h


class tangent_plane_setup(object):

	def __init__(self,nside,ang_size,glat,glon,rescale=1.,coord="G"):
		'''
		nside : The Healpix resolution of the map.
		ang_size: The side of the patch in degrees.
		glat, glon: The galactic coordinates of the center of the tangent plane (degrees)
		rescale : rescales the pixel size to return the flat sky map at an alternate resolution.
		'''
		self.nside=nside
		self.ang_size=ang_size
		self.glat=glat
		self.glon=glon
		self.rescale=rescale

		self.pixel_size=self.rescale*np.sqrt(4.*np.pi/h.nside2npix(self.nside))*(180./np.pi)*60.
		self.npix=np.int(self.ang_size*60./self.pixel_size)
		self.projop=h.projector.GnomonicProj(xsize=self.npix,ysize=self.npix,coord=coord,reso=self.pixel_size,rot=[self.glon,self.glat])


	def vec2pix(self,x,y,z):
		return h.vec2pix(self.nside,x,y,z)


	def get_tangent_plane(self,map):
		'''
		This function returns the tangent plane centered on the coordinate used to
		definite the projection operator.
		'''
		image=self.projop.projmap(map,self.vec2pix)
		return image


	def ij2ang(self,i,j):
		'''
		This function returns the galactic longitude and latitude given the pixel numbers along x and y axis.
		'''
		x,y=self.projop.ij2xy(i,j)
		glon,glat=self.projop.xy2ang(x,y,lonlat=True)
		
		return glon,glat


	def ang2ij(self,glon,glat):
		'''
		This function returns the pixel numbers along x and y axis given the coordinates in galactic coordinates (degrees.)
		'''
		x,y=self.projop.ang2xy(theta=glon,phi=glat,lonlat=True)
		ix,iy=self.projop.xy2ij(x=x,y=y)
		return ix,iy


	def distance(self,ic,jc,i,j):
		'''
		This function returns the distance between the points on the plane in arcminutes,
		given the cartesian indices of the two locations on the plane.
		'''
		xc,yc=self.projop.ij2xy(ic,jc)
		x,y=self.projop.ij2xy(i,j)
		d=np.sqrt((x-xc)**2. + (y-yc)**2.)*180.*60./np.pi
		return d
