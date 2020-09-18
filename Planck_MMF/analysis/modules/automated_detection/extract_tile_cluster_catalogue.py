##################################################################################################
# Author: Aditya Rotti, Jodrell Bank Center for Astrophysics, University of Manchester           #
# Date created:  17 September 2020		     				 		                             #
# Date modified: 17 September 2020																 #
##################################################################################################

import os
import numpy as np
import collections
from modules.settings import global_mmf_settings as gset
from modules.masking import gen_masks as gm
from data_preprocess import get_tangent_planes as gtp
from modules.flat_sky_codes import tangent_plane_analysis as tpa
from modules.flat_sky_codes import flat_sky_analysis as fsa
from filters import modular_multi_matched_filter as mmf
from simulate.spatial_template import sim_cluster as sc
from simulate.spatial_template import sz_pressure_profile as szp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
from read_write_dict import write_dict
from read_write_dict import read_dict

emask=gm.return_edge_apodized_mask(edge_width=30.,fwhm=30.)

def extract_tile_cluster_catalogue(tile_px,tiledef,tmplt,genfig=True,snrthr=4):
	filename=tiledef[tile_px]["FILENAME"]
	tilename=tiledef[tile_px]["TILENAME"]
	ext_ps_mask=gtp.return_ext_ps_mask(filename)
	ps_mask=gtp.return_ps_mask(filename)
	gmask=gtp.return_galactic_mask(filename) ; fsky=np.sum(gmask)/np.size(gmask)
	det_mask=gmask*ext_ps_mask*emask
	data=gtp.return_data(filename) ; data=data*gmask*emask*ps_mask
	glon=tiledef[tile_px]["GLON"] ; glat=tiledef[tile_px]["GLAT"]
	projop=tpa.tangent_plane_setup(gset.mmfset.nside,gset.mmfset.xsize,glat,glon,rescale=1.)
	cluspath=gset.mmfset.paths["result_figs"] + tilename + "/" ; ensure_dir(cluspath)
	Ycyl2YsphR500=szp.convert_Ycyl_xR500_Ysph_xR500(xcyl=tmplt.cutoff,xsph=1.)

	tile_char=collections.OrderedDict()
	tile_char["pix"]=tile_px
	tile_char["fsky"]=fsky
	tile_char["theta500"]=np.array([])
	tile_char["err_Y500"]=np.array([])
	tile_char["cat"]={}

	opmmf=mmf.multi_matched_filter(tmplt.sp_ft_bank,tmplt.sz_spec_bank,tmplt.chfiltr,tmplt.fn_yerr_norm)

	for ith,theta500 in enumerate(tmplt.theta500[::-1]):
		opmmf.get_data_ft(data)
		fdata,err,snr,yc=opmmf.return_snr(theta500,0.,mask_fdata=False)
		err=err/np.sqrt(tile_char["fsky"])
		tile_char["theta500"]=np.append(tile_char["theta500"],theta500)
		tile_char["err_Y500"]=np.append(tile_char["err_Y500"],err*tmplt.fn_yerr_norm(theta500)*Ycyl2YsphR500)
		tile_char["cat"]=return_cluster_cat(fdata*det_mask,err,theta500,tile_char["cat"],projop,tmplt,snrthr=snrthr,verbose=False)
		filename=cluspath + tilename + "_clusdet_iter" + str(ith).zfill(2) + ".jpeg"
		if genfig:
			gen_tile_figs(fdata*det_mask,theta500,err,tile_char["cat"],filename,showplt=False)

	if genfig:
		cmd="convert -delay 50 " + cluspath + "*.jpeg " + cluspath + tilename + "def.gif"
		os.system(cmd)

	write_dict(tiledef[tile_px]["CATNAME"],tile_char)

def return_final_cluster_catalogue(tiledef,verbose=False):
    tile_cluscat={}
    for px in tiledef.keys():
        temp_cat=read_dict(tiledef[px]["CATNAME"])
        tile_cluscat[tilename]=temp_cat["cat"]
	
    final_cat={}
    for tile in tile_cluscat.keys():
        for ict in tile_cluscat[tile].keys():
            nc=[]
            for jct in final_cat.keys():
                dist=return_distance(tile_cluscat[tile][ict]["mp_gal_coord"],final_cat[jct]["mp_gal_coord"])
                nc=nc + [dist<tile_cluscat[tile][ict]["mp_thetac"] or dist<final_cat[jct]["mp_thetac"]]

            if any(nc):
                myprint("This cluster exists in the catalogue",verbose)
                match_ict=np.where(nc)[0][0]
                if tile_cluscat[tile][ict]["mp_snr"]>final_cat[match_ict]["mp_snr"]:
                    myprint("Updating the cluster definition",verbose)
                    final_cat[match_ict]=tile_cluscat[tile][ict]
            else:
                myprint("New cluster detected, adding to the cluster catalogue",verbose)
                match_ict=len(final_cat.keys())
                final_cat[match_ict]=tile_cluscat[tile][ict]
    cat_summary={}
    cat_summary["Total clusters"]=len(final_cat.keys())
    cat_summary["Catalogue"]=final_cat
    catname=gset.mmfset.paths["result_data"] + "full_sky_catalogue.dict"
    write_dict(catname,cat_summary)
    return cat_summary

def return_cluster_cat(data,err,theta500,cluscat,projop,tmplt,snrthr=4,verbose=False):
    snrthrmask=np.ones_like(data)
    Ycyl2YsphR500=szp.convert_Ycyl_xR500_Ysph_xR500(xcyl=tmplt.cutoff,xsph=1.)
    Ycyl2Ysph5R500=szp.convert_Ycyl_xR500_Ysph_xR500(xcyl=tmplt.cutoff,xsph=5.)
    while (max((data*snrthrmask/err).ravel())>snrthr):
        max_snr=max((data*snrthrmask/err).ravel())
        temp_coord=np.where(data*snrthrmask/err == max_snr)
        x=temp_coord[0][0] ; y=temp_coord[1][0]
        glon,glat=projop.ij2ang(x,y)

        det_cluster=collections.OrderedDict()
        det_cluster["yc"]=[data[x,y]]
        det_cluster["err_yc"]=[err]
        det_cluster["snr"]=[data[x,y]/err]
        det_cluster["thetac"]=[theta500]
        det_cluster["cart_coord"]=[(x,y)]
        det_cluster["gal_coord"]=[(glon,glat)]
        det_cluster["YR500"]=[data[x,y]*tmplt.fn_yerr_norm(theta500)*Ycyl2YsphR500]
        det_cluster["err_YR500"]=[err*tmplt.fn_yerr_norm(theta500)*Ycyl2YsphR500]
        det_cluster["Y5R500"]=[data[x,y]*tmplt.fn_yerr_norm(theta500)*Ycyl2Ysph5R500]
        det_cluster["err_Y5R500"]=[err*tmplt.fn_yerr_norm(theta500)*Ycyl2Ysph5R500]

        nc=[]
        for ict in cluscat.keys():
            dist=return_distance(det_cluster["gal_coord"][0],cluscat[ict]["mp_gal_coord"])
            #dist=return_cartesian_distance(det_cluster["cart_coord"],cluscat[ict]["cart_coord"])
            nc=nc + [dist<theta500 or dist<cluscat[ict]["mp_thetac"]]

        if any(nc):
            myprint("This cluster exists in the catalogue",verbose)
            match_ict=np.where(nc)[0][0]
            for key in det_cluster.keys():
                cluscat[match_ict][key]=cluscat[match_ict][key] + det_cluster[key]
            if det_cluster["snr"][0]>cluscat[match_ict]["mp_snr"]:
                myprint("Updating the cluster definition",verbose)
                for key in det_cluster.keys():
                    cluscat[match_ict]["mp_" + key]=det_cluster[key][0]
        else:
            myprint("New cluster detected, adding to the cluster catalogue",verbose)
            match_ict=len(cluscat.keys())
            cluscat[match_ict]=det_cluster
            for key in det_cluster.keys():
                cluscat[match_ict]["mp_" + key]=det_cluster[key][0]
        # Here you can define the radius as the size of the cluster of points and that will make it more robust.
        snrthrmask=snrthrmask*gen_peak_mask(det_cluster["cart_coord"][0],max(5.*det_cluster["thetac"][0],15.))
        snrthrmask=snrthrmask*return_cluster_mask(det_cluster["cart_coord"][0],data/err,snrthr=4.)
    return cluscat

# The functions below do not depend on the location of the tile.
def return_cluster_mask(cart_coord,snrmap,snrthr=4,frac=0.7):
    '''
    frac: Number of pixels you want to be above the snr thresholds
    '''
    x,y=cart_coord
    pmask=np.zeros_like(snrmap)
    pmask[x,y]=1.
    step=0
    #     while all(snrmap[pmask==1]>=snrthr):
    while sum((snrmap[pmask==1]>=snrthr)*1.)/len((snrmap[pmask==1]>=snrthr)) > frac:
        step=step+1
        pmask=1-gen_peak_mask(cart_coord,step*gset.mmfset.reso)
    radius=(step-1)*gset.mmfset.reso
    return gen_peak_mask(cart_coord,radius)

def gen_peak_mask(cart_coord,radius):
    ix,iy=cart_coord
    tmask=np.ones((gset.mmfset.npix,gset.mmfset.npix),np.float64)
    distance=np.zeros((gset.mmfset.npix,gset.mmfset.npix),np.float64)
    x,y=np.indices((distance.shape))
    distance=np.sqrt((x-ix)**2. +(y-iy)**2.)*gset.mmfset.reso
    tmask[distance<=radius]=0
    return tmask

def return_distance(gal_coord1,gal_coord2):
    glon1,glat1=gal_coord1
    glon2,glat2=gal_coord2
    theta1=(90.-glat1)*np.pi/180.
    theta2=(90.-glat2)*np.pi/180.
    phi1=glon1*np.pi/180.
    phi2=glon2*np.pi/180.
    cosbeta=np.sin(theta1)*np.sin(theta2)*np.cos(phi2-phi1)+np.cos(theta1)*np.cos(theta2)
    beta=np.arccos(cosbeta)*180.*60./np.pi
    return beta

def gen_tile_figs(data,theta500,err,cluscat,filename,showplt=False):
	plt.ioff()
	if showplt:
		plt.ion()

	ang_dist=gset.mmfset.npix*gset.mmfset.reso/2./60.
	extent=[-ang_dist,ang_dist,-ang_dist,ang_dist]
	fig, ax1 = plt.subplots(ncols=1)
	snrthrmask=np.ones_like(data)
	for ict in cluscat.keys():
		x,y=cluscat[ict]["mp_cart_coord"]
		x=x*gset.mmfset.reso/60. - ang_dist
		y=y*gset.mmfset.reso/60. - ang_dist
		snr=cluscat[ict]["mp_snr"]
		radius=(snr)*1.5*gset.mmfset.reso/60.
		circ = Circle((y,x),radius,edgecolor='red', facecolor="none",alpha=0.85,linewidth=0.9)
		ax1.add_patch(circ)
	img1 = ax1.imshow(data/err,vmin=-2,vmax=10.,origin="lower",cmap="cividis",extent=extent)
#	img1 = ax1.imshow(data/err,vmin=-2,vmax=10.,origin="lower",cmap="viridis",extent=extent)
	colorbar(img1)
	ax1.set_title("Filtered data SNR [ $\sigma=$"+ str(round(err*1e5,3)) + r" ; $\theta_{500}=$" + str(round(theta500)) + "]",fontsize=8)
	plt.savefig(filename,bbox_inches="tight",dpi=150)
	plt.clf()
	plt.close()

def return_cartesian_distance(cart_coord1,cart_coord2):
    x1,y1=cart_coord1
    x2,y2=cart_coord2
    beta=np.sqrt((x1-x2)**2. + (y1-y2)**2)*gset.mmfset.reso
    return beta

def return_mf_cluster_model(cluscat):
    cmodel=np.zeros((gset.mmfset.npix,gset.mmfset.npix),float)
    multi_freq_cluster_model=np.zeros(data.shape,float)
    for ict in cluscat.keys():
        glon,glat=cluscat[ict]["mp_gal_coord"]
        x,y=projop.ang2ij(glon,glat)
        theta500=cluscat[ict]["mp_thetac"]
        yc=cluscat[ict]["mp_yc"]
        cmodel=cmodel + sc.gen_field_cluster_template(x,y,theta500,yc,gset.mmfset.npix,gset.mmfset.reso)
    cmodel_ft=fsa.map2alm(cmodel,gset.mmfset.reso)
    for i, ch in enumerate(gset.mmfset.channels):
        multi_freq_cluster_model[i,]=fsa.alm2map(cmodel_ft*op.chfiltr[ch]*op.sz_spec_bank[0][ch],gset.mmfset.reso)
    return multi_freq_cluster_model

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def myprint(text,verbose):
    if verbose:
        print text

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
