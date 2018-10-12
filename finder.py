""" Finder chart script, using ztfquery """

import numpy as np
import pyfits
import matplotlib.pyplot as plt
import pandas as pd
import astropy.wcs
from scipy.ndimage.filters import gaussian_filter
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from ztfquery import query,marshal
from ztfquery.io import download_single_url
from ztfquery.io import get_cookie


def deg2hour(ra, dec, sep=":"):
    '''
    Transforms the coordinates in degrees into HH:MM:SS DD:MM:SS with the requested separator.
    '''
    if ( type(ra) is str and type(dec) is str ):
        return ra, dec
    c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    ra = c.ra.to_string(unit=u.hourangle, sep=sep, precision=2, pad=True)
    dec = c.dec.to_string(sep=sep, precision=2, alwayssign=True, pad=True)
    return str(ra), str(dec)


def get_offset(ra1, dec1, ra2, dec2):
    '''
    Code from Nadia
    Computes the offset in arcsec between two coordinates.
    The offset is from (ra1, dec1) which is the offset star
    to ra2, dec2 which is the fainter target
    '''
    bright_star = SkyCoord(ra1, dec1, frame='icrs', unit=(u.deg, u.deg))
    target = SkyCoord(ra2, dec2, frame='icrs', unit=(u.deg, u.deg))
    dra, ddec = bright_star.spherical_offsets_to(target)
    return dra.to(u.arcsec).value, ddec.to(u.arcsec).value 


def get_pos(name):
    """
    Get position of target, if you need to
    This is very slow, so it's probably better
    to supply an RA and DEC
    """
    m = marshal.MarshalAccess(auth=['annayqho', 'd1ewdw11z_growth'])
    m.load_target_sources()
    coords = m.get_target_coordinates(name)
    ra = coords.ra.values[0]
    dec = coords.dec.values[0]
    return ra, dec


def get_lc(name):
    """ Get light curve of target """
    m.download_lightcurve(name) 
    lc = marshal.get_local_lightcurves(name)
    lc_dict = [lc[key] for key in lc.keys()][0]
    return lc_dict


def get_refstars(xpos, ypos, cat):
    """
    Select reference stars.
    
    Parameters
    ----------
    xpos: x position of target
    ypos: y position of target
    cat: ZTF source catalog
    """
    sep_pix = np.sqrt(
            (sourceinfo['xpos']-allsourcesinfo['xpos'])**2 + \
            (sourceinfo['ypos']-allsourcesinfo['ypos'])**2)
    
    # should be separated by at least 10 pixels
    crit_a = np.logical_and(sep_pix > 10, allsourcesinfo['flags']==0)
    crit_b = np.logical_and(
            allsourcesinfo['chi'] < 2, allsourcesinfo['snr'] > 10)
    crit_c = allsourcesinfo['sharp'] < 0.3
    crit_ab = np.logical_and(crit_a, crit_b)
    crit = np.logical_and(crit_ab, crit_c)
    
    # should be bright
    mag_crit = np.logical_and(
            allsourcesinfo['mag'] >= 15, allsourcesinfo['mag'] <= 19)
    choose_ind = np.where(np.logical_and(crit, mag_crit))
    
    # choose the closest three stars
    nref = 3
    order = np.argsort(sep_pix[choose_ind])
    
    # hack for now
    filt = 'R'
    colors = ['red', 'orange', 'purple']
    shapes = ['box', 'hexagon', 'circle']
    
    refstars = []
    for i in range(0,nref):
        refstars.append({
                          'name':  'S%s' %i,
                          'color':  colors[i],
                          'shape':  shapes[i],
                          'dist':  sep_pix[choose_ind][order][i] * 1.0, # pixel scale
                          'x_sub':  allsourcesinfo['xpos'][choose_ind][order][i],
                          'y_sub':  allsourcesinfo['ypos'][choose_ind][order][i],
                          'ra':  allsourcesinfo['ra'][choose_ind][order][i],
                          'dec':  allsourcesinfo['dec'][choose_ind][order][i],
                          'mag': allsourcesinfo['mag'][choose_ind][order][i],
                          'mag_err': allsourcesinfo['emag'][choose_ind][order][i],
                          'filter': filt
                        })
    return refstars


def choose_ref(ra, dec):
    """ Choose a reference image to use, and download the file
    including the associated PSF cat

    Parameters
    ----------
    ra: position of source in decimal degrees
    dec: position of source in decimal degrees

    Returns
    -------
    the filename of what you just downloaded (IPAC link)
    the location of the file (local link)
    """
    zquery.load_metadata(kind="ref",radec=[ra, dec], size=0.0001)
    out = zquery.metatable
    # choose the index of the file with the deepest maglimit
    ind = np.argmax(out['maglimit'])
    urls, dl_loc = zquery.download_data(nodl=True)
    imfile = dl_loc[ind]
    # default is refimg
    download_single_url(urls[ind], dl_loc[ind], cookies=None) 
    # download the associated PSFcat
    urls, dl_loc = zquery.download_data(nodl=True, suffix='refpsfcat.fits')
    catfile = dl_loc[ind]
    download_single_url(
            urls[ind], dl_loc[ind], cookies=None)
    return imfile, catfile


# Name of target
name = "ZTF18abkmbpy"

# Position
ra = 256.309518
dec = 56.216698

# Get metadata of all images at this location
print("Querying for metadata...")
zquery = query.ZTFQuery()
zquery.load_metadata(
        radec=[ra,dec], size=0.01,
        auth=['ah@astro.caltech.edu', 'd1ewdw11z_IRSA'])
out = zquery.metatable

# Do you need to use a reference image?
need_ref = len(out) == 0
if need_ref:
    print("Using a reference image")
    imfile, catfile = choose_ref(ra, dec)

# Count the number of detections where limmag > 19.5
# If 0, allow limmag > 19
# limmag = out.maglimit.values
# limmag_val = 19.5 # must be deeper than this value
# choose = limmag > limmag_val
# while sum(choose) == 0:
#     limmag_val += 0.5
#     choose = limmag > limmag_val
# im_ind = np.argmax(limmag[choose])
# Need a way to download just one of them...
# out[choose][im_ind] # index of the image you choose

# get the cutout
inputf = pyfits.open(imfile)
im = inputf[0].data
head = inputf[0].header
inputf.close()

# Get the x and y position of the target,
# as per the IPAC catalog
wcs = astropy.wcs.WCS(head)
target_pix = wcs.wcs_world2pix([(np.array([ra,dec], np.float_))], 1)[0]
xpos = target_pix[0]
ypos = target_pix[1]

# plot the finder chart (code from Nadia)

# aesthetics

# adjust counts
im[np.isnan(im)] = 0
im[im > 30000] = 30000

# extract 600x600 region around the position of the target
width = 600
height = 600
xmax = xpos + width/2
xmin = xpos - width/2
ymax = ypos + height/2
ymin = ypos - height/2

plt.figure(figsize=(8,6))
plt.set_cmap('gray_r')
smoothedimage = gaussian_filter(im, 1.3)
plt.imshow(
        smoothedimage[int(ymin):int(ymax),int(xmin):int(xmax)], 
        origin='lower', # convention for IPAC images
        vmin=np.percentile(im.flatten(), 10),
        vmax=np.percentile(im.flatten(), 99.0))

# Mark target: should just be the center of the image, now
# horizontal line
plt.plot([300+5,300+20],[300,300], 'g-', lw=2)
# vertical line
plt.plot([300,300],[300+5,300+20], 'g-', lw=2)
# and the offset of the original coordinate system with the new coordinates
offset_x = xpos-300
offset_y = ypos-300

# Choose offset stars
cat = pyfits.open(catfile)[1].data
zp = pyfits.open(catfile)[0].header['MAGZP']
sep_pix = np.sqrt(
        (xpos-cat['xpos'])**2 + \
        (ypos-cat['ypos'])**2)

# should be separated by at least 10 pixels
crit_a = np.logical_and(sep_pix > 10, cat['flags']==0)
crit_b = np.logical_and(
        cat['chi'] < 2, cat['snr'] > 10)
crit_c = cat['sharp'] < 0.3
crit_ab = np.logical_and(crit_a, crit_b)
crit = np.logical_and(crit_ab, crit_c)

# should be bright
mag_crit = np.logical_and(cat['mag']+zp >= 15, cat['mag']+zp <= 19)
choose_ind = np.where(np.logical_and(crit, mag_crit))

# mark the closest three stars
nref = 3
order = np.argsort(sep_pix[choose_ind])
cols = ['orange', 'purple', 'red']

for ii in np.arange(nref):
    ref_xpos = cat['xpos'][choose_ind][order][ii] - offset_x
    ref_ypos = cat['ypos'][choose_ind][order][ii] - offset_y
    plt.plot(
            [ref_xpos+5,ref_xpos+20],[ref_ypos,ref_ypos], 
            c=cols[ii], ls='-', lw=2)
    plt.plot(
            [ref_xpos,ref_xpos],[ref_ypos+5,ref_ypos+20], 
            c = cols[ii], ls='-', lw=2) 
    refra = cat['ra'][choose_ind][order][ii]
    refdec = cat['dec'][choose_ind][order][ii]
    dra, ddec = get_offset(refra, refdec, ra, dec)

    plt.text(
            1.02, 0.60-0.1*ii, 'Ref %s' %ii, 
            transform=plt.axes().transAxes, fontweight='bold', color=cols[ii])
    # plt.text(
    #         1.02, 0.65-0.2*ii, "%.5f %.5f"%(refra, refdec),
    #         transform=plt.axes().transAxes, color=cols[ii])
    # plt.text(
    #         1.02, 0.60-0.2*ii,refrah+"  "+np.round(ddec,2), 
    #         transform=plt.axes().transAxes, color=cols[ii])
    plt.text(
            1.02, 0.55-0.1*ii, 
            str(np.round(ddec,2)) + "'' N, " + str(np.round(dra,2)) + "'' E", 
            color=cols[ii], transform=plt.axes().transAxes)


# Plot compass
plt.plot(
        [width-10,height-40], [10,10], 'k-', lw=2)
plt.plot(
        [width-10,height-10], [10,40], 'k-', lw=2)
plt.annotate(
    "N", xy=(width-20, 40), xycoords='data',
    xytext=(-4,5), textcoords='offset points')
plt.annotate(
    "E", xy=(height-40, 20), xycoords='data',
    xytext=(-12,-5), textcoords='offset points')


# Get rid of axis labels
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)

# Set size of window (leaving space to right for ref star coords)
plt.subplots_adjust(right=0.65,left=0.05, top=0.99, bottom=0.05)

# List the offsets in a table

# List name, coords, mag of reference
plt.text(1.02, 0.85, name, transform=plt.axes().transAxes, fontweight='bold')
plt.text(1.02, 0.80, "%.5f %.5f"%(ra, dec),transform=plt.axes().transAxes)
rah, dech = deg2hour(ra, dec)
plt.text(1.02, 0.75,rah+"  "+dech, transform=plt.axes().transAxes)

plt.show()
