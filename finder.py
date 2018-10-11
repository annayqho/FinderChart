""" Finder chart script, using ztfquery """

import numpy as np
import pyfits
import matplotlib.pyplot as plt
import pandas as pd
import astropy.wcs
from scipy.ndimage.filters import gaussian_filter
from astropy.time import Time
from ztfquery import query,marshal


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
    m.download_lightcurve(name) # dirout option doesn't seem to work?
    out_dir = "Data/marshal/lightcurves/%s/" %name
    lc_dict = pd.read_csv(out_dir + "marshal_lightcurve_%s.csv" %name)


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


# Name of target
name = "ZTF18abkmbpy"

# Position
ra = 256.309518
dec = 56.216698

# Get metadata of all images at this location
zquery = query.ZTFQuery()
zquery.load_metadata(
        radec=[ra,dec], size=0.01,
        auth=['ah@astro.caltech.edu', 'd1ewdw11z_IRSA'])
out = zquery.metatable

# Do you need to use a reference image?
need_ref = len(out) == 0

# If there are none, then use a reference image
if need_ref:
    zquery.load_metadata(kind="ref",radec=[ra, dec], size=0.0001)
    out = zquery.metatable
    zquery.download_data()

# Count the number of detections where limmag > 19.5
# If 0, allow limmag > 19
limmag = out.maglimit.values
limmag_val = 19.5 # must be deeper than this value
choose = limmag > limmag_val
while sum(choose) == 0:
    limmag_val += 0.5
    choose = limmag > limmag_val
im_ind = np.argmax(limmag[choose])
# Need a way to download just one of them...
# out[choose][im_ind] # index of the image you choose

# For now, just choose the file
ref_file = "Data/ref/000/field000796/zr/ccd11/q1/ztf_000796_zr_c11_q1_refimg.fits"

# get the cutout
inputf = pyfits.open(ref_file)
im = inputf[0].data
head = inputf[0].header
inputf.close()

# Get the x and y position of the target
wcs = astropy.wcs.WCS(head)
target_pix = wcs.wcs_world2pix([(np.array([ra,dec], np.float_))], 1)[0]
xpos = target_pix[0]
ypos = target_pix[1]

# plot the finder chart (code from Nadia)

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
        np.fliplr(smoothedimage[int(ymin):int(ymax),int(xmin):int(xmax)]), 
        origin='upper',
        vmin=np.percentile(im.flatten(), 10),
        vmax=np.percentile(im.flatten(), 99.0))

# Mark target: should just be the center of the image, now
plt.plot([300+20,300+10],[300,300], 'g-', lw=2)
plt.plot([300,300],[300+10,300+20], 'g-', lw=2)


# Set size of window (leaving space to right for ref star coords)
plt.subplots_adjust(right=0.65,left=0.05, top=0.99, bottom=0.05)

plt.show()
