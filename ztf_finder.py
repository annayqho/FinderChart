""" Finder chart script, using ztfquery """

import numpy as np
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import astropy.wcs
from astropy.io import fits
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


def hour2deg(ra, dec):
    '''
    Transforms string HH:MM:SS DD:MM:SS coordinates into degrees (floats).
    '''
    try:
        ra = float(ra)
        dec = float(dec)
        
    except:
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
        
        ra = c.ra.deg
        dec = c.dec.deg
    
    return ra, dec


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
    m = marshal.MarshalAccess()
    m.load_target_sources()
    coords = m.get_target_coordinates(name)
    ra = coords.ra.values[0]
    dec = coords.dec.values[0]
    return ra, dec


def get_lc(name):
    """ Get light curve of target """
    m = marshal.MarshalAccess()
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


def choose_ref(zquery, ra, dec):
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
    # If no files are returned,
    if len(out) == 0:
        print("Error: couldn't find any reference at this position.")
    else:
        # choose the index of the file with the deepest maglimit
        ind = out['maglimit'].idxmax()
        ind = 1
        urls, dl_loc = zquery.download_data(nodl=True)
        imfile = dl_loc[ind]
        # Temp bug fix: check to make sure the URL is correct
        imfile_array = imfile.split("/")
        imfile_array[3] = imfile_array[4][5:8]
        imfile = '/'.join(imfile_array)
        # default is refimg
        download_single_url(urls[ind], dl_loc[ind], cookies=None) 
        # download the associated PSFcat
        urls, dl_loc = zquery.download_data(nodl=True, suffix='refpsfcat.fits')
        catfile = dl_loc[ind]
        download_single_url(
                urls[ind], dl_loc[ind], cookies=None)
        return imfile, catfile


def choose_sci(zquery, out, name, ra, dec):
    """ Choose a science image to use, and download the file """
    lc = get_lc(name)
    # Count the number of detections where limmag > 19.5
    # If 0, allow limmag > 19
    limmag = lc.limmag.values
    limmag_val = 19.5 # must be deeper than this value
    choose = limmag > limmag_val
    while sum(choose) == 0:
        limmag_val += 0.5
        choose = limmag > limmag_val

    if sum(choose) > 1:
        # Of all these images, choose the one where the transient
        # is brightest
        ind = np.argmin(lc.magpsf.values[choose])
        jd_choose = lc['jdobs'][choose].values[ind] 
        mag_choose = lc['magpsf'][choose].values[ind]
        filt_choose = lc['filter'][choose].values[ind]
    elif sum(choose) == 1:
        # If there is only one choices...
        jd_choose = lc['jdobs'][choose].values[0]
        mag_choose = lc['magpsf'][choose].values[0]
        filt_choose = lc['filter'][choose].values[0]

    # Download the corresponding science image
    ind = np.argmin(np.abs(out.obsjd-jd_choose))
    urls, dl_loc = zquery.download_data(nodl=True)
    imfile = dl_loc[ind]
    download_single_url(urls[ind], dl_loc[ind], cookies=None)
    urls, dl_loc = zquery.download_data(nodl=True, suffix='psfcat.fits')
    catfile = dl_loc[ind]
    download_single_url(
            urls[ind], dl_loc[ind], cookies=None)
    return imfile, catfile


def get_finder(ra, dec, name, rad, debug=False, starlist=None, print_starlist=True, telescope="P200", directory=".", minmag=15, maxmag=18.5, mag=np.nan):
    """ Generate finder chart (Code modified from Nadia) """

    name = str(name)
    ra = float(ra)
    dec = float(dec)

    # Get metadata of all images at this location
    print("Querying for metadata...")
    zquery = query.ZTFQuery()
    zquery.load_metadata(
            radec=[ra,dec], size=0.01)
    out = zquery.metatable

    # Do you need to use a reference image?
    need_ref = len(out) == 0
    if need_ref:
        print("Using a reference image")
        imfile, catfile = choose_ref(zquery, ra, dec)
    else:
        print("Using a science image")
        imfile, catfile = choose_sci(zquery, out, name, ra, dec)

    # get the cutout
    inputf = pyfits.open(imfile)
    im = inputf[0].data
    inputf.close()
    head = fits.getheader(imfile)

    # Get the x and y position of the target,
    # as per the IPAC catalog
    wcs = astropy.wcs.WCS(head)
    target_pix = wcs.wcs_world2pix([(np.array([ra,dec], np.float_))], 1)[0]
    xpos = target_pix[0]
    ypos = target_pix[1]

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
    # pad the image
    im_padded = np.pad(smoothedimage, 300, mode='constant', constant_values=0)

    # If it's a reference image, you have to flip it up/down and left/right
    if need_ref:
        croppedimage = np.fliplr(np.flipud(
            im_padded[int(ymin)+300:int(ymax)+300,
                int(xmin)+300:int(xmax)+300]))

    # If it's a science image, you just flip it up/down
    else:
        croppedimage = np.flipud(
            im_padded[int(ymin)+300:int(ymax)+300,
                int(xmin)+300:int(xmax)+300])

    plt.imshow(
            croppedimage, origin='lower', # convention for IPAC images
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
    mag_crit = np.logical_and(cat['mag']+zp >= minmag, cat['mag']+zp <= maxmag)
    choose_ind = np.where(np.logical_and(crit, mag_crit))

    # mark the closest three stars
    nref = 3
    order = np.argsort(sep_pix[choose_ind])
    cols = ['orange', 'purple', 'red']

    # prepare to print starlist
    if telescope == "Keck":
        commentchar = "#"
        separator = ""
    else:
        commentchar = "!"
        separator = "!"

    for ii in np.arange(nref):
        ref_xpos_original = cat['xpos'][choose_ind][order][ii] - offset_x
        ref_ypos_original = cat['ypos'][choose_ind][order][ii] - offset_y

        # transform to flipped plot
        if need_ref:
            ref_xpos = 600-ref_xpos_original
            ref_ypos = 600-ref_ypos_original
        else:
            ref_xpos = ref_xpos_original
            ref_ypos = 600-ref_ypos_original

        plt.plot(
                [ref_xpos+5,ref_xpos+20],[ref_ypos,ref_ypos], 
                c=cols[ii], ls='-', lw=2)
        plt.plot(
                [ref_xpos,ref_xpos],[ref_ypos+5,ref_ypos+20], 
                c = cols[ii], ls='-', lw=2) 
        refra = cat['ra'][choose_ind][order][ii]
        refdec = cat['dec'][choose_ind][order][ii]
        if telescope == 'Keck':
            refrah, refdech = deg2hour(refra, refdec,sep=" ")
        elif telescope == 'P200':
            refrah, refdech = deg2hour(refra, refdec,sep=":")
        else:
            print("I don't recognize this telescope")
        refmag = cat['mag'][choose_ind][order][ii]+zp
        dra, ddec = get_offset(refra, refdec, ra, dec)

        offsetnum = 0.2
        plt.text(
                1.02, 0.60-offsetnum*ii, 
                'Ref S%s, mag %s' %((ii+1), np.round(refmag,1)), 
                transform=plt.axes().transAxes, fontweight='bold', color=cols[ii])
        plt.text(
                1.02, 0.55-offsetnum*ii, 
                '%s %s' %(refrah,refdech),
                color=cols[ii], transform=plt.axes().transAxes)
        plt.text(
                1.02, 0.50-offsetnum*ii, 
                str(np.round(ddec,2)) + "'' N, " + str(np.round(dra,2)) + "'' E", 
                color=cols[ii], transform=plt.axes().transAxes)

        # Print starlist for telescope
        if telescope == 'Keck':
            # Target name is columns 1-16
            # RA must begin in 17, separated by spaces
            print ("{:s}{:s} {:s} 2000.0 {:s} raoffset={:.2f} decoffset={:.2f} {:s} r={:.1f} ".format((name+"_S%s" %(ii+1)).ljust(16), refrah, refdech, separator, dra, ddec, commentchar, refmag))
        elif telescope == 'P200':
            print ("{:s} {:s} {:s}  2000.0 {:s} raoffset={:.2f} decoffset={:.2f} r={:.1f} {:s} ".format((name+"_S%s" %(ii+1)).ljust(20), refrah, refdech, separator, dra, ddec, refmag, commentchar))
        else:
            print("I don't recognize this telescope.")

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

    # List name, coords, mag of the target
    plt.text(1.02, 0.85, name, transform=plt.axes().transAxes, fontweight='bold')
    # Can't print mag, because we don't know how bright the target is
    #plt.text(1.02, 0.80, "%s"%mag, transform=plt.axes().transAxes, fontweight='bold')
    plt.text(1.02, 0.80, "%.5f %.5f"%(ra, dec),transform=plt.axes().transAxes)
    rah, dech = deg2hour(ra, dec)
    plt.text(1.02, 0.75,rah+"  "+dech, transform=plt.axes().transAxes)

    # Print the starlist
    if telescope == "Keck":
        commentchar = "#"
        separator = ""
    else:
        commentchar = "!"
        separator = "!"
        
    #Write to the starlist if the name of the starlist was provided.
    if (not starlist is None) and (telescope =="Keck"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2}  2000.0 # {3} \n".format(name.ljust(17), r, d, target_mag) ) 
            if (len(catalog)>0):
                f.write ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"_S1").ljust(17), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"_S2").ljust(17), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')

    if (not starlist is None) and (telescope =="P200"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2}  2000.0 ! {3}\n".format(name.ljust(19), r, d, target_mag) )
            if (len(catalog)>0):
                f.write ( "{:s} {:s} {:s}  2000.0 ! raoffset={:.2f} decoffset={:.2f} r={:.1f}  \n".format( (name+"_S1").ljust(19), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s} {:s}  2000.0 ! raoffset={:.2f} decoffset={:.2f} r={:.1f}  \n".format( (name+"_S2").ljust(19), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')     

    plt.savefig("finder_chart_%s.png" %name)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
        '''
        Creates the finder chart for the given RA, DEC and NAME.
        
        Usage: ztf_finder.py <RA [deg]> <Dec [deg]> <Name> <rad [deg]> <telescope [P200|Keck]>
            
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
        
    print ("Usage: ztf_finder.py <RA> <Dec> <Name>  <rad [deg]> <telescope [P200|Keck]>")
    
    #Check if correct number of arguments are given
    if len(sys.argv) < 4:
        print ("Not enough parameters given. \
                Please, provide at least: finder_chart.py <RA> <Dec> <Name>")
        sys.exit()
     
    ra=sys.argv[1]
    dec=sys.argv[2]
    name=str(sys.argv[3])
    if (len(sys.argv)>=5):
        rad = float(sys.argv[4])
        if (rad > 15./60):
            print ('Requested search radius of %.2f arcmin is larger than 15 arcmin.\
                    Not sure why you need such a large finder chart... \
                    reducing to 10 armin for smoother operations...'%(rad * 60))
            rad = 10./60
    else:
        rad = 2./60
        
    print ('Using search radius of %.1f arcsec.'%(rad*3600))

    if (len(sys.argv)>5):
        telescope = sys.argv[5]
    else:
        telescope = "P200"
        print ('Assuming that the telescope you observe will be P200. \
                If it is "Keck", please specify otherwise.')
    
    get_finder(ra, dec, name, rad, telescope=telescope, debug=False, minmag=7, maxmag=18)
