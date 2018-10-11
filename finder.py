""" Finder chart script, using ztfquery """

import numpy as np
import pyfits
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from ztfquery import query,marshal

# Name of target
name = "ZTF18abkmbpy"

# Get position of target
m = marshal.MarshalAccess(auth=['annayqho', 'd1ewdw11z_growth'])
m.load_target_sources()
coords = m.get_target_coordinates(name)
ra = coords.ra.values[0]
dec = coords.dec.values[0]

# Get light curve of target
m.download_lightcurve(name) # dirout option doesn't seem to work?
out_dir = "Data/marshal/lightcurves/%s/" %name
lc_dict = pd.read_csv(out_dir + "marshal_lightcurve_%s.csv" %name)

# Get metadata of all images at this location
zquery = query.ZTFQuery()
zquery.load_metadata(
        radec=[ra,dec], size=0.01,
        auth=['ah@astro.caltech.edu', 'd1ewdw11z_IRSA'])
out = zquery.metatable

# Do you need to use a reference image?
need_ref = ndet == 0

# If there are none, then use a reference image
if need_ref:
    zquery.load_metadata(kind="ref",radec=[ra, dec], size=0.0001)
    out = zquery.metatable

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
inputf.close()
