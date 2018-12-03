""" Take the Marshal output, and add offset stars """

import numpy as np
from astropy.coordinates import SkyCoord
from ztf_finder import get_finder

dat = np.loadtxt(
    "181203_starlist.txt", dtype=str)
names = dat[:,0]
hh = dat[:,1]
mm = dat[:,2]
ss = dat[:,3]
ra = np.array([hh[i]+"h"+mm[i]+"m"+ss[i]+"s" for i in np.arange(len(hh))])
dd = dat[:,4]
mm = dat[:,5]
ss = dat[:,6]
dec = np.array([dd[i]+"d"+mm[i]+"m"+ss[i]+"s" for i in np.arange(len(hh))])

c = SkyCoord(ra, dec, frame='icrs')
radeg = c.ra.deg
decdeg = c.dec.deg

# get_finder(
#         float(radeg), float(decdeg), str(name), 
#         rad=0.2, telescope='Keck', debug=False, minmag=7, maxmag=18)

for ii,name in enumerate(names[33:]):
    print(name)
    print(radeg[ii])
    print(decdeg[ii])
    get_finder(
            float(radeg[ii]), float(decdeg[ii]), str(name), 
            rad=0.2, telescope='Keck', debug=False, minmag=7, maxmag=18)
