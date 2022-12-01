# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:41:15 2021

@author: oryan

This script will utilise the coordinates provided by Galaxy Zoo: Mergers project in order to calculate where the secondary galaxy is and use those inputs as x and y.
"""
## Imports
import pandas as pd
import os
import sys

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

## Functions
def checker(x, y, neg, t):
    if neg == 'n' and t == 'n':
        return x,y
    elif neg == 'x' and t == 'n':
        return x,-y
    elif neg == 'y' and t == 'n':
        return -x,y
    elif neg == 'xy' and t == 'n':
        return -x, -y
    elif neg == 'n' and t == 'y':
        return y,x
    elif neg == 'x' and t == 'y':
        return -y,x
    elif neg == 'y' and t == 'y':
        return y, -x
    elif neg == 'xy' and t == 'y':
        return -y, -x
    else:
        print('WARNING: Negative and transpose flags are incorrect. Double check.')
        sys.exit()

## Main Function
class Secondary_Placer:
    
    def get_secondary_coords(gal_name, redshifts):
        cwd = os.getcwd() + '/PySPAM_Original_Python_MCMC'

        file_path = cwd + '/data/all_coords.csv'

        all_coords_tmp = pd.read_csv(file_path, index_col = 0)
        all_coords = (
            all_coords_tmp
            .assign(stripped_names = all_coords_tmp.Names.apply(lambda x: x.strip()))
            )

        curr_coords = all_coords.query('stripped_names == @gal_name')
        
        if len(curr_coords) == 0:
            return None, None, None, True

        row = redshifts.query('Names == @gal_name')
        z = row.Redshift.iloc[0]
        block_reduce = row.block_reduce.iloc[0]
        negative = row.neg.iloc[0]
        t = row.t.iloc[0]

        if negative == 'rereduce' or t == 'rereduce':
            return None, None, None, True

        prim_coords = SkyCoord(
            ra = curr_coords.Prim_RA.iloc[0] * u.deg,
            dec = curr_coords.Prim_DEC.iloc[0] * u.deg,
            frame = 'fk5'
        )

        sec_coords = SkyCoord(
            ra = curr_coords.Sec_RA.iloc[0] * u.deg,
            dec = curr_coords.Sec_DEC.iloc[0] * u.deg,
            frame = 'fk5'
        )

        dra, ddec = prim_coords.spherical_offsets_to(sec_coords)

        dra_m = dra.to(u.arcmin)
        ddec_m = ddec.to(u.arcmin)

        cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.275 * u.K, Om0 = 0.308)

        conversion = cosmo.kpc_proper_per_arcmin(z)
        
        phys_x = conversion * dra_m
        phys_y = conversion * ddec_m

        sim_x = float(phys_x / (15 * u.kpc))
        sim_y = float(phys_y / (15 * u.kpc))

        sim_x, sim_y = checker(sim_x, sim_y, negative, t)

        Resolution = float((0.396 * u.arcsec)  * conversion.to(u.kpc / u.arcsec) * block_reduce / (15 * u.kpc))
    
        return [sim_x, sim_y], Resolution, z, False