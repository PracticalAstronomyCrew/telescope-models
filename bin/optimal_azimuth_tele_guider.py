import argparse
import numpy as np

from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed, dump, load

parser = argparse.ArgumentParser(
            allow_abbrev=True, 
            description='Produce a grid of the obstruction of the main aperture/finder/guider by the dome for all possible HAs, Decs, and dome azimuths'
        )

parser.add_argument('-a', '--aperture', action='store', type=str, default='telescope', help='select aperture: telescope, finder, guider | default: telescope')

args = parser.parse_args()

# Constants for generating/saving the data
APERTURE_NAME = args.aperture
LOAD_DATE_SIGNATURE = '14_Jun_2021'
STORE_DATE_SIGNATURE = datetime.now().strftime('%d_%h_%Y')
OBSTR_DATA_FILE_1 = Path.cwd().parent / 'notebooks' / 'analysis' / 'data' / 'obstruction_cube_{}_{}.npy'.format('telescope', LOAD_DATE_SIGNATURE)
OBSTR_DATA_FILE_2 = Path.cwd().parent / 'notebooks' / 'analysis' / 'data' / 'obstruction_cube_{}_{}.npy'.format('guider', LOAD_DATE_SIGNATURE)
OPT_DATA_FILE = Path.cwd() / 'data' / 'optimal_azimuth_{}_{}.csv'.format('telescope_guider', STORE_DATE_SIGNATURE)

obstruction_data_tele = None
obstruction_data_guider = None

with OBSTR_DATA_FILE_2.open('rb') as f:
    obstruction_data_tele = np.load(f)

with OBSTR_DATA_FILE_1.open('rb') as f:
    obstruction_data_guider = np.load(f)

# Define a fine grid for the HA, Dec, and dome Az
_az = np.linspace(0, 359, 360)
_ha = np.linspace(-179, 180, 360) 
_dec = np.linspace(-90, 90, 181)

az, ha, dec = np.meshgrid(_az, _ha, _dec, indexing='ij')

# Extract data where there is no obstruction by the dome
cond = (obstruction_data_tele == obstruction_data_tele.min())&(obstruction_data_guider < 0.5)

ha_zero = ha[cond]
ha_zero = np.where(ha_zero < 0, ha_zero + 360, ha_zero) # shift the HA to be from 0h to 24h

dec_zero = dec[cond]
az_zero = az[cond]

# Define the ranges of HA and Dec corresponding to 0% obstruction
ha_range = np.arange(ha_zero.min(), ha_zero.max() + 1, 1)
dec_range = np.arange(dec_zero.min(), dec_zero.max() + 1, 1)

# -------------------------- #
#        BIG BOY STUFF       # 
# -------------------------- #

# def ha_dist(ha_array, ha_0):
#     try:
#         start = np.argwhere(np.isclose(ha_array, ha_0)).ravel()[0]
#     except IndexError:
#         return -1
    
#     diff = np.diff(ha_array)
#     diff_shifted = diff[start:]
    
#     if not diff_shifted.size:
#         return 0

#     if np.all(np.isclose(diff_shifted, 1)):
#         return diff_shifted.size

#     breaks = np.argwhere(~np.isclose(diff_shifted, 1)).ravel()
#     ha_dist = breaks[0]

#     if np.isclose(diff_shifted[ha_dist], -359):
#         if breaks.size > 1:
#             return diff_shifted[:breaks[1]].size
        
#         return diff_shifted.size

#     return ha_dist

def ha_dist(ha_array, ha_0):
    try:
        # Verify that the initial HA is indeed an option in ha_array
        start = np.argwhere(np.isclose(ha_array, ha_0)).ravel()[0]
    except IndexError:
        return -1

    ha_shifted = (ha_array - ha_0) % 360 # Set ha_0 to be 0
    ha_shifted.sort()
    
    idx = np.argwhere(~np.isclose(np.diff(ha_shifted), 1)).ravel()

    if idx.size > 0:
        return ha_shifted[:idx[0]].size
    
    return ha_shifted.size

def optimal_az(az_options, ha, dec):
    dec_sel = np.argwhere(np.isclose(dec_zero, dec)).ravel()
    
    azimuths = []
    delta_hs = []
    
    for az in az_options:
        az_sel = np.argwhere(np.isclose(az_zero, az)).ravel()
        
        indices = np.intersect1d(az_sel, dec_sel)
    
        # Extract the range of possible HAs
        hs = ha_zero[indices]
        
        # print('possible HAs:', hs)
        
        if hs.size:
            dh = ha_dist(hs, ha)
            
            azimuths.append(az)
            delta_hs.append(dh)
    
    azimuths = np.array(azimuths)
    delta_hs = np.array(delta_hs)
    
    return azimuths[delta_hs.argmax()], delta_hs.max()

# ------------------#
#  Parallelisation  #
# ------------------#

def file_name(h):
    return 'data/{}/azimuth/ha_{:.0f}.joblib'.format('telescope_guider', h)

def gen_grid(h):
    optimal_decs = []
    optimal_azimuths = []
    optimal_ha_dist = []

    for d in dec_range:
        # Select indices according to the given ha/dec
        dec_sel = np.argwhere(np.isclose(dec_zero, d)).ravel()
        ha_sel   = np.argwhere(np.isclose(ha_zero, h)).ravel()

        indices = np.intersect1d(ha_sel, dec_sel)

        # Find the range of azimuth values
        az = az_zero[indices]

        # Display the results
        if az.size > 0:
            az_opt, ha_dist = optimal_az(az, h, d)

            optimal_decs.append(d)
            optimal_azimuths.append(az_opt)
            optimal_ha_dist.append(ha_dist)


    opt_data = np.column_stack([optimal_decs, optimal_azimuths, optimal_ha_dist])
    
    with open(file_name(h), 'wb') as f:
        dump(opt_data, f)

    print('finished ha = {:>3.0f} degrees at {}'.format(h, datetime.now().strftime('%H:%M')))

def stitch_together():
    opt_data = np.empty((1, 4))

    for ha in ha_range:
        with open(file_name(ha), 'rb') as f:
            data = load(f)

            h_col = ha * np.ones((data.shape[0], 1))
            res = np.hstack([h_col, data])
            opt_data = np.vstack([opt_data, res])
    
    np.savetxt(str(OPT_DATA_FILE), opt_data, delimiter=',')

    print('finished generating optimal azimuth grid; data stored at {}...'.format(str(OPT_DATA_FILE.name)))

if __name__ == '__main__':
    print('start [{}] run at {:}'.format('telescope + guider', datetime.now().strftime('%H:%M')))

    results = Parallel(n_jobs=-1)(delayed(gen_grid)(h) for h in ha_range)

    print('finish [{}] run at {:}'.format('telescope + guider', datetime.now().strftime('%H:%M')))

    stitch_together()