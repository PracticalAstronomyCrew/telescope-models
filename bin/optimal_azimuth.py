import argparse
import numpy as np

from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed, dump, load


parser = argparse.ArgumentParser(
            allow_abbrev=True, 
            description='Produce a grid with optimal azimuth angles for the dome, given the appropriate (combination of) aperture(s).'
        )

parser.add_argument('-a', '--aperture', action='store', type=str, default='telescope', help='select aperture: telescope, finder, guider, telescope_guider | default: telescope')
parser.add_argument('-g', '--guider_requirement', action='store', type=float, default=0.5, help='ratio of how much of the guider should be unobstructed | default: 0.5')

args = parser.parse_args()


# LOAD_DATE_SIGNATURE = args.date
SRC = Path.cwd() / 'data'
OPT_TARGET = Path.cwd() / 'data' / 'optimal_azimuth_{}_{}.csv'.format(args.aperture, datetime.now().strftime('%d_%h_%Y'))


# Verify whether the appropriate file structure exists/otherwise create required folders
req_path = SRC / args.aperture / 'azimuth'

try:
    req_path.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    pass
else:
    print('Created the data/.../azimuth folder(s)...')


# Define a grid for the HA, Dec, and dome Az
_az = np.linspace(0, 359, 360)
_ha = np.linspace(0, 359, 360)
_dec = np.linspace(-90, 90, 181)

az, ha, dec = np.meshgrid(_az, _ha, _dec, indexing='ij')

ha_zero = None
dec_zero = None
az_zero = None

# Load the appropriate obstruction data set; generated w/ obstruction_grid.py
if args.aperture == 'telescope_guider':
    obstruction_data_tele = None
    obstruction_data_guider = None
    
    fn = input('Insert obstruction cube file name obstruction_cube_telescope_*.npy:')
    path = SRC / fn
    
    with path.open('rb') as obstr_file:
        obstruction_data_tele = np.load(obstr_file)
    
    fn = input('Insert obstruction cube file name obstruction_cube_guider_*.npy:')
    path = SRC / fn

    with path.open('rb') as obstr_file:
        obstruction_data_guider = np.load(obstr_file)

    # Extract the (HA, Dec) coordinates w/ 0% obstruction for the telescope + guider
    cond = (obstruction_data_tele == obstruction_data_tele.min())&(obstruction_data_guider < args.guider_requirement)

    ha_zero = ha[cond]
    dec_zero = dec[cond]
    az_zero = az[cond]

else:
    fn = input('Insert obstruction cube file name obstruction_cube_*_*.npy:')
    path = SRC / fn

    with path.open('rb') as obstr_file:
        obstruction_data = np.load(obstr_file)
    
    # Extract the (HA, Dec) coordinates w/ 0% obstruction for a single aperture
    cond = obstruction_data == obstruction_data.min()

    ha_zero = ha[cond]
    dec_zero = dec[cond]
    az_zero = az[cond]


# Define the ranges of HA and Dec values corresponding to 0% obstruction
ha_range = np.arange(ha_zero.min(), ha_zero.max() + 1, 1)
dec_range = np.arange(dec_zero.min(), dec_zero.max() + 1, 1)


def ha_dist(ha_array, ha_0):
    """Compute the time the dome could remain at a certain position.
    
    Parameters
    ----------
    ha_array: 1d array w/ all possible hour angles
    ha_0: initial hour angle
    """
    try:
        # Verify that the initial HA is indeed an option in ha_array
        start = np.argwhere(np.isclose(ha_array, ha_0)).ravel()[0]
    except IndexError:
        return -1

    # Shift the HA options s.t. ha_0 corresponds to 0 h
    ha_shifted = (ha_array - ha_0) % 360
    ha_shifted.sort()
    
    idx = np.argwhere(~np.isclose(np.diff(ha_shifted), 1)).ravel()

    if idx.size > 0:
        return ha_shifted[:idx[0]].size
    
    return ha_shifted.size


def optimal_az(az_options, ha, dec):
    """Compute the optimal dome azimuth angle.
    
    Parameters
    ----------
    az_options: all possible azimuth angles w/ 0% obstruction
    ha: the initial hour angle
    dec: the initial declination
    """
    dec_sel = np.argwhere(np.isclose(dec_zero, dec)).ravel()
    
    azimuths = []
    delta_hs = []
    
    for az in az_options:
        az_sel = np.argwhere(np.isclose(az_zero, az)).ravel()
        
        indices = np.intersect1d(az_sel, dec_sel)
    
        # Extract the range of possible HAs
        hs = ha_zero[indices]
        
        if hs.size:
            dh = ha_dist(hs, ha)
            
            azimuths.append(az)
            delta_hs.append(dh)
    
    azimuths = np.array(azimuths)
    delta_hs = np.array(delta_hs)
    
    return azimuths[delta_hs.argmax()], delta_hs.max()


def get_ha_path(h):
    fn = 'ha_{}.joblib'.format(int(h))

    return req_path / fn


def generate_azimuth_grid(h):
    """Store the optimal azimuth angle for a single hour angle.
    
    Parameters
    ----------
    h: the hour angle in degrees
    """
    optimal_decs = []
    optimal_azimuths = []
    optimal_ha_dist = []

    for d in dec_range:
        # Select indices according to the given ha/dec
        dec_sel = np.argwhere(np.isclose(dec_zero, d)).ravel()
        ha_sel = np.argwhere(np.isclose(ha_zero, h)).ravel()

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
    
    with get_ha_path(h).open('wb') as ha_file:
        dump(opt_data, ha_file)

    print('Finished ha = {:>3.0f} degrees at {}'.format(h, datetime.now().strftime('%H:%M')))

def combine():
    """
    Load the optimal azimuth grids (1 per hour angle) and stitch 
    them together and store them as -- easy-to-read -- csv files.
    """
    opt_data = np.empty((1, 4))

    for ha in ha_range:
        with get_ha_path(ha).open('rb') as ha_file:
            data = load(ha_file)

            h_col = ha * np.ones((data.shape[0], 1))
            res = np.hstack([h_col, data])
            opt_data = np.vstack([opt_data, res])
    
    np.savetxt(str(OPT_TARGET), opt_data[1:], delimiter=',')

    print('finished generating optimal azimuth grid; data stored at {}...'.format(str(OPT_TARGET.name)))

if __name__ == '__main__':
    print('start [{}] run at {:}'.format(args.aperture, datetime.now().strftime('%H:%M')))

    results = Parallel(n_jobs=-1)(delayed(generate_azimuth_grid)(h) for h in ha_range)

    print('finish [{}] run at {:}'.format(args.aperture, datetime.now().strftime('%H:%M')))

    combine()