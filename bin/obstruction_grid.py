import numpy as np
import argparse

from joblib import Parallel, delayed, dump, load
from datetime import datetime

from obstruction.aperture import TelescopeAperture, GuiderAperture, FinderAperture

parser = argparse.ArgumentParser(
            allow_abbrev=True, 
            description='Produce a grid of the obstruction of the main aperture/finder/guider by the dome for all possible HAs, Decs, and dome azimuths'
        )

parser.add_argument('-a', '--aperture', action='store', type=str, default='telescope', help='select aperture: telescope, finder, guider | default: telescope')
parser.add_argument('-r', '--rate', action='store', type=int, default=3, help='no. rays (for decent results >3; preferably 4-10) | default: 3')

args = parser.parse_args()

# sample HA from -12 h to 12 h; Dec from -90 to 90 deg
h = np.linspace(0, 359, 360)
dec = np.linspace(-90, 90, 181)

# Create a data file per possible azimuth!
az_range = np.arange(0, 360, 1)

# Define a global variable for the selected aperture
if args.aperture == 'guider':
    APERTURE = GuiderAperture(rate=args.rate)
elif args.aperture == 'finder':
    APERTURE = FinderAperture(rate=args.rate)
else:
    APERTURE = TelescopeAperture(rate=args.rate)

def file_name(n):
    fn = 'data/' + APERTURE.get_name() + '/blocked/az_' + str(n) + '.joblib'

    return fn

def gen_grid(n_az):
    p = np.zeros((h.size, dec.size))

    for i in range(h.size):
        for j in range(dec.size):
            percentage = APERTURE.obstruction(h[i], dec[j], n_az)

            p[i, j] = percentage

    print('finished az = {:>3.0f} degrees at {}'.format(n_az, datetime.now().strftime('%H:%M')))

    with open(file_name(n_az), 'wb') as f:
        dump(p, f)

def to_file():
    # Empty array to store the new 3D data set in
    obstr_cube = []

    for n_az in az_range:
        with open(file_name(n_az), 'rb') as f:
            data = load(f)
            obstr_cube.append(data)

    obstr_cube = np.array(obstr_cube)

    date_signature = datetime.now().strftime('%d_%h_%Y')
    file_path = 'data/obstruction_cube_{}_{}.npy'.format(args.aperture, date_signature)

    with open(file_path, 'wb') as f:
        np.save(f, obstr_cube)
    
    print('Result is stored in "{}"'.format(file_path))

if __name__ == '__main__':
    print('Start [{}] run at {:}'.format(APERTURE.get_name(), datetime.now().strftime('%H:%M')))
    results = Parallel(n_jobs=-1)(delayed(gen_grid)(i) for i in az_range)

    # Stich the loose joblib files together
    to_file()
