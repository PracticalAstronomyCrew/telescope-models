import argparse

from obstruction.aperture import TelescopeAperture, GuiderAperture, FinderAperture

parser = argparse.ArgumentParser(
            allow_abbrev=True, 
            description='Compute the % obstruction of the main aperture/finder/guider by the dome'
        )

parser.add_argument('--az', action='store', type=float, default=0.0, help='dome azimuth: 0 to 360 deg | default: 0 deg')
parser.add_argument('--ha', action='store', type=float, default=0.0, help='telescope hour angle: 0 to 24 h | default: 0 h')
parser.add_argument('--dec', action='store', type=float, default=0.0, help='telescope declination -90 to 90 deg | default: 0 deg')
parser.add_argument('-a', '--aperture', action='store', type=str, default='telescope', help='select aperture: telescope, finder, guider | default: telescope')
parser.add_argument('-r', '--rate', action='store', type=int, default=4, help='no. radial circles of rays (for decent results > 3; preferably 10+) | default: 4')
parser.add_argument('-v', '--visualise', default=False, action='store_true')

args = parser.parse_args()

if __name__ == '__main__':    
    blockage = None

    if args.aperture == 'telescope':
        telescope = TelescopeAperture(rate=args.rate)
        blockage = telescope.obstruction(args.ha*15, args.dec, args.az, plot_result=args.visualise)

    elif args.aperture == 'guider':
        guider = GuiderAperture(rate=args.rate)
        blockage = guider.obstruction(args.ha*15, args.dec, args.az, plot_result=args.visualise)

    elif args.aperture == 'finder':
        finder = FinderAperture(rate=args.rate)
        blockage = finder.obstruction(args.ha*15, args.dec, args.az, plot_result=args.visualise)

    if blockage is not None:
        print('Obstruction = {:.2%}'.format(blockage))
    else:
        print('ERROR:The % obstruction could not be computed!')