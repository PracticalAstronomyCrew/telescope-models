# Obstruction Modeling Source Code

This folder contains the code used to model the obstruction of the Gratama telescope aperture(s) by the dome. For a command utility refer to [MOCCA](https://github.com/mickveldhuis/mocca).

## What's here & how to use it?

This folder contains the `obstruction` package and the `obstruction_cube` and `optimal_azimuth` scripts! Additionally, there is a `requirements.txt` file with the relevant packages.

### The obstruction package

The `obstruction` package defines an interface to compute the % obstruction of an arbitrary aperture, in accordance with geometrical properties of the dome and telescope defined in the `config.ini` file in the `resources` folder.

The package contains an `Aperture` class in `aperture.py`, which can be inherited to define any aperture using the (right handed) coordinate transformations in `transformations.py`. See `aperture.py` and specifically the `TelescopeAperture`, `GuiderAperture`, and `FinderAperture` as examples.

### Optimal azimuth grid generation

As part of the new dome control system, see [dome-control](https://github.com/PracticalAstronomyCrew/dome-control), we require a grid of azimuth data given hour angle and declination coordinates. We generate this grid, sequentially, using two scripts: `obstruction_grid.py` and `optimal_azimuth.py`,

- `obstruction_grid.py` calculates the percentage obstruction for all possible hour angles, declinations, and dome azimuth angles on a 1 degree spaced grid, and finally writes the data to a `.npy` file.
- `optimal_azimuth.py` uses that data file to compute the optimal azimuth assuming the selected aperture (or combination of apertures) should always be fully unobstructed.
