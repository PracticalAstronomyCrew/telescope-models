import enum, configparser
import numpy as np
import matplotlib.pyplot as plt

from obstruction.transformations import vec3, vec4, transform, rot_x, rot_z
from pytransform3d import transformations as pt

# Load the geometry parameters of the telescope/dome
config = configparser.ConfigParser()
config.read('resources/config.ini')

# Telescope:
L_1 = config['mount'].getfloat('length_1') # distance floor-HA axis
L_2 = config['mount'].getfloat('length_2') # distance HA axis-Dec axis
L_3 = config['mount'].getfloat('length_3') # distance Dec axis-tube center
L_4 = config['guider'].getfloat('offset') # distance primary tube center-guider center
L_5 = config['finder'].getfloat('offset') # distance guider center-finder center

GUIDER_ANGLE = np.radians(config['guider'].getfloat('angle'))
FINDER_ANGLE =  np.radians(config['finder'].getfloat('angle'))

APERTURE_RADIUS = config['telescope'].getfloat('diameter')/2
APERTURE_SEC_RADIUS = config['telescope'].getfloat('sec_diameter')/2

GUIDER_RADIUS = config['guider'].getfloat('diameter')/2
GUIDER_SEC_RADIUS = config['guider'].getfloat('sec_diameter')/2

FINDER_RADIUS = config['finder'].getfloat('diameter')/2

# Dome:
RADIUS = config['dome'].getfloat('diameter')/2   # radius
EXTENT = config['dome'].getfloat('extent')  # extent of cylindrical dome wall
SLIT_WIDTH = config['dome'].getfloat('slit_width') # Slit width

# Observatory:
LAT = config['observatory'].getfloat('latitude') # degrees


class Instruments(enum.Enum):
    """Enum of possible apertures."""
    TELESCOPE = enum.auto()
    GUIDER = enum.auto()
    FINDER = enum.auto()

    @classmethod
    def get_default(c):
        return c.TELESCOPE


def plot_aperture(ap_x, ap_z, is_blocked, aperture_r, dome_az):
    """"Plot the aperture w/ obstructed sample points.
    
    Parameters
    ----------

    ap_x, ap_z: x, z coordinates of the sampled points
    is_blocked: boolean array of len(ap_x) signifying whether a point is obstructed
    aperture_r: radius of the aperture in meters
    dome_az: position of the dome (azimuth angle in deg)
    """
    percentage = is_blocked[is_blocked].size/is_blocked.size

    fig = plt.figure(figsize=(4.5, 4.5), num='MOCCA - Visualisation')
    frame = fig.add_subplot(1, 1, 1)

    frame.plot(ap_x[is_blocked], ap_z[is_blocked], ls='', marker='o', ms=3, color='xkcd:salmon', label='{:.1%} Blocked'.format(percentage))
    frame.plot(ap_x[~is_blocked], ap_z[~is_blocked], ls='', marker='o', ms=3, color='black', label='{:.1%} Clear'.format(1-percentage))

    frame.set_xlabel(r'$x$ (m)', fontsize=18)
    frame.set_ylabel(r'$z$ (m)', fontsize=18)
    frame.grid(ls='--', alpha=0.5)

    frame.set_ylim(-2*aperture_r, 2*aperture_r)
    frame.set_xlim(-2*aperture_r, 2*aperture_r)

    frame.set_title('Dome azimuth = {:.1f} deg'.format(float(dome_az) % 360), fontsize=18)

    frame.legend(fontsize=12, loc='lower right')
    fig.tight_layout()

    plt.show()


def find_intersection(point, direction):
    """Find ray-capsule (i.e. ray-dome) intersection.
    
    Parameters
    -----------

    point: ray origin (3-vector)
    direction: ray unit direction vector

    Returns
    -------

    has_intersection: boolean signifying whether there is an intersection
    t               : distance between the ray origin and intersection
    """
    has_intersection = False
    t = None
    
    # In case the ray is ~parallel to the dome z-axis
    if np.isclose(direction[0], 0) and np.isclose(direction[1], 0):
        z = EXTENT + np.sqrt(RADIUS**2 - point[0]**2 - point[1]**2)
        t = z - point[2]
        
        has_intersection = True
        return has_intersection, t
    
    # If the direction vector is not (nearly) parallel to the dome z-axis
    a2 = direction[0]**2 + direction[1]**2
    a1 = point[0]*direction[0] + point[1]*direction[1]
    a0 = point[0]**2 + point[1]**2 - RADIUS**2
    
    delta = a1**2-a0*a2
    t     = (-a1+np.sqrt(delta))/a2
    
    if point[2] + t * direction[2] >= EXTENT:
        a0 = point[0]**2 + point[1]**2 + (point[2] - EXTENT)**2 - RADIUS**2
        a1 = point[0]*direction[0] + point[1]*direction[1] + (point[2] - EXTENT)*direction[2]
        
        t = -a1+np.sqrt(a1**2-a0)
    
    if t:
        has_intersection = True
    
    return has_intersection, t


def get_ray_intersection(point, direction, t):
    """
    Return the ray intersection, based on the origin 
    (point) and direction vectors.
    """
    return point + t*direction


class Aperture:
    """Class representing the telescope aperture.
    
    Public methods
    --------------

    obstruction (float): return the % obstruction of the aperture by the dome
    get_name (str): return an aperture "name"/identifier
    """
    def __init__(self, radius, sec_radius=0, rate=3):
        """"The Aperture class constructor.
        
        Parameters
        ----------
        
        radius (float): aperture radius in meters
        sec_radius (float): radius of secondary obstruction in meters
        rate (int): aperture sample rate (in terms of no. radial circles around the center)
        """
        self.radius = radius 
        self.sec_radius = sec_radius
        self.sample_rate = rate

        self._name = None

        # Add a vectorized instance of the _is_ray_blocked function
        self._is_blocked = np.vectorize(self._is_ray_blocked, signature='(d),(),(),()->()')

    def _transform(self, ha, dec):
        """"Get the transformation matrix to the aperture.
        
        Parameters
        ----------

        ha (float): hour angle in degrees
        dec (float): declination in degrees
        """
        H_01 = transform(0, 0, L_1)
        H_12 = rot_x(90-LAT) @ rot_z(-ha) @ transform(0, 0, L_2)
        H_23 = rot_x(dec) @ transform(-L_3, 0, 0)

        H = H_01 @ H_12 @ H_23

        return H

    def _sample_disk(self, r_min=0):
        """
        Equidistant disk sampling based on:
        http://www.holoborodko.com/pavel/2015/07/23/generating-equidistant-points-on-unit-disk/

        Parameters
        ----------

        r_min (float): ratio of the circle that is obstructed at the center
        """
        if not 0 <= r_min < 1:
            raise ValueError('r_min should be between 0 and 1...')
        
        dr = 1/self.sample_rate
        
        x = np.empty(0)
        y = np.empty(0)
        
        rs = np.linspace(r_min, 1, self.sample_rate) 
        k = np.ceil(r_min*(self.sample_rate+1))
        
        if not r_min:
            x = np.concatenate([x, [0]])
            y = np.concatenate([y, [0]])
            
            rs = np.linspace(dr, 1, self.sample_rate)
            k = 1
        
        for r in rs:
            n = int(np.round(np.pi/np.arcsin(1/(2*k))))
            
            theta = np.linspace(0, 2*np.pi, n+1)
            
            x_r = r * np.cos(theta)
            y_r = r * np.sin(theta)
            
            x = np.concatenate([x, x_r])
            y = np.concatenate([y, y_r])
            
            k += 1
        
        xy = self.radius*np.column_stack([x,y])
        
        return xy
    
    def _sample_aperture(self, ha, dec, x, z):
        """
        Compute the position of a vector in 
        the aperture's frame.

        Parameters
        ----------

        ha (float): hour angle in degrees
        dec (float): declination in degrees
        x (float ndarray): x coordinate of a point in the aperture
        z (float ndarray): z coordinate of a point in the aperture
        """
        y = np.zeros(x.size)
        dummy = np.ones(x.size)
        points = np.column_stack((x, y, z, dummy))
        
        pose_matrix = self._transform(ha, dec)

        product = pt.transform(pose_matrix, points)

        return product[:, :3]
    
    def _aperture_direction(self, ha, dec):
        """
        Return the pointing direction of the aperture
        in the frame of the dome.

        Parameters
        ----------

        ha (float): hour angle in degrees
        dec (float): declination in degrees
        """
        H_ap = self._transform(ha, dec)
        H_unit = transform(0, 1, 0)

        H_diff = H_ap @ H_unit - H_ap

        direction = H_diff @ vec4(0, 0, 0)
        
        return vec3(direction)

    def _is_ray_blocked(self, point, ha, dec, dome_az):
        """
        Checks whether an individual ray is blocked.

        Parameters
        ----------

        point (3-vector): ray origin
        ha (float): hour angle in degrees
        dec (float): declination in degrees
        dome_az (float): dome azimuth (clockwise convention)
        """
        is_blocked = True
        
        direction = self._aperture_direction(ha, dec)

        try:
            has_intersection, t = find_intersection(point, direction)

            if has_intersection:
                points = get_ray_intersection(point, direction, t)
                
                az_corrected = (dome_az-180) % 360 # Correction assuming the azimuth is zero at the South
                rot = rot_z(az_corrected)

                dummy = np.ones(points[0].size)
                pp = np.column_stack((points[0], points[1], points[2], dummy))

                product = pt.transform(rot, pp)
                
                r = RADIUS * np.sin(np.radians(15))

                x_cond = -SLIT_WIDTH/2 < product[:, 0] < SLIT_WIDTH/2
                y_cond = -r < product[:, 1] < RADIUS

                is_ray_in_slit = points[2] > EXTENT and x_cond and y_cond

                if is_ray_in_slit:
                    is_blocked = False
                
        except Exception as ex:
            print('ERROR OCCURRED DURING _is_blocked CALC...!\nERROR MSG:', str(ex))
            
        return is_blocked

    def obstruction(self, ha, dec, dome_az, plot_result=False):
        """
        Compute the % obstruction of the aperture by the dome.

        Parameters
        ----------

        ha (float): hour angle in degrees
        dec (float): declination in degrees
        dome_az (float): dome azimuth (clockwise convention)
        plot_result (bool): if True, a plot with the sampled aperture and obstructed points will be shown
        """
        ratio = None
        
        # Sample points in a disk; resembling the aperture
        ap_xz = self._sample_disk(r_min=self.sec_radius/self.radius)
        
        ap_x, ap_z = ap_xz.T

        # Transfor those points to the aperture frame
        ap_pos = self._sample_aperture(ha, dec, -ap_x, ap_z)

        # Compute the no. rays, emanating from those points, blocked by the dome
        blocked = self._is_blocked(ap_pos, ha, dec, dome_az)
    
        ratio = blocked[blocked].size/blocked.size

        if plot_result:
            plot_aperture(ap_x, ap_z, blocked, self.radius, dome_az)

        return ratio
    
    def get_name(self):
        """Return aperture identifier."""
        return self._name


class TelescopeAperture(Aperture):
    """Primary aperture."""
    def __init__(self, rate=4):
        super().__init__(APERTURE_RADIUS, sec_radius=APERTURE_SEC_RADIUS, rate=rate)
        
        self._name = 'telescope'


class GuiderAperture(Aperture):
    """Autoguider aperture."""
    def __init__(self, rate=3):
        super().__init__(GUIDER_RADIUS, sec_radius=GUIDER_SEC_RADIUS, rate=rate)

        self._name = 'guider'

    def _transform(self, ha, dec):
        # Transform telescope aperture to guider aperture
        H_34 = transform(L_4*np.cos(GUIDER_ANGLE), 0, L_4*np.sin(GUIDER_ANGLE))

        # Get the telescope aperture pose
        H_telescope = super()._transform(ha, dec)

        H = H_telescope @ H_34

        return H


class FinderAperture(Aperture):
    """Finderscope aperture."""
    def __init__(self, rate=3):
        super().__init__(FINDER_RADIUS, rate=rate)

        self._name = 'finder'

    def _transform(self, ha, dec):
        # Transform telescope aperture to guider aperture & guider to finder
        H_34 = transform(L_4*np.cos(GUIDER_ANGLE), 0, L_4*np.sin(GUIDER_ANGLE))
        H_45 = transform(-L_5*np.cos(FINDER_ANGLE), 0, L_5*np.sin(FINDER_ANGLE))

        # Get the telescope aperture pose
        H_telescope = super()._transform(ha, dec)

        H = H_telescope @ H_34 @ H_45

        return H