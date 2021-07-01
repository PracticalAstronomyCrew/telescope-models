import numpy as np

def vec4(x, y, z):
    """
    Return a 4-element vector, appropriate 
    for coordinate transformations.
    """
    return np.array([x, y, z, 1])

def vec3(v4):
    """
    Convert a 4-element vector to a 3-element vector.
    """
    v3 = v4[:3].reshape(3)

    return v3

def transform(x, y, z):
    """Transform a vector (x', y', z', 1)."""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])

def rot_x(angle):
    """
    Rotate a vector (x, y, z, 1) about the x-axis 
    in a right-handed coordinate system.
    """
    angle = np.radians(angle)

    return np.array([
        [1,             0,              0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle),  np.cos(angle), 0],
        [0,             0,              0, 1],
    ])

def rot_y(angle):
    """
    Rotate a vector (x, y, z, 1) about the y-axis 
    in a right-handed coordinate system.
    """
    angle = np.radians(angle)

    return np.array([
        [np.cos(angle),  0,  -np.sin(angle), 0],
        [0,              1,               0, 0],
        [-np.sin(angle), 0,   np.cos(angle), 0],
        [0,              0,               0, 1],
    ])

def rot_z(angle):
    """
    Rotate a vector (x, y, z, 1) about the z-axis 
    in a right-handed coordinate system.
    """
    angle = np.radians(angle)

    return np.array([
        [np.cos(angle),  -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle),  0, 0],
        [0,                          0,  1, 0],
        [0,                          0,  0, 1],
    ])