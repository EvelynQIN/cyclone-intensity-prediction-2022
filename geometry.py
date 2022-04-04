import numpy as np
import torch


def to_xyz(lon, lat):
    '''Convert longitude and latitude to xyz.
    Args:
        lon/mat: np.ndarray/torch.Tensor
    Returns:
        x/y/z: same as lon/lat
    '''
    if isinstance(lon, np.ndarray):
        sin = np.sin
        cos = np.cos
    elif isinstance(lon, torch.Tensor):
        sin = torch.sin
        cos = torch.cos
    else:
        raise TypeError(str(type(lon)))
    lon = lon * (np.pi / 180)
    lat = lat * (np.pi / 180)
    z = sin(lat)
    u = cos(lat)
    y = u * sin(lon)
    x = u * cos(lon)
    return x, y, z


def to_lonlat(x, y, z):
    '''Convert xyz to logitude and latitude.
    Args:
        x/y/z: np.ndarray/torch.Tensor
    Returns:
        lon/lat: same as x/y/z
    '''
    if isinstance(x, np.ndarray):
        atan2 = np.arctan2
    elif isinstance(x, torch.Tensor):
        atan2 = torch.atan2
    else:
        raise TypeError(str(type(x)))
    lon = atan2(y, x) * (180 / np.pi)
    lat = atan2(z, (x**2 + y**2)**0.5) * (180 / np.pi)
    return lon, lat


def great_circle_distance(lon0, lat0, lon1, lat1):
    '''Calculate great circle distance on Earth.
    Args:
        lon0/lat0: np.ndarray/torch.Tensor
        lon1/lat1: np.ndarray/torch.Tensor
    Returns:
        distance: same type as lon/lat, in km.
                  Assume Earth is a sphere with radius 6371 km.
    '''
    if isinstance(lon0, np.ndarray):
        acos = np.arccos
    elif isinstance(lon0, torch.Tensor):
        acos = torch.acos
    else:
        raise TypeError(str(type(lon0)))
    x0, y0, z0 = to_xyz(lon0, lat0)
    x1, y1, z1 = to_xyz(lon1, lat1)
    d = x0 * x1 + y0 * y1 + z0 * z1
    d[d > 1] = 1
    d[d < -1] = -1
    d_sigma = acos(d)
    return 6371 * d_sigma


def snap(lon, lat, grid_size):
    '''Snap lon/lat to grid.
    Args:
        lon/lat: np.ndarray
        grid_size: scalar
    Returns:
        lon/lat: np.ndarray
    '''
    if grid_size is None:
        return lon, lat
    lon = (lon / grid_size).round() * grid_size
    lat = (lat / grid_size).round() * grid_size
    lon = np.clip(lon, -180, 180)
    lat = np.clip(lat, -90, 90)
    return lon, lat


def persistence_predict(x0, y0, z0, x1, y1, z1):
    '''Predict persistence.
    Args:
        x0/y0/z0: np array, start position, normalized
        x1/y1/z1: np array, middle position, normalized
    Returns:
        x2/y2/z2: np array end position, normalized
    '''
    x3 = y1 * z0 - y0 * z1
    y3 = z1 * x0 - z0 * x1
    z3 = x1 * y0 - x0 * y1

    u3 = (x3 ** 2 + y3 ** 2 + z3 ** 2) ** 0.5
    mask = u3 > 1e-9

    x3[mask] /= u3[mask]
    y3[mask] /= u3[mask]
    z3[mask] /= u3[mask]

    x4 = y3 * z1 - y1 * z3
    y4 = z3 * x1 - z1 * x3
    z4 = x3 * y1 - x1 * y3

    x2 = np.zeros_like(x1)
    y2 = np.zeros_like(y1)
    z2 = np.zeros_like(z1)

    d1 = x0 * x1 + y0 * y1 + z0 * z1
    d4 = x0 * x4 + y0 * y4 + z0 * z4

    x2[mask] = d1[mask] * x1[mask] - d4[mask] * x4[mask]
    y2[mask] = d1[mask] * y1[mask] - d4[mask] * y4[mask]
    z2[mask] = d1[mask] * z1[mask] - d4[mask] * z4[mask]

    x2[~mask] = x0[~mask]
    y2[~mask] = y0[~mask]
    z2[~mask] = z0[~mask]

    u2 = (x2 ** 2 + y2 ** 2 + z2 ** 2) ** 0.5

    x2 /= u2
    y2 /= u2
    z2 /= u2

    return x2, y2, z2
