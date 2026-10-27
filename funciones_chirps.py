import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist

def idw_interpolate_dataarray(data_array, power=2, xdim='x', ydim='y'):
    """Rellena los NaN de un DataArray 2D (y, x) con IDW."""
    da = data_array.transpose(ydim, xdim)
    vals = da.values.astype(float)

    X, Y = np.meshgrid(da[xdim].values, da[ydim].values)

    nan_mask = np.isnan(vals)
    if nan_mask.all():
        print("No hay puntos conocidos para interpolar.")
        return data_array.copy()
    if not nan_mask.any():
        print("No hay NaN para interpolar.")
        return data_array.copy()

    known_coords = np.column_stack([X[~nan_mask], Y[~nan_mask]])
    known_values = vals[~nan_mask]
    interp_coords = np.column_stack([X[nan_mask], Y[nan_mask]])

    d = cdist(interp_coords, known_coords)
    d = np.maximum(d, 1e-12)
    w = 1.0 / d**power

    out = vals.copy()
    out[nan_mask] = (w * known_values).sum(axis=1) / w.sum(axis=1)

    return da.copy(data=out).transpose(*data_array.dims)