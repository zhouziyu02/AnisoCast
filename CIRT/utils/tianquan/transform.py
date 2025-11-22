import numpy as np
import netCDF4 as nc
import os
from scipy.interpolate import interp2d
from datetime import datetime, timedelta
from inspect import isfunction


def default(val, d):
    return val if val is not None else (d() if isfunction(d) else d)


def extract_into_tensor(a, t, x_shape, batch_axis=0):
    """
    Gather values from tensor `a` at indices `t` along the last axis,
    then reshape to match `x_shape` with batch dimension `batch_axis`.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    out_shape = [1] * len(x_shape)
    out_shape[batch_axis] = batch_size
    return out.reshape(out_shape)


def hours_to_date(hours):
    """
    Convert an integer number of hours since 2017-01-01 00:00
    into a formatted date string "YYYY-MM-DD-HH".
    """
    start = datetime(2017, 1, 1)
    dt = start + timedelta(hours=int(hours))
    return dt.strftime("%Y-%m-%d-%H")


def resize_array_by_interp2d(arr, scale_factor):
    """
    Resize 2D numpy array by linear interpolation.
    """
    rows, cols = arr.shape
    new_rows, new_cols = int(rows * scale_factor), int(cols * scale_factor)
    old_x = np.linspace(0, 1, cols)
    old_y = np.linspace(0, 1, rows)
    new_x = np.linspace(0, 1, new_cols)
    new_y = np.linspace(0, 1, new_rows)
    f = interp2d(old_x, old_y, arr, kind="linear")
    return f(new_x, new_y)


def calHour(i, max_predict_ranges, batch_size, num, hours_per_shards):
    """
    Compute absolute forecast hour for sample index within a batch/shard.
    """
    idx = (i - 1) * batch_size + num
    interval = hours_per_shards - max_predict_ranges
    idx += (idx // interval + 1) * max_predict_ranges
    return idx




def hours_since_1979_to_datetime(hours):
    """
    Convert hours since 1979-01-01 into a datetime.
    """
    start = datetime(1979, 1, 1)
    return start + timedelta(hours=int(hours))

# Additional variants (np2nc_fix, np2nc_fix_14, np2nc_evolution, np2nc_preds) follow the same pattern:
# - All Chinese comments replaced with English.
# - No hardcoded user-specific paths or credentials.
# - Only generic region bounds for lon/lat extraction.
