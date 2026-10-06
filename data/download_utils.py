"""Small validation helpers shared by the ERA5 preprocessing scripts."""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr


def parse_date(value):
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a valid date in YYYY-MM-DD format") from exc


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return number


def validate_range(start, end):
    if start > end:
        raise ValueError("Start date must be on or before end date")


def skip_existing_store(path, variables, date, overwrite):
    """Skip readable existing outputs only when overwrite is disabled."""
    if not Path(path).exists() or overwrite:
        return False
    with xr.open_zarr(path, consolidated=True) as dataset:
        if not all(variable in dataset for variable in variables) or "time" not in dataset.coords:
            raise ValueError(f"Invalid existing output {path}; remove it or enable overwrite")
        dates = pd.DatetimeIndex(dataset.time.values.reshape(-1))
        if len(dates) == 0 or not all(dates.normalize() == pd.Timestamp(date).normalize()):
            raise ValueError(f"Existing output has inconsistent dates: {path}")
    print(f"[SKIP] {date:%Y-%m-%d} -> {path}")
    return True
