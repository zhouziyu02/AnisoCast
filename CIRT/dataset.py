from datetime import datetime
from pathlib import Path
import re
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from CIRT import config


class S2SDataset(Dataset):
    """One daily input and two averages over 28 future target days.

    n_step is a target window length, not input history. With lead_time=15,
    outputs average days +15..+28 and +29..+42 relative to the input date.
    """

    def __init__(self, data_dir: str, years: List[int], n_step: int = 28,
                 lead_time: int = 1, kernel_size: int = 4, single_vars=None,
                 pred_single_vars=None, pred_pressure_vars=None,
                 is_normalized: bool = True):
        if n_step != 28:
            raise ValueError('S2SDataset requires n_step=28 for two 14-day target means.')
        if not isinstance(lead_time, int) or lead_time < 1:
            raise ValueError('lead_time must be a positive integer number of days.')
        self.n_step, self.lead_time = n_step, lead_time
        self.years = sorted(set(str(year) for year in years))
        self.single_vars = list(single_vars or [])
        self.pred_single_vars = list(pred_single_vars or [])
        self.pred_pressure_vars = list(pred_pressure_vars or [])
        self.is_normalized = is_normalized
        if not self.years:
            raise ValueError('At least one dataset year is required.')
        if not self.pred_pressure_vars and not self.pred_single_vars:
            raise ValueError('At least one target variable is required.')
        root = Path(data_dir).expanduser()
        self.data_dir = [root / 'pressure_level_1.5', root / 'single_level_1.5']
        self.normalization_file = [
            root / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
            root / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
        ]

        def dated_files(directory):
            by_date = {}
            for path in directory.glob('*.zarr'):
                match = re.search(r'(\d{8})\.zarr$', path.name)
                if not match or match.group(1)[:4] not in self.years:
                    continue
                date = np.datetime64(datetime.strptime(match.group(1), '%Y%m%d'), 'D')
                if date in by_date:
                    raise ValueError(f'Duplicate daily files for {date} in {directory}')
                by_date[date] = path
            missing = set(self.years) - {str(date)[:4] for date in by_date}
            if missing:
                raise FileNotFoundError(f'No daily data for years {sorted(missing)} in {directory}')
            return by_date

        pressure_files = dated_files(self.data_dir[0])
        self.dates = sorted(pressure_files)
        use_single = bool(self.single_vars or self.pred_single_vars)
        single_files = dated_files(self.data_dir[1]) if use_single else {}
        if use_single and set(pressure_files) != set(single_files):
            raise ValueError('Pressure and single-level daily dates do not match.')
        for first, second in zip(self.dates, self.dates[1:]):
            if second - first != np.timedelta64(1, 'D'):
                raise ValueError(f'Daily data are not contiguous: gap between {first} and {second}.')
        self.file_paths = [[pressure_files[date] for date in self.dates],
                           [single_files[date] for date in self.dates] if use_single else []]
        self.data_length = len(self.dates) - lead_time - n_step + 1
        if self.data_length <= 0:
            raise ValueError(f'Need at least {lead_time + n_step} contiguous days; found {len(self.dates)}.')

        self.statistics = {}
        if is_normalized:
            selections = {
                'pressure': (0, [f'{var}-{level}' for var in config.ERA5_PRESSURE_LIST
                                 for level in config.PRESSURE_LEVELS]),
                'pressure_pred': (0, [f'{var}-{level}' for var in self.pred_pressure_vars
                                      for level in config.PRESSURE_LEVELS]),
                'single': (1, self.single_vars), 'single_pred': (1, self.pred_single_vars),
            }
            for name, (source, params) in selections.items():
                if not params:
                    continue
                with xr.open_dataset(self.normalization_file[source], engine='zarr') as ds:
                    mean = ds['mean'].sel(param=params).values[:, None, None]
                    sigma = ds['sigma'].sel(param=params).values[:, None, None]
                if not np.isfinite(mean).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
                    raise ValueError(f'Invalid mean/sigma statistics in {self.normalization_file[source]}')
                self.statistics[name] = (mean, sigma)

    def __len__(self):
        return self.data_length

    def _read(self, file_idx, variables, pressure):
        source = 0 if pressure else 1
        with xr.open_dataset(self.file_paths[source][file_idx], engine='zarr') as ds:
            time = np.asarray(ds.time.values).reshape(-1)
            if time.size != 1 or np.datetime64(time[0], 'D') != self.dates[file_idx]:
                raise ValueError(f'File date/time mismatch in {self.file_paths[source][file_idx]}')
            selected = ds[variables]
            if 'time' in selected.dims:
                selected = selected.isel(time=0, drop=True)
            selected = selected.rename({dim: {'lat': 'latitude', 'lon': 'longitude'}[dim]
                                        for dim in ('lat', 'lon') if dim in selected.dims})
            if pressure:
                selected = selected.sel(level=config.PRESSURE_LEVELS)
                array = selected.to_array().transpose('variable', 'level', 'latitude', 'longitude').values
                array = array.reshape(-1, *array.shape[-2:])
            else:
                array = selected.to_array().transpose('variable', 'latitude', 'longitude').values
            return array

    def _fields(self, file_idx, target=False):
        parts = []
        for variables, pressure, name in (
            (self.pred_pressure_vars if target else config.ERA5_PRESSURE_LIST,
             True, 'pressure_pred' if target else 'pressure'),
            (self.pred_single_vars if target else self.single_vars,
             False, 'single_pred' if target else 'single'),
        ):
            if not variables:
                continue
            values = self._read(file_idx, variables, pressure)
            if self.is_normalized:
                mean, sigma = self.statistics[name]
                values = (values - mean) / sigma
            parts.append(torch.from_numpy(np.asarray(values, dtype=np.float32)))
        return torch.cat(parts, dim=0)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        x = self._fields(idx)
        daily_targets = torch.stack([
            self._fields(target_idx, target=True)
            for target_idx in range(idx + self.lead_time, idx + self.lead_time + self.n_step)
        ])
        y = torch.stack((daily_targets[:14].mean(dim=0), daily_targets[14:28].mean(dim=0)))
        timestamp = int(self.dates[idx].astype('datetime64[ns]').astype(np.int64))
        return timestamp, x, y
