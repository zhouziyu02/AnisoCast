import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime
import re
from tqdm import tqdm
from CIRT import config 
from torch_geometric.data import Data

class S2SDataset(Dataset):
    """
    Dataset object to handle input reanalysis.

    Params:
        years <List[int]>      : list of years to load and process,
        n_step <int>           : number of contiguous timesteps included in the data (default: 1)
        lead_time <int>        : delta_t ahead in time, useful for direct prediction (default: 1)
        single_vars <List[str]>  : list of land variables to include (default: empty)`
        ocean_vars <List[str]> : list of sea/ice variables to include (default: empty)`
        is_normalized <bool>   : flag to indicate whether we should perform normalization or not (default: True)
    """

    def __init__(
        self,
        data_dir: str,
        years: List[int],
        n_step: int = 1,
        lead_time: int = 1,
        kernel_size: int = 4,
        era5_vars: List[str] = [],
        lra5_vars: List[str] = [],
        oras5_vars: List[str] = [],
        pred_era5_vars: List[str] = [],
        pred_lra5_vars: List[str] = [],
        pred_oras5_vars: List[str] = [],
        is_normalized: bool = False,
    ) -> None:
        # self.data_dir = [
        #     Path(data_dir) / 'era5',
        #     Path(data_dir) / 'lra5',
        #     Path(data_dir) / 'oras5',
        # ]
        # self.normalization_file = [
        #     Path(data_dir) / 'climatology_1.5' / 'climatology_era5.zarr',
        #     Path(data_dir) / 'climatology_1.5' / 'climatology_lra5.zarr',
        #     Path(data_dir) / 'climatology_1.5' / 'climatology_oras5.zarr',
        # ]
        self.data_dir = [
            Path(data_dir) / 'pressure_level_1.5',
            Path(data_dir) / 'single_level_1.5',
        ]
        self.normalization_file = [
            Path(data_dir) / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
            Path(data_dir) / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
        ]

        self.years = [str(year) for year in years]
        self.n_step = n_step
        self.lead_time = lead_time
        self.era5_vars = era5_vars
        self.lra5_vars = lra5_vars
        self.oras5_vars = oras5_vars
        self.pred_era5_vars = pred_era5_vars
        self.pred_lra5_vars = pred_lra5_vars
        self.pred_oras5_vars = pred_oras5_vars
        self.is_normalized = is_normalized

        # Subset files that match with patterns (eg. years specified)
        era5_files, lra5_files, oras5_files = list(), list(), list()
        # pressure_level_files, single_level_merge_files = list(), list()
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'

            curr_files = [
                list(self.data_dir[0].glob(f'*{year}*.zarr')),
                list(self.data_dir[1].glob(f'*{year}*.zarr')),
                list(self.data_dir[2].glob(f'*{year}*.zarr'))
            ]

            era5_files.extend([f for f in curr_files[0] if re.match(pattern, str(f.name))])
            lra5_files.extend([f for f in curr_files[1] if re.match(pattern, str(f.name))])
            oras5_files.extend([f for f in curr_files[2] if re.match(pattern, str(f.name))])

        era5_files.sort(); lra5_files.sort(); oras5_files.sort()
        self.file_paths = [era5_files, lra5_files, oras5_files]

        # Retrieve climatology (i.e., mean and sigma) to normalize
        self.mean_era5 = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].sel(param=[f"{param}-{level}" for param in self.era5_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.mean_lra5 = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].sel(param=self.lra5_vars).values[:, np.newaxis, np.newaxis]
        self.mean_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['mean'].sel(param=self.oras5_vars).values[:, np.newaxis, np.newaxis]
        self.mean_era5_pred = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].sel(param=[f"{param}-{level}" for param in self.pred_era5_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.mean_lra5_pred = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].sel(param=self.pred_lra5_vars).values[:, np.newaxis, np.newaxis]
        self.mean_oras5_pred = xr.open_dataset(self.normalization_file[2], engine='zarr')['mean'].sel(param=self.pred_oras5_vars).values[:, np.newaxis, np.newaxis]


        self.sigma_era5 = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].sel(param=[f"{param}-{level}" for param in self.era5_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.sigma_lar5 = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].sel(param=self.lra5_vars).values[:, np.newaxis, np.newaxis]
        self.sigma_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['sigma'].sel(param=self.oras5_vars).values[:, np.newaxis, np.newaxis]
        self.sigma_era5_pred = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].sel(param=[f"{param}-{level}" for param in self.pred_era5_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.sigma_lar5_pred = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].sel(param=self.pred_lra5_vars).values[:, np.newaxis, np.newaxis]
        self.sigma_oras5_pred = xr.open_dataset(self.normalization_file[2], engine='zarr')['sigma'].sel(param=self.pred_oras5_vars).values[:, np.newaxis, np.newaxis]


    def __len__(self):
        data_length = len(self.file_paths[0]) - self.n_step - self.lead_time
        return data_length

    def __getitem__(self, idx):
        pred_indices =  [target_idx for target_idx in range(idx + self.lead_time, idx + self.lead_time + self.n_step)]
        [idx]
        # pressure_level_data, single_level_merge_data, oras5_data = list(), list(), list()
        ear5_data, lra5_data, oras5_data = list(), list(), list()
        era5_data_pred, lra5_data_pred, oras5_data_pred = list(), list(), list()

        # ear5_data.append(xr.open_dataset(self.file_paths[0][idx], engine='zarr')[self.era5_vars].to_array().values)
        ear5_data.append(xr.open_dataset(self.file_paths[0][idx], engine='zarr')[self.era5_vars].sel(
            level=[50, 100, 200, 300, 500, 700, 850, 925, 1000]).sortby('level', ascending=False).to_array().values)

        # Process single_level_merge
        if len(self.lra5_vars) > 0:
            lra5_data.append(xr.open_dataset(self.file_paths[1][idx], engine='zarr')[self.lra5_vars].to_array().values)
        if len(self.oras5_vars) > 0:
            oras5_data.append(xr.open_dataset(self.file_paths[2][idx], engine='zarr')[self.oras5_vars].to_array().values)

        for step_idx in pred_indices:
            # # Process era5
            # era5_data_pred.append(xr.open_dataset(self.file_paths[0][step_idx], engine='zarr')[self.pred_era5_vars].to_array().values)
            era5_data_pred.append(xr.open_dataset(self.file_paths[0][step_idx], engine='zarr')[self.pred_era5_vars].sel(
                level=[50, 100, 200, 300, 500, 700, 850, 925, 1000]).sortby('level', ascending=False).to_array().values)
            # Process lra5 or orsa5
            if len(self.pred_lra5_vars) > 0:
                lra5_data_pred.append(xr.open_dataset(self.file_paths[1][step_idx], engine='zarr')[self.pred_lra5_vars].to_array().values)
            if len(self.pred_oras5_vars) > 0:
                oras5_data_pred.append(xr.open_dataset(self.file_paths[2][step_idx], engine='zarr')[self.pred_oras5_vars].to_array().values)

        # Permutation / reshaping
        era5_data, lra5_data, oras5_data = np.array(ear5_data), np.array(lra5_data), np.array(oras5_data)
        era5_data = era5_data.reshape(era5_data.shape[0], -1, era5_data.shape[-2], era5_data.shape[-1]) # Merge (param, level) dims

        era5_data_pred, lra5_data_pred, oras5_data_pred = np.array(era5_data_pred), np.array(lra5_data_pred), np.array(oras5_data_pred)
        era5_data_pred = era5_data_pred.reshape(era5_data_pred.shape[0], -1, era5_data_pred.shape[-2], era5_data_pred.shape[-1]) # Merge (param, level) dims

        # era5_data_pred
        if self.is_normalized:
            era5_data = (era5_data - self.mean_era5[np.newaxis, :, :, :]) / self.sigma_era5[np.newaxis, :, :, :]
            lra5_data = (lra5_data - self.mean_lra5[np.newaxis, :, :, :]) / self.sigma_lar5[np.newaxis, :, :, :]
            oras5_data = (oras5_data - self.mean_oras5[np.newaxis, :, :, :]) / self.sigma_oras5[np.newaxis, :, :, :]
            era5_data_pred = (era5_data_pred - self.mean_era5_pred[np.newaxis, :, :, :]) / self.sigma_era5_pred[np.newaxis, :, :, :]
            lra5_data_pred = (lra5_data_pred - self.mean_lra5_pred[np.newaxis, :, :, :]) / self.sigma_lar5_pred[np.newaxis, :, :, :]
            oras5_data_pred = (oras5_data_pred - self.mean_oras5_pred[np.newaxis, :, :, :]) / self.sigma_oras5_pred[np.newaxis, :, :, :]

        # Concatenate along parameter dimension, only if they are specified (i.e., non-empty)
        input_data = [t for t in [torch.tensor(era5_data), torch.tensor(lra5_data), torch.tensor(oras5_data)] if t.nelement() > 0]
        input_data = torch.cat(input_data, dim=1)
        # 下采样 [1, 63, 121, 240] -> [1, 63, 61, 120]
        # input_data = downsample_tensor(input_data)

        output_data = [t for t in [torch.tensor(era5_data_pred), torch.tensor(lra5_data_pred), torch.tensor(oras5_data_pred)] if t.nelement() > 0]
        output_data = torch.cat(output_data, dim=1)
        # 下采样 [1, 63, 121, 240] -> [1, 63, 61, 120]
        # output_data = downsample_tensor(output_data)

        timestamp = xr.open_dataset(self.file_paths[0][idx], engine='zarr').time.values.item()
        x, y = input_data[0].float(), torch.stack([torch.mean(output_data[0:14].float(),dim=0), torch.mean(output_data[14:28].float(),dim=0)],dim=0)
        return timestamp, x, y




