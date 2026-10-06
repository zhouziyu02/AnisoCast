import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path

from CIRT import config

def get_adjusting_weights():
    latitudes = torch.arange(90, -91.5, -1.5)
    latitudes_rad = torch.deg2rad(latitudes)
    weights = torch.cos(latitudes_rad)
    
    return weights[None, :, None]



class RMSE(nn.Module):
    """
    Compute root mean squared error (RMSE)
    """
    
    def __init__(self,
                 lat_adjusted=True):
        
        super(RMSE, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = nn.Parameter(get_adjusting_weights()) if lat_adjusted else None

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Adjust by latitude
        if self.lat_adjusted:
            weights = self.weights
            # Align weights latitude dimension with inputs (handle 120 vs 121, etc.)
            H = squared_diff.shape[-2]
            if weights.size(1) != H:
                weights = weights[:, :H, :]
            squared_diff = weights.size(1) * (weights / torch.sum(weights)) * squared_diff 
            
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        # Take the square root to get the RMSE
        rmse = torch.sqrt(mean_squared_error)
        
        return rmse



class MSE(nn.Module):
    """
    Compute mean squared error (MSE)
    """
    
    def __init__(self):
        
        super(MSE, self).__init__()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        return mean_squared_error



class Bias(nn.Module):
    """Compute bias (predictions - targets)
    """
    
    def __init__(self,
                lat_adjusted=True):
        
        super(Bias, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
        
        # Calculate difference between predictions and targets
        bias = predictions - targets
        
        if self.lat_adjusted:
            weights = self.weights
            H = bias.shape[-2]
            if weights.size(1) != H:
                weights = weights[:, :H, :]
            bias = weights.size(1) * (weights / torch.sum(weights)) * bias 
        
        # Calculate the mean bias
        mean_bias = torch.nanmean(bias)
        
        return mean_bias



class ACC(nn.Module):
    """
    Compute anomaly correlation coefficient (ACC) given climatology
    """
    def __init__(self,
                 lat_adjusted=True,
                 data_dir=None):
        
        super(ACC, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = nn.Parameter(get_adjusting_weights()) if lat_adjusted else None
        
        # Use provided data_dir or fall back to config.DATA_DIR
        if data_dir is None:
            data_dir = config.DATA_DIR
        
        # Retrieve climatology
        self.normalization_file = {
                'pressure_level': Path(data_dir) / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr', 
                'single_level': Path(data_dir) / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr', 
        }
        
        self.normalization_mean = {}
        for source, path in self.normalization_file.items():
            with xr.open_dataset(path, engine='zarr') as ds:
                self.normalization_mean[source] = ds['mean'].load()
        
    def forward(self, predictions, targets, param, source):
        
        # Retrieve mean climatology
        if predictions.shape != targets.shape:
            raise ValueError('ACC predictions and targets must have identical shapes.')
        climatology = torch.as_tensor(self.normalization_mean[source].sel(param=param).values,
                                      dtype=predictions.dtype, device=predictions.device)
        
        # Compute anomalies
        anomalies_targets = targets - climatology
        anomalies_predictions = predictions - climatology
        
        valid = torch.isfinite(anomalies_targets) & torch.isfinite(anomalies_predictions)
        anomalies_targets = torch.where(valid, anomalies_targets, 0)
        anomalies_predictions = torch.where(valid, anomalies_predictions, 0)
        if self.lat_adjusted:
            if predictions.shape[-2] != self.weights.shape[1]:
                raise ValueError('Latitude-adjusted ACC expects a 121-row 1.5-degree grid.')
            weights = self.weights.to(predictions).clamp_min(0)
        else:
            weights = predictions.new_tensor(1.0)

        # Compute ACC
        numerator = torch.sum(weights * anomalies_targets * anomalies_predictions)
        denominator = torch.sqrt(torch.sum(weights * anomalies_targets ** 2) *
                                 torch.sum(weights * anomalies_predictions ** 2))

        acc = torch.where(denominator > 0, numerator / denominator,
                          predictions.new_tensor(float('nan')))
        
        return acc



class MS_SSIM(nn.Module):
    """
    Compute Multi-Scale Structural SIMilarity(MS-SSIM) index
    """
    
    def __init__(
        self,
        data_range=255,
        size_average=True,
        kernel_size=11,
        sigma=1.5,
        weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        k1=0.01,
        k2=0.03
    ):
        """
        Args:
            data_range: max-min, usually use 1 or 255
            kernel_size: size of the Gaussian kernel
            sigma: standard deviation of the Gaussian kernel
            
        """
        super(MS_SSIM, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.weights = weights
        self.k1 = k1
        self.k2 = k2

    def rescale(self, data):
        """
        (B, H, W) -> (B,1,H,W) and rescale to (0,255) for each sample
        """
        # Add the additional axis to the data
        data_reshaped = data.unsqueeze(1)
        data_rescaled = torch.zeros_like(data_reshaped)
        
        for i in range(len(data)):
            min_val = data_reshaped[i].min()
            max_val = data_reshaped[i].max()
            span = max_val - min_val
            denominator = torch.where(span > 0, span, torch.ones_like(span))
            data_rescaled[i] = self.data_range * (data_reshaped[i] - min_val) / denominator
        
        return data_rescaled
        
        
    def gaussian_1d(self):
        """
        1-d Gaussian filter
        """
        coords = torch.arange(self.kernel_size, dtype=torch.float)
        coords -= self.kernel_size // 2
        
        g = torch.exp(-(coords ** 2)/(2*self.sigma**2))
        g /= torch.nansum(g)
        
        return g.unsqueeze(0).unsqueeze(0)
    
    
    def gaussian_filter(self, data, gaussian_kernel):
        """
        Gaussian filtering
        """
        conv = nn.functional.conv2d
        C = data.shape[1]
        out = data
        for i, s in enumerate(data.shape[2:]):
            if s>= gaussian_kernel.shape[-1]:
                out = conv(out, weight=gaussian_kernel.transpose(2+i, -1), stride=1, padding=0, groups=C)
                
        return out
            
        
    def ssim(self,
             X,
             Y,
             gaussian_kernel,
    ):
        C1 = (self.k1 * self.data_range) ** 2
        C2 = (self.k2 * self.data_range) ** 2
        
        compensation = 1.0
        
        gaussian_kernel = gaussian_kernel.to(X.device, dtype=X.dtype)
        
        mu1 = self.gaussian_filter(X, gaussian_kernel)
        mu2 = self.gaussian_filter(Y, gaussian_kernel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = compensation * (self.gaussian_filter(X * X, gaussian_kernel) - mu1_sq)
        sigma2_sq = compensation * (self.gaussian_filter(Y * Y, gaussian_kernel) - mu2_sq)
        sigma12 = compensation * (self.gaussian_filter(X * Y, gaussian_kernel) - mu1_mu2)

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

        ssim_per_channel = torch.nanmean(torch.flatten(ssim_map, 2), dim=-1)
        cs = torch.nanmean(torch.flatten(cs_map, 2), dim=-1)
        return ssim_per_channel, cs
    
    def ms_ssim(
        self,
        predictions,
        targets,
        size_average=True
    ):
        
        """
        predictions: a batch of a specific predicted physical variable at a specific level (B,H,W)
        targets: a batch of a specific target physical variable at a specific level (B,H,W)
        """
        
        # Handling missing values in predictions
        pred_means = torch.nanmean(predictions, dim=(-2, -1))[:, None, None]
        predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
        
        predictions = self.rescale(predictions).squeeze(dim=-1).squeeze(dim=-1)
        targets = self.rescale(targets).squeeze(dim=-1).squeeze(dim=-1)
        
        avg_pool = nn.functional.avg_pool2d
        
        window = self.gaussian_1d()
        window = window.repeat([predictions.shape[1]] + [1] * ( len(predictions.shape)-1))
        
        weights_tensor = predictions.new_tensor(self.weights)
        
        levels = len(self.weights)
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = self.ssim(predictions, targets, window)
            
            if i < levels - 1:
                mcs.append(torch.relu(cs))
                padding = [s%2 for s in predictions.shape[2:]]
                predictions = avg_pool(predictions, kernel_size=2, padding=padding)
                targets = avg_pool(targets, kernel_size=2, padding=padding)
        
        ssim_per_channel = torch.relu(ssim_per_channel)  
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

        if size_average:
            return torch.nanmean(ms_ssim_val)
        
        else:
            return torch.nanmean(ms_ssim_val, dim=1)

                
    def forward(self, predictions, targets):
        
        return self.ms_ssim(
            predictions,
            targets,
        )



class SpectralDiv(nn.Module):
    """
    Compute Spectral divergence given the top-k percentile wavenumber (higher k means higher frequency)
    (1) Validation mode: targeting specific top-k percentile wavenumber (higher k means higher frequency) is permissible
    (2) Training mode: computing metric along the entire wavenumber since some operation e.g., binning is nonautograd-able
    """
    def __init__(
        self,   
        percentile=0.9,
        input_shape=(121,240),
        is_train=True
    ):
        
        super(SpectralDiv, self).__init__()
        
        self.percentile = percentile
        self.is_train = is_train
        
        # Default k grid placeholders based on input_shape; will be recomputed per input in forward
        nx, ny = input_shape
        kx = torch.fft.fftfreq(nx) * nx
        ky = torch.fft.fftfreq(ny) * ny
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        self.k = torch.sqrt(kx**2 + ky**2).reshape(-1)
        self.k_low = 0.5
        self.k_upp = torch.max(self.k)
        self.k_nbin = torch.arange(self.k_low, torch.max(self.k), 1).size(0)
        self.k_percentile_idx = int(self.k_nbin * self.percentile)
        
    def forward(self, predictions, targets):
        
        # Recompute k grid dynamically from input spatial size to avoid mismatches (e.g., 120x240 vs 121x240)
        device = predictions.device
        nx, ny = predictions.shape[-2], predictions.shape[-1]
        kx = (torch.fft.fftfreq(nx, device=device) * nx)
        ky = (torch.fft.fftfreq(ny, device=device) * ny)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k = torch.sqrt(kx**2 + ky**2).reshape(-1)
        k_low = self.k_low
        k_upp = torch.max(k)
        k_nbin = torch.arange(k_low, k_upp, 1, device=device).size(0)
        k_percentile_idx = int(k_nbin * self.percentile)
        
        predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-2], predictions.shape[-1])
        targets = targets.reshape(targets.shape[0], -1, targets.shape[-2], targets.shape[-1])
        
        assert predictions.shape[1] == targets.shape[1]
        nc = predictions.shape[1]
        
        # Handling missing values in predictions
        pred_means = torch.nanmean(predictions, dim=(-2, -1), keepdim=True)
        predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
        
        # Compute along mini-batch
        predictions, targets = torch.nanmean(predictions, dim=0), torch.nanmean(targets, dim=0)
        
        # Transform prediction and targets onto the Fourier space and compute the power
        predictions_power, targets_power = torch.fft.fft2(predictions), torch.fft.fft2(targets)
        predictions_power, targets_power = torch.abs(predictions_power)**2, torch.abs(targets_power)**2

        ## If validation, we can target specific quantiles by binning and sorting
        if not self.is_train:
            import torchist
            x_vec = k.repeat(nc)
            w_pred = predictions_power.reshape(-1)
            w_targ = targets_power.reshape(-1)
            base_hist = torchist.histogram(x_vec, k_nbin, k_low, k_upp)
            predictions_Sk = torchist.histogram(x_vec, k_nbin, k_low, k_upp, weights=w_pred) / base_hist
            targets_Sk = torchist.histogram(x_vec, k_nbin, k_low, k_upp, weights=w_targ) / base_hist
            
            # Extract top-k percentile wavenumber and its corresponding power spectrum
            predictions_Sk = predictions_Sk[k_percentile_idx:]
            targets_Sk = targets_Sk[k_percentile_idx:]
            
            # Normalize as pdf along ordered k
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk)
        
        ## If training, compute the entire power spectrum 
        ## NOTE: targeting specific quantiles is yet to be implemented in autograd (i.e., binning operation)
        else:
            predictions_Sk, targets_Sk = predictions_power, targets_power
        
            # Normalize as pdf of each channel dimension
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk, dim=(-2, -1), keepdim=True)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk, dim=(-2, -1), keepdim=True)

        # Compute spectral Sk divergence
        div = torch.nansum(targets_Sk * torch.log(torch.clamp(targets_Sk / predictions_Sk, min=1e-9)))
        return div



class SpectralRes(nn.Module):
    """
    Compute Spectral residual 
    (1) Validation mode: targeting specific top-k percentile wavenumber (higher k means higher frequency) is permissible
    (2) Training mode: computing metric along the entire wavenumber since some operation e.g., binning is nonautograd-able
    """
    def __init__(
        self,   
        percentile=0.9,
        input_shape=(121,240),
        is_train=True
    ):
        
        super(SpectralRes, self).__init__()
        
        self.percentile = percentile
        self.is_train = is_train
        
        # Compute the discrete Fourier Transform sample frequencies for a signal of size
        nx, ny = input_shape
        kx = torch.fft.fftfreq(nx) * nx
        ky = torch.fft.fftfreq(ny) * ny
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        
        self.k = torch.sqrt(kx**2 + ky**2).reshape(-1)
        self.k_low = 0.5
        self.k_upp = torch.max(self.k)
        self.k_nbin = torch.arange(self.k_low, torch.max(self.k), 1).size(0)
        
        # Get percentile index
        self.k_percentile_idx = int(self.k_nbin * self.percentile)
        
    def forward(self, predictions, targets):
        
        # Recompute k grid dynamically from input spatial size to avoid mismatches (e.g., 120x240 vs 121x240)
        device = predictions.device
        nx, ny = predictions.shape[-2], predictions.shape[-1]
        kx = (torch.fft.fftfreq(nx, device=device) * nx)
        ky = (torch.fft.fftfreq(ny, device=device) * ny)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k = torch.sqrt(kx**2 + ky**2).reshape(-1)
        k_low = self.k_low
        k_upp = torch.max(k)
        k_nbin = torch.arange(k_low, k_upp, 1, device=device).size(0)
        k_percentile_idx = int(k_nbin * self.percentile)
        
        predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-2], predictions.shape[-1])
        targets = targets.reshape(targets.shape[0], -1, targets.shape[-2], targets.shape[-1])
        
        assert predictions.shape[1] == targets.shape[1]
        nc = predictions.shape[1]
        
        # Handling missing values in predictions
        pred_means = torch.nanmean(predictions, dim=(-2, -1), keepdim=True)
        predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
        
        # Compute along mini-batch
        predictions, targets = torch.nanmean(predictions, dim=0), torch.nanmean(targets, dim=0)
        
        # Transform prediction and targets onto the Fourier space and compute the power
        predictions_power, targets_power = torch.fft.fft2(predictions), torch.fft.fft2(targets)
        predictions_power, targets_power = torch.abs(predictions_power)**2, torch.abs(targets_power)**2

        ## If validation, we can target specific quantiles by binning and sorting
        if not self.is_train:
            import torchist
            x_vec = k.repeat(nc)
            w_pred = predictions_power.reshape(-1)
            w_targ = targets_power.reshape(-1)
            base_hist = torchist.histogram(x_vec, k_nbin, k_low, k_upp)
            predictions_Sk = torchist.histogram(x_vec, k_nbin, k_low, k_upp, weights=w_pred) / base_hist
            targets_Sk = torchist.histogram(x_vec, k_nbin, k_low, k_upp, weights=w_targ) / base_hist
            
            # Extract top-k percentile wavenumber and its corresponding power spectrum
            predictions_Sk = predictions_Sk[k_percentile_idx:]
            targets_Sk = targets_Sk[k_percentile_idx:]
            
            # Normalize as pdf along ordered k
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk)
        
        ## If training, compute the entire power spectrum 
        ## NOTE: targeting specific quantiles is yet to be implemented in autograd (i.e., binning operation)
        else:
            predictions_Sk, targets_Sk = predictions_power, targets_power
        
            # Normalize as pdf of each channel dimension
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk, dim=(-2, -1), keepdim=True)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk, dim=(-2, -1), keepdim=True)

        # Compute spectral Sk residual
        res = torch.sqrt(torch.nanmean(torch.square(predictions_Sk - targets_Sk)))
        return res
