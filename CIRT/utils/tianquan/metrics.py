import numpy as np
import torch
from scipy import stats
import xarray as xr
import xskillscore as xs


def mse(pred, y, vars, lat=None, mask=None):
    """
    Mean Squared Error per variable.

    Args:
        pred (Tensor[B, V, H, W]): Predictions.
        y (Tensor[B, V, H, W]): Ground truth.
        vars (list[str]): Variable names.
        mask (Tensor or None): Optional mask of shape [H, W].
    Returns:
        dict: MSE per variable and overall loss.
    """
    error = (pred - y) ** 2
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = error[:, i].mean()
        if mask is not None:
            loss_dict["loss"] = (error.mean(dim=1) * mask).sum() / mask.sum()
        else:
            loss_dict["loss"] = error.mean(dim=1).mean()
    return loss_dict


def get_max_min_blocks(x, block_size=8):
    """
    Partition tensor [B, V, H, W] into non-overlapping blocks and compute each block’s max and min.

    Args:
        x (Tensor[B, V, H, W]): Input tensor.
        block_size (int): Block height and width.

    Returns:
        max_vals (Tensor[B, V, num_blocks]), min_vals (Tensor[B, V, num_blocks])
    """
    B, V, H, W = x.shape
    assert H % block_size == 0 and W % block_size == 0

    # Create blocks
    blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    B, V, nH, nW, bH, bW = blocks.shape
    blocks = blocks.contiguous().view(B, V, nH * nW, bH, bW)

    max_vals = blocks.max(dim=-1)[0].max(dim=-1)[0]
    min_vals = blocks.min(dim=-1)[0].min(dim=-1)[0]
    return max_vals, min_vals


def expand_blocks_to_tensor(block_vals, orig_shape, block_size=8):
    """
    Expand per-block values back to a full-resolution tensor by tiling.

    Args:
        block_vals (Tensor[B, V, num_blocks]): Values per block.
        orig_shape (tuple): Desired shape (B, V, H, W).
        block_size (int): Block size.

    Returns:
        Tensor[B, V, H, W]
    """
    B, V, num_blocks = block_vals.shape
    _, _, H, W = orig_shape
    out = torch.zeros(orig_shape, device=block_vals.device, dtype=block_vals.dtype)
    idx = 0
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            out[:, :, i:i+block_size, j:j+block_size] = block_vals[:, :, idx].unsqueeze(-1).unsqueeze(-1)
            idx += 1
    return out


def lat_weighted_mse(pred, y, vars, lat, mask=None, anomaly=None, extreme=False, weight=None):
    """
    Latitude-weighted MSE with optional anomaly or extreme-block terms.

    Args:
        pred (Tensor[B, V, H, W])
        y (Tensor[B, V, H, W])
        vars (list[str])
        lat (array-like[H])
        mask (Tensor or None)
        anomaly (Tensor or None)
        extreme (bool): If True, include blockwise max/min error.
        weight (Tensor or None): Per-variable weights of length V.
    """
    if anomaly is not None:
        error = (pred - y) ** 2 + (pred - anomaly) ** 2
    elif extreme:
        pmax, pmin = get_max_min_blocks(pred)
        ymax, ymin = get_max_min_blocks(y)
        max_err = (pmax - ymax) ** 2
        min_err = (pmin - ymin) ** 2
        error = (pred - y) ** 2 + 0.1 * expand_blocks_to_tensor(max_err, pred.shape) \
                                     + 0.1 * expand_blocks_to_tensor(min_err, pred.shape)
    elif weight is not None:
        error = (pred - y) ** 2 * weight.view(1, -1, 1, 1).to(pred.device)
    else:
        error = (pred - y) ** 2

    # Latitude weighting
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(error.device).to(error.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            term = error[:, i] * w_lat
            loss_dict[var] = (term * mask).sum() / mask.sum() if mask is not None else term.mean()
        if mask is not None:
            loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
        else:
            # Custom aggregation to balance variables
            base = (error[:, 0] * w_lat).mean()
            agg = 0.0
            for i in range(len(vars)):
                term = (error[:, i] * w_lat).mean()
                agg += term / (term / base).detach()
            loss_dict["loss"] = agg
    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Validation latitude-weighted MSE, logging per-variable with a suffix.

    Args:
        transform: function to apply to pred and y before error computation.
        clim: (unused) climatology placeholder.
    """
    error = (pred - y) ** 2
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(error.device).to(error.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()
        loss_dict["w_mse"] = np.mean([v.cpu().item() for v in loss_dict.values()])
    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Validation latitude-weighted RMSE (root MSE).

    Args:
        transform: callable to apply to pred and y.
    """
    pred_t = transform(pred)
    y_t = transform(y)
    error = (pred_t - y_t) ** 2

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(error.device).to(error.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            rmse_i = torch.sqrt((error[:, i] * w_lat).mean(dim=(-2, -1)))
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = rmse_i.mean()
        loss_dict["w_rmse"] = np.mean([v.cpu().item() for v in loss_dict.values()])
    return loss_dict


def rmse(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Validation unweighted RMSE.
    """
    pred_t = transform(pred)
    y_t = transform(y)
    error = (pred_t - y_t) ** 2

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            rmse_i = torch.sqrt(error[:, i].mean(dim=(-2, -1)))
            loss_dict[f"rmse_{var}_{log_postfix}"] = rmse_i.mean()
        loss_dict["rmse"] = np.mean([v.cpu().item() for v in loss_dict.values()])
    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix, anomaly=None):
    """
    Latitude-weighted anomaly correlation (accuracy).

    Args:
        clim (Tensor[V, H, W]): Climatology to remove mean seasonal cycle.
    """
    pred_t = transform(pred) - clim.unsqueeze(0)
    y_t = transform(y) - clim.unsqueeze(0)

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).view(1, -1, 1).to(pred.device).to(pred.dtype)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            p = pred_t[:, i] - pred_t[:, i].mean()
            g = y_t[:, i] - y_t[:, i].mean()
            num = (w_lat * p * g).sum()
            den = torch.sqrt((w_lat * p**2).sum() * (w_lat * g**2).sum())
            loss_dict[f"acc_{var}_{log_postfix}"] = (num / den).item()
        loss_dict["acc"] = np.mean(list(loss_dict.values()))
    return loss_dict


def crps(y_pred, y_true, transform, vars, lat, clim, log_postfix=None, anomaly=None):
    """
    Continuous Ranked Probability Score per variable and overall.

    Args:
        y_pred (Tensor[B, M, V, H, W]): Ensemble predictions.
        y_true (Tensor[B, V, H, W])
    """
    if transform is not None:
        y_pred = transform(y_pred)
        y_true = transform(y_true)

    crps_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            yp = y_pred[:, :, i].cpu().numpy()
            yt = y_true[:, i].cpu().numpy()
            da_true = xr.DataArray(yt, dims=["batch", "lat", "lon"])
            da_pred = xr.DataArray(yp, dims=["batch", "member", "lat", "lon"])
            crps_var = xs.crps_ensemble(da_true, da_pred).item()
            crps_dict[f"crps_{var}_{log_postfix}"] = crps_var

        # overall CRPS
        ytp = y_pred.cpu().numpy()
        ytt = y_true.cpu().numpy()
        da_true_all = xr.DataArray(ytt, dims=["batch", "var", "lat", "lon"])
        da_pred_all = xr.DataArray(ytp, dims=["batch", "member", "var", "lat", "lon"])
        crps_dict["crps"] = xs.crps_ensemble(da_true_all, da_pred_all).item()
    return crps_dict


def remove_nans(pred, gt):
    """
    Remove NaNs and infinities from two 1D tensors.

    Returns filtered (pred, gt).
    """
    m = ~torch.isnan(pred) & ~torch.isinf(pred) & ~torch.isnan(gt) & ~torch.isinf(gt)
    return pred[m], gt[m]


def pearson(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    Pearson correlation at specified forecast steps.

    Args:
        pred, y (Tensor[B, T, V, H, W])
        log_steps (list[int]), log_days (list[int])
    """
    pred_t = transform(pred)
    y_t = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                p = pred_t[:, step-1, i].reshape(-1)
                g = y_t[:, step-1, i].reshape(-1)
                p, g = remove_nans(p, g)
                loss_dict[f"pearsonr_{var}_day_{day}"] = stats.pearsonr(p.cpu().numpy(), g.cpu().numpy())[0]
        loss_dict["pearsonr"] = np.mean(list(loss_dict.values()))
    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    Combined normalized RMS error and bias, weighted by latitude.

    Uses lat_weighted_nrmses and lat_weighted_nrmseg.
    """
    # Normalized error (space mean squared difference of time means)
    def nrmses():
        loss = {}
        pred_t = transform(pred)
        y_t = transform(y)
        w = torch.from_numpy(np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean()).view(1, -1, 1).to(pred.device)
        for i, var in enumerate(vars):
            pr = pred_t[:, i]
            gt = y_t[:, i]
            mse = ((pr.mean(dim=0) - gt.mean(dim=0))**2 * w).mean()
            loss[f"w_nrmses_{var}"] = torch.sqrt(mse) / clim
        return loss

    # Normalized error of ensemble-mean bias
    def nrmseg():
        loss = {}
        pred_t = transform(pred)
        y_t = transform(y)
        w = torch.from_numpy(np.cos(np.deg2rad(lat)) / np.cos(np.deg2rad(lat)).mean()).view(1, -1, 1).to(pred.device)
        for i, var in enumerate(vars):
            pr = (pred_t[:, i] * w).mean(dim=(-2, -1))
            gt = (y_t[:, i] * w).mean(dim=(-2, -1))
            loss[f"w_nrmseg_{var}"] = torch.sqrt(((pr - gt)**2).mean()) / clim
        return loss

    e1 = nrmses()
    e2 = nrmseg()
    combined = {k: e1[k] + 5 * e2[k] for k in e1}
    combined.update(e1)
    combined.update(e2)
    return combined
