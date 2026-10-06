# AnisoCast

Symmetric Composition of Anisotropic Operators for Global Subseasonal-to-Seasonal Climate Forecasting.

The supported model uses the legacy identifier `soon` and is implemented in `CIRT/models/soon.py`. Run commands from the repository root.

## Installation

Use Python 3.11 and an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

For GPU training, install a matching CUDA build of PyTorch 2.3.1 and torchvision 0.18.1 before installing these requirements. The preprocessing code uses Zarr 2 and NumCodecs 0.12.1.

## Data preparation

The default data root is `./data/S2S`. The `data/` directory contains preprocessing scripts, not the ERA5 arrays. Downloads read the public WeatherBench 2 daily ERA5 Zarr store specified in the scripts, select every sixth point from the 0.25-degree grid to obtain a 121 × 240 grid, and replace missing values with zero. This is index-stride sampling rather than conservative area regridding.

Download 1979–2018 pressure-level and single-level data:

```bash
PYTHON=python WORKERS=4 BATCH_JOBS=1 bash data/download_pressure.sh
PYTHON=python WORKERS=4 BATCH_JOBS=1 bash data/download_single.sh
```

For a small download, specify an inclusive date range:

```bash
python data/parallel_download_pressure_1p5.py --start 1979-01-01 --end 1979-01-01 --workers 1 --anon
python data/parallel_download_single_1p5.py --start 1979-01-01 --end 1979-01-01 --workers 1 --anon
```

Downloaders overwrite outputs by default. `--no-overwrite` validates and skips existing daily stores. Failed dates return a nonzero exit status. Root-level download scripts forward to the maintained `data/` implementations.

Generate normalization statistics using only the training period:

```bash
python data/calculate_climatology.py --dataset_name pressure_level_1.5 \
  --start 1979-01-01 --end 2016-12-31 --agg pairwise --scheduler threads --order_levels_by_config
python data/calculate_climatology.py --dataset_name single_level_1.5 \
  --start 1979-01-01 --end 2016-12-31 --agg pairwise --scheduler threads
```

Statistics use float64 accumulation to compute the unweighted time/grid mean and population standard deviation (`ddof=0`) independently for each channel. Missing dates, duplicate timestamps, and required-variable errors stop computation. Outputs follow this layout:

```text
data/S2S/
├── pressure_level_1.5/era5_pressure_full_1.5deg_YYYYMMDD.zarr
├── single_level_1.5/era5_single_full_1.5deg_YYYYMMDD.zarr
└── climatology_1.5/
    ├── climatology_pressure_level_1.5_new.zarr
    └── climatology_single_level_1.5_new.zarr
```

To move the data, set `ANISOCAST_DATA_DIR` consistently for preprocessing and training, or update the generated YAML's `data_args.data_dir` for evaluation. Arrays, logs, and checkpoints are excluded from Git.

## Forecast task

The input is one daily initial state with 63 channels: six pressure variables at ten levels, followed by 10 m u wind, 10 m v wind, and 2 m temperature. Pressure variables are geopotential, specific humidity, temperature, u wind, v wind, and vertical velocity. Levels are ordered `10, 50, 100, 200, 300, 500, 700, 850, 925, 1000 hPa`.

With `lead_time=15` and `n_step=28`, output index 0 is the mean of days **+15 through +28**, and index 1 is the mean of days **+29 through +42**, inclusive. `n_step` describes the future target span. Output shape is `[batch, 2, 63, 121, 240]`.

Training uses 1979–2016, validation uses 2017, and testing uses 2018. All targets stay inside their respective split. Pressure/surface dates must match and be contiguous.

## Training and evaluation

```bash
bash train_ours.sh --model-name soon --lr 1e-3 --embed 256 --depth 7 \
  --decoder-depth 1 --np 1 --tag aniso_depth7 --foreground
```

Use `--np 8` on an eight-GPU host. The launcher defaults to eight GPUs and background execution. Foreground mode returns training or evaluation failures directly. The current training loss is normalized, unweighted MSE. The default cosine scheduler uses `T_max=500` epochs; change `--t-max` explicitly when using a different schedule.

A tagged run writes `config.yaml`, `checkpoints/best.ckpt`, `metrics/metrics.csv`, `run_summary.json`, and `evaluation/` beneath `logs/soon/<tag>/`. The best checkpoint minimizes validation loss. Choose a new tag for a new experiment. The launcher calls `auto_evaluate.py` for its own checkpoint after successful training.

Separate evaluation:

```bash
python auto_evaluate.py --model_type soon \
  --config_file logs/soon/aniso_depth7/config.yaml \
  --checkpoint_path logs/soon/aniso_depth7/checkpoints/best.ckpt \
  --output_dir results/soon --tag aniso_depth7
```

The uploaded automatic evaluator is adapted to this repository's supported model and `inference.evaluate_soon` entry point. Checkpoint architecture is loaded strictly; explicit YAML overrides allow data location, test years, batch size and worker changes. Evaluation includes the final partial batch and writes per-channel/window RMSE, Bias, ACC, MS-SSIM, SpecDiv, and SpecRes. Predictions and targets are denormalized before evaluation. ACC uses cosine-latitude weighting and the scalar training-channel mean as its anomaly baseline.

```bash
bash train_ours.sh --help
python auto_evaluate.py --help
python -m inference.evaluate_soon --help
```

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The checks cover syntax, entry points, synthetic statistics, forecast windows, normalization and metrics, model gradients, checkpoint loading, and evaluation routing. Full ERA5 training and published forecast scores are outside these checks. Pretrained paper weights and experiment logs are not included. See `IMPLEMENTATION_DETAILS.txt` for the current implementation's definitions.
