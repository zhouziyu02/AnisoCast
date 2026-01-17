# SOON Model

SOON (Symmetric Orthogonal Operator Network) for Sub-seasonal to Seasonal (S2S) Weather Forecasting.

## Usage

### Step 1: Download Data

Download ERA5 data using scripts in the `data/` folder:

```bash
# Download pressure level data
bash data/download_pressure.sh

# Download single level data
bash data/download_single.sh
```

Or use parallel download scripts:

```bash
python data/parallel_download_pressure_1p5.py
python data/parallel_download_single_1p5.py
```

Data will be saved in `S2S/pressure_level_1.5/` and `S2S/climatology_1.5/` directories.

### Step 2: Train

Train the model with recommended configuration:

```bash
bash train_ours.sh --model-name soon --lr 1e-3 --embed 256 --depth 7 --decoder-depth 1 --tag lr1e-3_embed256_depth7_dedep1
```

Training logs and checkpoints will be saved in `logs/soon/<tag>/` directory.

For more training options, run:

```bash
bash train_ours.sh --help
```
