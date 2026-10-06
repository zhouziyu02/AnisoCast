# AnisoCast Model

Symmetric Composition of Anisotropic Operators for Global Subseasonal-to-Seasonal Climate Forecasting

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

The default data root is `./data/S2S`, as defined in `data/config.py`. Downloaded files are saved in `data/S2S/pressure_level_1.5/` and `data/S2S/single_level_1.5/`. Normalization statistics are generated separately in `data/S2S/climatology_1.5/` using `data/calculate_climatology.py`. The `data/` folder contains preprocessing scripts, not the ERA5 dataset itself.

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
