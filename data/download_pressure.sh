# #!/usr/bin/env bash
# set -euo pipefail

# Limit thread usage to reduce resource contention
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

mkdir -p logs

# Download in parallel chunks (5 years per chunk), 12 workers per chunk
python data/parallel_download_pressure_1p5.py \
  --start 1979-01-01 --end 1983-12-31 --workers 12 --anon \
  > logs/1979_1983.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 1984-01-01 --end 1988-12-31 --workers 12 --anon \
  > logs/1984_1988.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 1989-01-01 --end 1993-12-31 --workers 12 --anon \
  > logs/1989_1993.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 1994-01-01 --end 1998-12-31 --workers 12 --anon \
  > logs/1994_1998.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 1999-01-01 --end 2003-12-31 --workers 12 --anon \
  > logs/1999_2003.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 2004-01-01 --end 2008-12-31 --workers 12 --anon \
  > logs/2004_2008.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 2009-01-01 --end 2013-12-31 --workers 12 --anon \
  > logs/2009_2013.log 2>&1 &

python data/parallel_download_pressure_1p5.py \
  --start 2014-01-01 --end 2018-12-31 --workers 12 --anon \
  > logs/2014_2018.log 2>&1 &

# Wait for all background tasks to complete
wait
echo "All 5-year chunks finished. Check logs/ for details."


