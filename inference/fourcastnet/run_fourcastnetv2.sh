export CUDA_VISIBLE_DEVICES=0

export AI_MODELS_ASSETS="/mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/fourcastnetv2_assets"


# python -m ai_models \
#    --download-assets --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/fourcastnetv2_assets fourcastnetv2-small --only-gpu

# python -m ai_models \
#   --download-assets \
#   --assets "$AI_MODELS_ASSETS" \
#   --input cds \
#   --date 20180101 \
#   --time 0000 \
#   fourcastnetv2-small \
#   --only-gpu



python /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/fourcastnet/generate.py \
  --input /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/fourcastnetv2-small.grib \
  --out_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet \
  --idx_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/.cfgrib_index \
  --levels_hpa 1000,925,850,700,600,500,400,300,250


# $ pip install "cdsapi>=0.7.7"

# cat > ~/.cdsapirc <<'EOF'
# url: https://cds.climate.copernicus.eu/api
# key: f4b09277-774e-499f-8d4d-fecabd1f08f4
# EOF
# chmod 600 ~/.cdsapirc