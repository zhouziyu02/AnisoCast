 
# python /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/fourcastnet/make_gt_from_era5.py \
#   --pl_dir  /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/S2S/pressure_level_1.5 \
#   --sfc_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/data/S2S/single_level_1.5 \
#   --start_time 2018-01-01T00:00:00 \
#   --year 2018 \
#   --steps 41 \
#   --levels_hpa 1000,925,850,700,600,500,400,300,250 \
#   --humidity specific \
#   --no_coarsen \
#   --out /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/gt_20180101_TVHW.npy


 
# python /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/fourcastnet/evaluate_fourcastnet.py \
#   --pred_npy /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/fourcastnetv2-small.npy \
#   --config_filepath /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/CIRT/configs/fourcastnetv2.yaml \
#   --output_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet \
#   --save_name fourcastnetv2.csv \
#   --skip_denorm


# python /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/fourcastnet/evaluate_fourcastnet.py \
#   --pred_npy /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/fourcastnetv2-small.npy \
#   --gt_npy   /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/gt_20180101_TVHW.npy \
#   --config_filepath /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/CIRT/configs/fourcastnetv2.yaml \
#   --output_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet \
#   --save_name fourcastnetv2_metrics.csv \
#   --skip_denorm


 

python /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/inference/fourcastnet/evaluate_fourcastnet.py \
  --pred_npy /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/fourcastnetv2-small_ztuv_u10v10.npy \
  --gt_npy   /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet/gt_20180101_TVHW_ztuv_u10v10.npy \
  --config_filepath /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/CIRT/configs/fourcastnetv2.yaml \
  --output_dir /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/fourcastnet \
  --save_name fourcastnetv2_metrics_ztuv_u10v10.csv \
  --skip_denorm
