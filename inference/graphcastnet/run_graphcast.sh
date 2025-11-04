export CUDA_VISIBLE_DEVICES=0

python -m ai_models \
   --download-assets --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/graphcast_assets graphcast --only-gpu

# python -m ai_models \
#    --input cds --date 20180101 --time 0000 graphcast \
#    --lead-time 1008 \
#    --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/graphcast \
#    --path /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/graphcast --only-gpu

