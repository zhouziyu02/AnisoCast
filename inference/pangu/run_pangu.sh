export CUDA_VISIBLE_DEVICES=0


# python - <<'PY'
# import json
# from pathlib import Path
# cfg = {
#     "url"   : "https://api.ecmwf.int/v1",
#     "key"   : "52bb85222e1309edfd5e1e79766a69e5",
#     "email" : "ziyuzhou30@gmail.com"
# }
# p = Path.home() / ".ecmwfapirc"
# p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
# p.chmod(0o600)
# print("✅ 写入完成：", p, "权限=600")
# PY

# python -m ai_models panguweather \
#   --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/panguweather_assets \
#   --input cds \
#   --date 20180101 \
#   --time 0000 \
#   --leadtime 42d

# python -m ai_models \
#   --input cds --date 20180101 --time 0000 panguweather --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/panguweather_assets/pangu_weather_6.onnx --only-gpu

# python -m ai_models \
#    --input cds --date 20180101 --time 0000 fourcastnetv2-small \
#    --lead-time 1008 \
#    --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/fourcastnetv2_assets --only-gpu

#    --lead-time 1008 \
#   --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/panguweather_assets/pangu_weather_6.onnx \
#   --path /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/Pangu_inference


# python -m ai_models   --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/fourcastnetv2_assets fourcastnetv2

# python -m ai_models \
#    --download-assets --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/panguweather_assets panguweather --only-gpu

python -m ai_models \
   --input cds --date 20180101 --time 0000 panguweather \
   --lead-time 1008 \
   --assets /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/panguweather_assets \
   --path /mnt/bn/gec-scl-ltm-forecast/zhouziyu/CirT/results/pangu --only-gpu