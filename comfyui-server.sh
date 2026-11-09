#!/bin/bash
# ComfyUI 后端启动脚本 (headless, 供 Windows 前端远程连接)
set -x
LOG=/home/user/logs/comfyui/comfyui-server.log
TQDM=/home/user/logs/comfyui/tqdm.log
mkdir -p /home/user/logs/comfyui
# stdout 到主日志，stderr 到 tqdm 日志（分离 tqdm 刷新）
echo "=== $(date) COMFYUI START ===" >> $LOG
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
source /home/user/comfyui-venv/bin/activate

cd /home/user/ComfyUI
# 后端监听 0.0.0.0:8188, 供 Windows 前端访问
# --disable-auto-launch: 不自动打开浏览器
# NOTE: --use-ck-attention 触发 Sage Attention CUDA-only int8 路径，在 HIP 上炸
#       改用 PyTorch ROCm SDPA（H3 attention 走 standard scaled_dot_product_attention）
# NOTE: --bf16-unet 强制扩散模型(UNet)用 bf16，避免 fp16 中间值溢出导致花屏/发灰/噪点
#       （bf16 指数范围同 fp32）。H3 的 int8 权重不受此开关影响。
exec python main.py \
  --listen 0.0.0.0 \
  --port 8188 \
  --disable-auto-launch \
  --disable-dynamic-vram \
  --disable-async-offload \
  --bf16-unet \
  >> $LOG 2> >(tee $TQDM > /dev/null)
