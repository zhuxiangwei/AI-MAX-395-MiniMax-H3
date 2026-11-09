# AI-MAX-395-MiniMax-H3 — MiniMax H3 / FastH3 视频生成部署工程

AMD Ryzen AI Max+ 395（Strix Halo）128 GB 统一内存推理机上的 **MiniMax H3 / FastH3 v2 视频生成底座**：ComfyUI + PyTorch ROCm headless 服务、模型权重部署、官方工作流与实测容量基线。

仓库内容：**推理服务（systemd unit + 启动器）、模型部署要点、实测性能**。LLM 推理后端（Qwen3.8）见姊妹仓库 [AI-MAX-395-Qwen3.8](https://github.com/zhuxiangwei/AI-MAX-395-Qwen3.8)。

> 已脱敏的公开副本：路径为示例，无任何凭据。
>
> **占位符约定**：`/home/user` → 实际部署用户的主目录（systemd **user** unit 里也可用 `%h`）。

---

## 1. 部署前提（推理底座）

- **APU**：AMD Ryzen AI Max+ 395 w/ Radeon 8060S（**gfx1151**），16C/32T；128 GB LPDDR5X 统一内存（CPU/GPU 共享），推理可用约 124 GB
- **GTT 124 GB**，GRUB 内核参数：
  `amd_iommu=off amdgpu.gttsize=126976 processor.max_cstate=1 ttm.pages_limit=32505856 ttm.page_pool_size=32505856 amdgpu.no_system_mem_limit=1`
- **ROCm 7.14.1**（`/opt/rocm/core-7.14`）

⚠️ **`ttm.pages_limit` 必须与 `amdgpu.gttsize` 对齐**（页数 = GiB × 262144），另需 `amdgpu.no_system_mem_limit=1`。

⚠️ **H3/FastH3 与 LLM 后端抢 GTT**（视频峰值 ~50 GiB），不能同跑；统一内存下切换需重启整机才能彻底归还 GTT。

---

## 2. ComfyUI 运行时

生产版本组合（实测）：

| 组件 | 版本 |
|------|------|
| ComfyUI | **v0.43.2** |
| venv | `~/comfyui-venv`（Python 3.14.4） |
| torch | **2.12.0+rocm7.14.1** |
| 扩展 | comfy-kitchen 0.2.40 + comfy-aimdo 0.6.0 |

- ⚠️ **勿手敲 `pip install torch`**（会降级成 CUDA 版）；ROCm wheel 按 gfx1151 对应的官方渠道装。
- ⚠️ **勿用 `--use-ck-attention`**：触发 Sage Attention CUDA-only int8 路径，在 HIP 上崩。H3 attention 走 PyTorch ROCm SDPA（standard scaled_dot_product_attention）。
- ⚠️ 依赖 `comfyui-workflow-templates-media-assets-02==0.1.23`（模板缩略图包）PyPI 上没有，缺了不影响引擎。
- ⚠️ `rocm-smi` 在本机不可用 → 查显存用 `curl -s http://127.0.0.1:8188/system_stats`。

---

## 3. 模型：权重清单与路径映射

权重不入库（体积原因），按家族目录放置，`extra_model_paths.yaml` 挂到 ComfyUI（装到 `~/ComfyUI/extra_model_paths.yaml`）：

| 家族 | 文件 | GiB | 用途 |
|---|---|---:|---|
| fasth3 | `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | 20.6 | FastH3 8-Step V2，8 步蒸馏档（t2v/i2v/flf2v 生产主力） |
| minimax-h3 | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 | 底座 FL2VA，首尾帧 |
| minimax-h3 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 | 底座 Ref2VA，多模态参考（r2v） |
| minimax-h3 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 25.3 | H3 文本编码器（CLIPLoader type 必须选 `minimax`） |
| minimax-h3 | `minimax_h3_video_vae_fp16.safetensors` | 4.85 | 视频 VAE |
| minimax-h3 | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 | 音频 VAE（参考音频驱动生成必需） |

目录结构：`~/models/comfyui/{minimax-h3,fasth3}/{diffusion_models,text_encoders,vae}`。

⚠️ `extra_model_paths.yaml` 挂多目录**必须用 YAML 多行字符串（`|-`）**，列表格式会崩；重装 ComfyUI 必丢该文件，注意备份。

---

## 4. 服务

- `comfyui-server.service`（user 级）→ `comfyui-server.sh` → venv 内 `python main.py`
- `ExecStartPre=rocm-wait-devices.sh`：开机后 `/dev/kfd` 约 +3s、`/dev/dri/renderD128` 约 +6s 才由 udev 创建；user 级 systemd 不支持 device unit，只能显式轮询，否则 HIP 初始化 abort。
- 启动参数：`--listen 0.0.0.0 --port 8188 --disable-auto-launch --disable-dynamic-vram --disable-async-offload --bf16-unet`
  - `--bf16-unet`：强制 UNet 用 bf16，避免 fp16 中间值溢出导致花屏/发灰（bf16 指数范围同 fp32）；H3 的 int8 权重不受影响。

⚠️ **ComfyUI 绑 `0.0.0.0` 且无鉴权**，隔离靠可信局域网 + NAT 无端口映射。**不要接到不可信网络，也不要做 8188 端口映射。**

```bash
systemctl --user daemon-reload
systemctl --user enable --now comfyui-server
curl -s http://127.0.0.1:8188/system_stats       # 就绪与显存
```

---

## 5. 官方工作流

`video_fastvideo_fasth3_t2v.json` / `video_fastvideo_fasth3_i2v.json` 是 FastH3 官方工作流镜像（装到 `~/ComfyUI/user/default/workflows/`），可直接在 ComfyUI 前端加载。

**FastH3 VSA 要点**：`BlockSparseAttention` 是动态组合，`selection="vsa"` 必须带 **`selection.keep_percent=10.0`**（FastH3-VSA 按此训练），漏了报 `required_input_missing`。

---

## 6. 模型用法关键点（踩坑实录）

1. **任务模式与步数分档**：t2v / i2v / flf2v 走 `MiniMaxH3ImageToVideo`，生产用 FastH3 v2 **8 步**蒸馏档；r2v 走 `MiniMaxH3ReferenceToVideo`（≤9 图 / ≤3 视频 / ≤3 音频），用 Ref2VA 底座 **20 步**（非蒸馏，不能一律 8 步）。
2. **H3 不用 CFG**：官方模板走 BasicGuider（无 negative、cfg=1），不是 CFGGuider。
3. **帧数必须落在 17k+5 网格**：24fps，5 秒 → 124 帧；训练范围 124–362（约 5–15 秒）。直接填 120 会被静默向上吸附。画幅边长须为 32 倍数。
4. **H3 是原生音画一体模型**：`MiniMaxH3ReferenceToVideo` 有 `ref_audios` / `ref_video_audios` 输入，参考音频作为条件输入参与生成，音轨在同一次生成里产出。正确工序是**先备好配音 → 作为参考音频喂进 H3**，而不是事后对口型。参考音频是条件输入，不等于口型验证（旁白/反应镜头不产生口型）。
5. **autogrow 键名**：`ref_audios.ref_audio_0`（prefix + 从 0 起的索引，max=3）；`audio_vae` 必须接 `minimax_h3_audio_vae_fp32.safetensors`，参考音频才真正驱动生成。
6. **配音时长必须 ≤ 生成片段时长**（124 帧 = 5.125s）；超了先吸附帧数（如 6.05s 配音 → 158 帧 / 6.542s）。
7. **上传通道**：ComfyUI 0.43.2 **没有 `/upload/audio`**（只有 /upload/image 与 /upload/mask）；音频也走 `/upload/image`，但表单**必须带 `type=input`**，否则落进 `output/`，而 `LoadAudio` 只从 `input/` 列文件。
8. **缓存加速**：服务端**没有 `MiniMaxH3Cache`** 节点；同类可用 EasyCache / LazyCache 等，算法不同，启用前需自行 A/B。
9. **`/history` 服务端重启即清**：任务状态要外部持久化，别依赖它做断点续跑。

---

## 7. 实测容量（部署事实）

8 步 / 124 帧 / FastH3 v2 / 同一提示词与 seed，三档画幅：

| 画幅 | MP | 端到端耗时 | 显存峰值 |
|------|---:|---:|---:|
| 576×1024 | 0.590 | 436.5 s | 52.24 GiB |
| 768×1344 | 1.032 | 933.4 s | 53.27 GiB |
| 864×1536 | 1.327 | 1441.5 s | 52.60 GiB |

**峰值显存 52–53 GiB 不随画幅变化**（远低于 124 GiB），耗时随像素面积**超线性**增长 ⇒ 容量瓶颈是**时间**不是显存；用时间换质量时优先控画幅。口径：1s 轮询 `/system_stats`，峰值是「总占用」（含约 49 GiB 常驻模型缓存）而非单任务增量。三条产物均经 ffprobe 校验 124 帧 / 5.167s / h264+aac，全解码零错误。

参考量级：768×1344 × 124 帧 8 步 ≈ 11 min/镜。

---

## 8. 文件清单

| 文件 | 部署位置 | 用途 |
|------|---------|------|
| `comfyui-server.service` / `comfyui-server.sh` | `~/.config/systemd/user/` / `~/scripts/` | 服务与启动器 |
| `rocm-wait-devices.sh` | `~/scripts/` | ExecStartPre：udev 设备节点延迟兜底 |
| `extra_model_paths.yaml` | `~/ComfyUI/` | 权重家族目录映射 |
| `video_fastvideo_fasth3_t2v.json` / `_i2v.json` | `~/ComfyUI/user/default/workflows/` | 官方工作流镜像 |

## 9. 许可证提示

MiniMax H3 / FastH3 权重适用 **MiniMax H3 社区许可证**（地域 / 营收 / 署名 / 二次训练限制），商用交付前逐项核对上游 LICENSE 与数据集条款；本仓库只含部署代码与官方工作流镜像，不分发权重。

---

*硬件：AMD Ryzen AI Max+ 395 · 128 GB 统一内存（GTT 124G）· ROCm 7.14.1 · ComfyUI 0.43.2 · torch 2.12.0+rocm7.14.1 · MiniMax H3 / FastH3 v2（int8）*