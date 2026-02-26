<div align="center">

# LCM-Dreamshaper-V7-rs

> Note: The tool automatically attempts to download a recompiled UNet RKNN from the HF repository `kautism/LCM_Dreamshaper_v7-RKNN-2.3.2` if present. If not found, it falls back to the original `whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0` RKNN. All model downloads are handled by the hf-hub library (no manual caching required).


**Fast text-to-image generation on Rockchip RK3588 NPU, powered by LCM Dreamshaper V7**

[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-link]
[![][github-contributors-shield]][github-contributors-link]
[![][last-commit-shield]][last-commit-link]
[![][license-shield]][license-link]

</div>

A Rust implementation of the [LCM Dreamshaper V7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) diffusion pipeline, optimised for the **Rockchip RK3588 NPU** via the RKNN runtime. Generate 512×512 images in a few seconds directly on an Orange Pi 5 / Rock 5 / NanoPi R6C or any other RK3588 board — no GPU required.

Supports both a **CLI mode** for one-shot image generation and a **serve mode** that exposes an **OpenAI-compatible `/v1/images/generations` endpoint**, so any OpenAI-compatible client can drive it out of the box.

---

## Features

- 🚀 **RK3588 NPU acceleration** — all three NPU cores engaged for the UNet
- ⚡ **LCM scheduling** — good-quality images in as few as 4 steps; 10 steps by default
- 🖼️ **512 × 512 PNG output**
- 🌐 **OpenAI-compatible REST API** (`POST /v1/images/generations`)
- 🔁 **Reproducible results** with optional seed; random seed when omitted
- 📦 **Auto-download** — RKNN models fetched from HuggingFace Hub on first run and cached

---

## Model Attribution

| Component | Source |
|---|---|
| Original model weights | [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) |
| ONNX conversion | [TheyCallMeHex/LCM-Dreamshaper-V7-ONNX](https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX) |
| RKNN models (text encoder, VAE) | [whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0](https://huggingface.co/whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0) |
| CLIP tokenizer | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |

> **Note on the UNet model:** The UNet in the HuggingFace repo above was compiled with rknn-toolkit2 **2.3.0** and crashes on systems with **librknnrt.so 2.3.2** (e.g. latest Orange Pi OS images). This project recompiles the UNet from the original ONNX weights using rknn-toolkit2 2.3.2. See [Building the UNet](#building-the-unet-for-rknnrt-232) below.

---

## Requirements

- **Hardware:** Rockchip RK3588 board (Orange Pi 5, Rock 5B, NanoPi R6C, …)
- **OS:** Linux aarch64 with RKNPU2 kernel driver (`/dev/dri/renderD*`)
- **Runtime:** `librknnrt.so` **2.3.2** (recommended)
- **User permission:** member of the `render` group (or run as root)
- **Rust toolchain:** stable (`rustup` recommended)


### RKNN runtime (librknnrt.so)

This project requires the Rockchip RKNN runtime library librknnrt.so to access the NPU. Installing the runtime typically requires root privileges and a platform-specific binary from your board vendor. If your distribution does not provide librknnrt.so, a copy can be downloaded from the Airockchip repository (raw aarch64 build):

https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

Installation notes:
- Installing system libraries requires root. Place the .so under `/usr/lib/` or `/lib/` (or the vendor-specified location) and run `ldconfig` as root.
- The binary will try to use whatever librknnrt.so is available on the system; for best results use version 2.3.2.
- You do not need to recompile any model to install the runtime; recompilation is only necessary when the precompiled RKNN model is incompatible with your runtime (see "Building the UNet" section).
- For NPU access also ensure the current user is in the `render` group (or run the tool as root): `sudo usermod -aG render $USER` and re-login.

We intentionally do not ship or auto-install librknnrt.so because installing system libraries requires root and platform-specific handling; please install the runtime yourself following the vendor instructions or the link above.

---

## Building the UNet for librknnrt 2.3.2

The UNet in the HF repo was compiled with an older toolkit and will **SIGSEGV** on librknnrt 2.3.2.
You must recompile it once and place it at `~/.cache/lcm-rs/unet_v232.rknn`.

```bash
# 1. Install rknn-toolkit2 2.3.2 (Python 3.11 recommended)
uv venv /tmp/rknn-venv --python 3.11
source /tmp/rknn-venv/bin/activate
pip install setuptools==75.0.0
pip install rknn-toolkit2==2.3.2   # or from your vendor package

# 2. Download the original ONNX UNet (~3.5 GB)
mkdir -p /tmp/unet_onnx
huggingface-cli download TheyCallMeHex/LCM-Dreamshaper-V7-ONNX \
    unet/model.onnx unet/model.onnx_data \
    --local-dir /tmp/unet_onnx

# 3. Convert
python3 - <<'EOF'
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588', float_dtype='float16', optimization_level=3)
rknn.load_onnx(
    '/tmp/unet_onnx/unet/model.onnx',
    inputs=['sample', 'timestep', 'encoder_hidden_states', 'timestep_cond'],
    input_size_list=[[1,4,64,64], [1], [1,77,768], [1,256]],
)
rknn.build(do_quantization=False)
rknn.export_rknn('/tmp/unet_rk3588_v232.rknn')
rknn.release()
EOF

# 4. Cache
mkdir -p ~/.cache/lcm-rs
cp /tmp/unet_rk3588_v232.rknn ~/.cache/lcm-rs/unet_v232.rknn
```

---

## Install

```bash
git clone https://github.com/darkautism/LCM-Dreamshaper-V7-rs.git
cd LCM-Dreamshaper-V7-rs/lcm-rs
cargo build --release
```

The binary is at `target/release/dreamshaper-cli`.

---

## CLI Usage

```
dreamshaper-cli [OPTIONS] [COMMAND]

Commands:
  generate   Generate an image from a text prompt (default)
  serve      Start an OpenAI-compatible HTTP server

Options:
  -p, --prompt          Text prompt [default: "a beautiful sunset over the ocean, photorealistic"]
  -s, --steps           Inference steps [default: 10]
  -g, --guidance-scale  Guidance scale [default: 7.5]
      --seed            Random seed (omit for a random seed each run)
  -o, --output          Output PNG path [default: output.png]
      --info-only       Print model tensor info and exit
```

### Examples

```bash
# Generate with default settings (random seed)
./dreamshaper-cli -p "a red panda in a bamboo forest, digital art"

# Reproducible result with fixed seed
./dreamshaper-cli -p "a futuristic city at night" --seed 1234 -o city.png

# Faster (lower quality) — 4 steps
./dreamshaper-cli -p "a mountain lake" -s 4 -o lake.png

# Explicit generate subcommand
./dreamshaper-cli generate -p "a cat sitting on a table" --seed 42
```

---

## Serve Mode (OpenAI-compatible API)

Start the HTTP server:

```bash
./dreamshaper-cli serve --host 0.0.0.0 --port 8080
```

### Endpoint

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/images/generations` | Generate image(s) — OpenAI Images API compatible |

### Request body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the image |
| `n` | integer | `1` | Number of images (only `1` currently supported) |
| `size` | string | `"512x512"` | Image size (only `"512x512"` supported) |
| `response_format` | string | `"b64_json"` | Response format (`"b64_json"`) |
| `seed` | integer\|null | `null` | Seed for reproducibility; omit for random |
| `steps` | integer | `10` | LCM inference steps (extension field) |
| `guidance_scale` | float | `7.5` | Guidance scale (extension field) |

### Response

```json
{
  "created": 1700000000,
  "data": [
    {
      "b64_json": "<base64-encoded PNG>",
      "seed": 3141592653
    }
  ]
}
```

### curl Examples

**Basic generation (random seed):**
```bash
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red panda in a bamboo forest, digital art"}' \
  | jq -r '.data[0].b64_json' | base64 -d > output.png
```

**Fixed seed for reproducibility:**
```bash
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic city at night, cyberpunk",
    "seed": 1234,
    "steps": 10,
    "guidance_scale": 7.5
  }' \
  | jq -r '.data[0].b64_json' | base64 -d > city.png
```

**Fast preview (4 steps):**
```bash
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a mountain lake at sunset", "steps": 4}' \
  | jq -r '.data[0].b64_json' | base64 -d > lake.png
```

**Check which seed was used:**
```bash
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a dragon"}' \
  | jq '{seed: .data[0].seed}'
```

---

## Performance

Measured on RK3588 (Orange Pi 5 Plus), all 3 NPU cores:

| Steps | Time |
|-------|------|
| 4     | ~12 s |
| 10    | ~25 s |

---

## Project Structure

```
LCM-Dreamshaper-V7-rs/
├── src/
│   ├── main.rs        # CLI entry point + subcommands
│   ├── pipeline.rs    # Shared generation pipeline (model loading, inference)
│   ├── serve.rs       # OpenAI-compatible HTTP server (axum)
│   ├── models.rs      # RKNN model wrappers (text encoder, UNet, VAE)
│   └── scheduler.rs   # LCM noise scheduler
└── Cargo.toml
```

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Support the Project

If this saves you time or helps your project, consider supporting continued development:

[![][ko-fi-shield]][ko-fi-link]
[![][paypal-shield]][paypal-link]

---

<!-- Link Definitions -->

[github-stars-shield]: https://img.shields.io/github/stars/darkautism/LCM-Dreamshaper-V7-rs?labelColor=black&style=flat-square&color=ffcb47
[github-stars-link]: https://github.com/darkautism/LCM-Dreamshaper-V7-rs
[github-issues-shield]: https://img.shields.io/github/issues/darkautism/LCM-Dreamshaper-V7-rs?labelColor=black&style=flat-square&color=ff80eb
[github-issues-link]: https://github.com/darkautism/LCM-Dreamshaper-V7-rs/issues
[github-contributors-shield]: https://img.shields.io/github/contributors/darkautism/LCM-Dreamshaper-V7-rs?color=c4f042&labelColor=black&style=flat-square
[github-contributors-link]: https://github.com/darkautism/LCM-Dreamshaper-V7-rs/graphs/contributors
[last-commit-shield]: https://img.shields.io/github/last-commit/darkautism/LCM-Dreamshaper-V7-rs?color=c4f042&labelColor=black&style=flat-square
[last-commit-link]: https://github.com/darkautism/LCM-Dreamshaper-V7-rs/commits/main
[license-shield]: https://img.shields.io/badge/license-MIT-white?labelColor=black&style=flat-square
[license-link]: https://github.com/darkautism/LCM-Dreamshaper-V7-rs/blob/main/LICENSE
[ko-fi-shield]: https://img.shields.io/badge/Ko--fi-F16061?style=for-the-badge&logo=ko-fi&logoColor=white
[ko-fi-link]: https://ko-fi.com/kautism
[paypal-shield]: https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white
[paypal-link]: https://paypal.me/kautism
