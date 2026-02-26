use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use image::{ImageBuffer, Rgb};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use tokenizers::Tokenizer;

use crate::models::{RknnModel, UNetModel};
use crate::scheduler::LcmScheduler;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

pub const LATENT_H: usize = 64;
pub const LATENT_W: usize = 64;
pub const LATENT_C: usize = 4;
pub const LATENT_SIZE: usize = LATENT_C * LATENT_H * LATENT_W;

pub const IMG_H: usize = 512;
pub const IMG_W: usize = 512;

const MAX_SEQ_LEN: usize = 77;
const TEXT_EMB_DIM: usize = 768;
const GUIDANCE_EMB_DIM: usize = 256;

/// VAE scaling factor from vae_decoder/config.json
const VAE_SCALE: f32 = 0.18215;

// ─────────────────────────────────────────────────────────────────────────────
// Generate request / result
// ─────────────────────────────────────────────────────────────────────────────

pub struct GenerateRequest {
    pub prompt: String,
    pub steps: usize,
    pub guidance_scale: f32,
    /// None → random seed chosen at runtime
    pub seed: Option<u64>,
}

pub struct GenerateResult {
    /// Raw PNG bytes
    pub png_bytes: Vec<u8>,
    /// Actual seed used (useful when caller requested random seed)
    pub seed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Model download helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn download_models() -> Result<(
    std::path::PathBuf, // text_encoder
    std::path::PathBuf, // unet
    std::path::PathBuf, // vae_decoder
    std::path::PathBuf, // tokenizer
)> {
    eprintln!("📥 Downloading models from HuggingFace (cached after first run)...");
    let api = Api::new().context("Failed to create HF Hub API")?;
    let lcm_repo = api.model("whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0".to_string());

    let text_encoder = lcm_repo
        .get("text_encoder/model.rknn")
        .context("Failed to download text_encoder/model.rknn")?;
    eprintln!("  text_encoder: {}", text_encoder.display());

    let unet = unet_model_path()?;
    eprintln!("  unet: {}", unet.display());

    let vae_decoder = lcm_repo
        .get("vae_decoder/model.rknn")
        .context("Failed to download vae_decoder/model.rknn")?;
    eprintln!("  vae_decoder: {}", vae_decoder.display());

    let tokenizer = download_clip_tokenizer()?;
    eprintln!("  tokenizer: {}", tokenizer.display());

    Ok((text_encoder, unet, vae_decoder, tokenizer))
}

/// Return path to the UNet RKNN model compatible with librknnrt.so 2.3.2.
/// Prefers locally recompiled model over the HF-hosted 2.3.0-compiled model.
fn unet_model_path() -> Result<std::path::PathBuf> {
    let local_path = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("lcm-rs")
        .join("unet_v232.rknn");
    if local_path.exists() {
        eprintln!("  (using locally recompiled UNet for librknnrt 2.3.2)");
        return Ok(local_path);
    }
    eprintln!("  WARNING: recompiled UNet not found at {}", local_path.display());
    eprintln!("  Falling back to HF model (may crash on librknnrt.so 2.3.2)");
    let api = Api::new().context("Failed to create HF Hub API")?;
    let lcm_repo = api.model("whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0".to_string());
    lcm_repo.get("unet/model.rknn").context("Failed to download unet/model.rknn")
}

/// Download CLIP tokenizer.json and cache it under ~/.cache/lcm-rs/.
fn download_clip_tokenizer() -> Result<std::path::PathBuf> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("lcm-rs");
    std::fs::create_dir_all(&cache_dir).context("Failed to create cache dir")?;

    let tokenizer_path = cache_dir.join("clip_tokenizer.json");
    if tokenizer_path.exists() {
        return Ok(tokenizer_path);
    }

    eprintln!("  Downloading CLIP tokenizer.json...");
    let url = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json";
    let bytes = reqwest::blocking::get(url)
        .context("Failed to fetch tokenizer.json")?
        .error_for_status()
        .context("Bad HTTP status for tokenizer.json")?
        .bytes()
        .context("Failed to read tokenizer.json bytes")?;

    std::fs::write(&tokenizer_path, &bytes).context("Failed to write tokenizer.json")?;
    Ok(tokenizer_path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline — holds all loaded models
// ─────────────────────────────────────────────────────────────────────────────

pub struct Pipeline {
    text_encoder: RknnModel,
    unet: UNetModel,
    vae_decoder: RknnModel,
    tokenizer: Tokenizer,
}

// RKNN contexts live in process memory without cross-thread sharing.
// We guard Pipeline with a Mutex so only one thread calls into RKNN at a time.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Pipeline {
    pub fn load() -> Result<Self> {
        let (te_path, unet_path, vae_path, tok_path) = download_models()?;

        eprintln!("🔧 Loading RKNN models...");
        let text_encoder = RknnModel::load(&te_path).context("Load text_encoder")?;
        let unet = UNetModel::load(&unet_path).context("Load unet")?;
        let vae_decoder = RknnModel::load(&vae_path).context("Load vae_decoder")?;
        eprintln!("  All models loaded");

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { text_encoder, unet, vae_decoder, tokenizer })
    }

    pub fn print_info(&self) -> Result<()> {
        eprintln!("\n── text_encoder ──");
        self.text_encoder.print_info()?;
        eprintln!("\n── unet ──");
        self.unet.print_info()?;
        eprintln!("\n── vae_decoder ──");
        self.vae_decoder.print_info()?;
        Ok(())
    }

    pub fn generate(&self, req: GenerateRequest) -> Result<GenerateResult> {
        // Tokenize
        let input_ids = tokenize(&self.tokenizer, &req.prompt)?;
        let n_tokens = input_ids.iter().position(|&x| x == 49407).unwrap_or(MAX_SEQ_LEN);
        eprintln!("🔤 Prompt: \"{}\" ({} tokens)", req.prompt, n_tokens.saturating_sub(1));

        // Text encoding
        eprintln!("📝 Running text encoder...");
        let text_emb = self.text_encoder
            .run_with_int32_inputs(&[(0, &input_ids)], &[])
            .context("Text encoder failed")?;
        let expected_emb = MAX_SEQ_LEN * TEXT_EMB_DIM;
        let text_emb_flat: Vec<f32> = text_emb[..expected_emb.min(text_emb.len())].to_vec();

        // Scheduler
        let mut scheduler = LcmScheduler::new();
        scheduler.set_timesteps(req.steps);
        eprintln!("📅 Timesteps ({} steps): {:?}", req.steps, scheduler.timesteps);

        // Random latent
        let seed = req.seed.unwrap_or_else(rand::random::<u64>);
        eprintln!("🎲 Seed: {}", seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let mut latent_nchw: Vec<f32> = (0..LATENT_SIZE).map(|_| normal.sample(&mut rng)).collect();

        // Guidance embedding
        let ts_cond = guidance_scale_embedding(req.guidance_scale, GUIDANCE_EMB_DIM);

        // Denoise loop
        eprintln!("🎨 Denoising ({} steps)...", req.steps);
        let timesteps = scheduler.timesteps.clone();
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            eprint!("  step {}/{} (t={}) ...", step_idx + 1, req.steps, timestep);
            let latent_nhwc = nchw_to_nhwc(&latent_nchw, LATENT_C, LATENT_H, LATENT_W);
            let noise_pred = self.unet
                .run(&latent_nhwc, timestep as i64, &text_emb_flat, &ts_cond)
                .with_context(|| format!("UNet step {step_idx} failed"))?;
            let (prev_latent, _denoised) =
                scheduler.step(&noise_pred, timestep, &latent_nchw, step_idx, &mut rng);
            latent_nchw = prev_latent;
            eprintln!(" done");
        }

        // VAE decode
        eprintln!("🖼️  Decoding with VAE...");
        let scaled_nchw: Vec<f32> = latent_nchw.iter().map(|&v| v / VAE_SCALE).collect();
        let scaled_nhwc = nchw_to_nhwc(&scaled_nchw, LATENT_C, LATENT_H, LATENT_W);
        let pixels = self.vae_decoder
            .run_f32(&[(0, &scaled_nhwc)])
            .context("VAE decoder failed")?;

        // Encode to PNG
        let img = vae_output_to_image(&pixels, IMG_H, IMG_W);
        let mut png_bytes: Vec<u8> = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode PNG")?;

        eprintln!("✅ Generation complete ({} bytes PNG)", png_bytes.len());
        Ok(GenerateResult { png_bytes, seed })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn tokenize(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let ids = encoding.get_ids();
    let mut input_ids = vec![0i32; MAX_SEQ_LEN];
    input_ids[0] = 49406; // BOS
    let content_len = ids.len().min(MAX_SEQ_LEN - 2);
    for i in 0..content_len {
        input_ids[1 + i] = ids[i] as i32;
    }
    let eos_pos = 1 + content_len;
    if eos_pos < MAX_SEQ_LEN {
        input_ids[eos_pos] = 49407; // EOS
    }
    Ok(input_ids)
}

fn guidance_scale_embedding(guidance_scale: f32, embedding_dim: usize) -> Vec<f32> {
    let w = guidance_scale - 1.0;
    let half_dim = embedding_dim / 2;
    let log_10000 = (10000.0f32).ln();
    let mut emb = vec![0.0f32; embedding_dim];
    for i in 0..half_dim {
        let freq = (-(log_10000 * i as f32 / ((half_dim as f32) - 1.0))).exp() * w;
        emb[i] = freq.sin();
        emb[half_dim + i] = freq.cos();
    }
    emb
}

/// Convert NCHW [1, C, H, W] → NHWC [1, H, W, C].
pub fn nchw_to_nhwc(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let nchw = ci * h * w + hi * w + wi;
                let nhwc = hi * w * c + wi * c + ci;
                out[nhwc] = data[nchw];
            }
        }
    }
    out
}

/// Convert VAE output (NCHW [1, 3, H, W], range [-1, 1]) to RGB image.
pub fn vae_output_to_image(data: &[f32], h: usize, w: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(w as u32, h as u32);
    let channel_size = h * w;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = ((data[i].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            let g = ((data[channel_size + i].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            let b = ((data[2 * channel_size + i].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img
}
