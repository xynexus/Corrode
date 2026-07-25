use crate::guide::{normalize_tools, restore_tool_names, JsonGuide};
use crate::tokenizer::{NeedleTokenizer, EOS_ID, PAD_ID, TOOLS_ID};
use anyhow::{bail, Context, Result};
use candle_core::{safetensors, DType, Device, Tensor, D};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub pad_token_id: u32,
    pub rope_theta: f32,
    pub dtype: String,
    pub activation: String,
    #[serde(default)]
    pub num_memory_slots: usize,
    #[serde(default)]
    pub dropout_rate: f32,
    #[serde(default)]
    pub contrastive_dim: usize,
    #[serde(default = "default_no_feedforward")]
    pub no_feedforward: bool,
}

fn default_no_feedforward() -> bool {
    true
}

impl TransformerConfig {
    pub fn total_layers(&self) -> usize {
        self.num_encoder_layers + self.num_decoder_layers
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }
}

#[derive(Debug, Clone)]
pub struct Assets {
    pub dir: PathBuf,
    pub weights: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl Assets {
    pub fn resolve(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let weights = dir.join("model.safetensors");
        let config = dir.join("config.json");
        let tokenizer = if dir.join("needle.model").exists() {
            dir.join("needle.model")
        } else {
            dir.join("tokenizer").join("needle.model")
        };
        for path in [&weights, &config, &tokenizer] {
            if !path.exists() {
                bail!(
                    "missing asset {}; run `python3 scripts/export_needle_checkpoint.py --checkpoint needle-weights/needle.pkl --tokenizer needle-weights/tokenizer/needle.model --out assets/needle`",
                    path.display()
                );
            }
        }
        Ok(Self {
            dir,
            weights,
            config,
            tokenizer,
        })
    }
}

#[derive(Debug)]
pub struct NeedleModel {
    pub config: TransformerConfig,
    device: Device,
    tensors: HashMap<String, Tensor>,
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_gen_len: usize,
    pub max_enc_len: usize,
    pub guided: bool,
    pub normalize: bool,
    pub guide_fast_forward: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_gen_len: 512,
            max_enc_len: 1024,
            guided: true,
            normalize: true,
            guide_fast_forward: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateStats {
    pub output: String,
    pub generated_tokens: usize,
    pub elapsed: Duration,
}

impl NeedleModel {
    pub fn load(assets: &Assets) -> Result<Self> {
        let config: TransformerConfig = serde_json::from_slice(
            &fs::read(&assets.config)
                .with_context(|| format!("reading {}", assets.config.display()))?,
        )
        .with_context(|| format!("parsing {}", assets.config.display()))?;
        let device = Device::Cpu;
        let loaded = safetensors::load(&assets.weights, &device)
            .with_context(|| format!("loading {}", assets.weights.display()))?;
        let mut tensors = HashMap::with_capacity(loaded.len());
        for (name, tensor) in loaded {
            tensors.insert(name, tensor.to_dtype(DType::F32)?);
        }
        Ok(Self {
            config,
            device,
            tensors,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn parameter_count(&self) -> usize {
        self.tensors.values().map(Tensor::elem_count).sum()
    }

    pub fn encode_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        let input = Tensor::from_vec(tokens.to_vec(), tokens.len(), &self.device)?;
        let x = (self.embedding()?.embedding(&input)?.reshape((
            1,
            tokens.len(),
            self.config.d_model,
        ))? * (self.config.d_model as f64).sqrt())?;
        let mut x = x;
        for layer in 0..self.config.num_encoder_layers {
            x = self.encoder_layer(layer, &x)?;
        }
        self.rms_norm(&x, &self.tensor("encoder/final_norm/scale")?)
    }

    pub fn decode_logits(&self, decoder_tokens: &[u32], encoder_out: &Tensor) -> Result<Tensor> {
        let cross_cache = self.build_cross_cache(encoder_out)?;
        self.decode_logits_with_cache(decoder_tokens, &cross_cache)
    }

    fn decode_logits_with_cache(
        &self,
        decoder_tokens: &[u32],
        cross_cache: &CrossAttentionCache,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(decoder_tokens.to_vec(), decoder_tokens.len(), &self.device)?;
        let x = (self.embedding()?.embedding(&input)?.reshape((
            1,
            decoder_tokens.len(),
            self.config.d_model,
        ))? * (self.config.d_model as f64).sqrt())?;
        let mut x = x;
        for layer in 0..self.config.num_decoder_layers {
            x = self.decoder_layer_cached(layer, &x, &cross_cache.layers[layer])?;
        }
        let x = self.rms_norm(&x, &self.tensor("decoder/ZCRMSNorm_0/scale")?)?;
        linear(&x, &self.embedding()?.t()?)
    }

    fn decode_next_logits_cached(
        &self,
        token: u32,
        cross_cache: &CrossAttentionCache,
        self_cache: &mut DecoderSelfAttentionCache,
    ) -> Result<Tensor> {
        self.decode_tokens_logits_cached(&[token], cross_cache, self_cache)?
            .get(0)
            .map_err(Into::into)
    }

    fn decode_tokens_logits_cached(
        &self,
        tokens: &[u32],
        cross_cache: &CrossAttentionCache,
        self_cache: &mut DecoderSelfAttentionCache,
    ) -> Result<Tensor> {
        if tokens.is_empty() {
            bail!("cached decode requires at least one token");
        }
        let input = Tensor::from_vec(tokens.to_vec(), tokens.len(), &self.device)?;
        let x = (self.embedding()?.embedding(&input)?.reshape((
            1,
            tokens.len(),
            self.config.d_model,
        ))? * (self.config.d_model as f64).sqrt())?;
        let mut x = x;
        for layer in 0..self.config.num_decoder_layers {
            x = self.decoder_layer_incremental(
                layer,
                &x,
                &cross_cache.layers[layer],
                &mut self_cache.layers[layer],
            )?;
        }
        let x = self.rms_norm(&x, &self.tensor("decoder/ZCRMSNorm_0/scale")?)?;
        linear(&x, &self.embedding()?.t()?)?
            .get(0)
            .map_err(Into::into)
    }

    pub fn generate(
        &self,
        tokenizer: &NeedleTokenizer,
        query: &str,
        tools: &str,
        max_gen_len: usize,
        guided: bool,
        normalize: bool,
    ) -> Result<String> {
        let mut options = GenerateOptions {
            max_gen_len,
            guided,
            normalize,
            ..GenerateOptions::default()
        };
        options.max_enc_len = self.config.max_seq_len;
        Ok(self
            .generate_with_options(tokenizer, query, tools, &options)?
            .output)
    }

    pub fn generate_with_options(
        &self,
        tokenizer: &NeedleTokenizer,
        query: &str,
        tools: &str,
        options: &GenerateOptions,
    ) -> Result<GenerateStats> {
        let started = Instant::now();
        let (tools, name_map) = if options.normalize {
            normalize_tools(tools)?
        } else {
            (tools.to_string(), HashMap::new())
        };
        let max_enc_len = options.max_enc_len.min(self.config.max_seq_len);
        let enc_tokens = build_encoder_input(tokenizer, query, &tools, max_enc_len)?;
        let encoder_out = self.encode_tokens(&enc_tokens)?;
        let cross_cache = self.build_cross_cache(&encoder_out)?;
        let mut guide = if options.guided {
            Some(JsonGuide::new(&tools, tokenizer)?)
        } else {
            None
        };
        let mut generated = Vec::new();
        let mut self_cache = DecoderSelfAttentionCache::new(self.config.num_decoder_layers);
        let mut logits = self.decode_next_logits_cached(EOS_ID, &cross_cache, &mut self_cache)?;
        let mut stopped = false;

        while !stopped && generated.len() < options.max_gen_len.saturating_sub(1) {
            if options.guide_fast_forward {
                if let Some(guide) = &mut guide {
                    let forced = guide.unique_fast_forward_tokens();
                    if !forced.is_empty() {
                        let mut advance = Vec::new();
                        for token in forced {
                            if generated.len() >= options.max_gen_len.saturating_sub(1) {
                                break;
                            }
                            guide.update(token);
                            if token == EOS_ID {
                                stopped = true;
                                break;
                            }
                            generated.push(token);
                            advance.push(token);
                        }
                        if !advance.is_empty() {
                            let forced_logits = self.decode_tokens_logits_cached(
                                &advance,
                                &cross_cache,
                                &mut self_cache,
                            )?;
                            logits = forced_logits.get(advance.len() - 1)?;
                        }
                        continue;
                    }
                }
            }

            let row = logits.to_vec1::<f32>()?;
            let mut row = row;
            if let Some(guide) = &guide {
                guide.mask_logits(&mut row);
            }
            let next = argmax(&row) as u32;
            if let Some(guide) = &mut guide {
                guide.update(next);
            }
            if next == EOS_ID {
                break;
            }
            generated.push(next);
            logits = self.decode_next_logits_cached(next, &cross_cache, &mut self_cache)?;
        }

        let mut text = tokenizer.decode(&generated)?.trim_start().to_string();
        if let Some(rest) = text.strip_prefix("<tool_call>") {
            text = rest.trim_start().to_string();
        }
        let output = restore_tool_names(&text, &name_map);
        Ok(GenerateStats {
            output,
            generated_tokens: generated.len(),
            elapsed: started.elapsed(),
        })
    }

    pub fn debug_probe(
        &self,
        tokenizer: &NeedleTokenizer,
        query: &str,
        tools: &str,
    ) -> Result<DebugProbe> {
        let (tools, _) = normalize_tools(tools)?;
        let enc_tokens = build_encoder_input(tokenizer, query, &tools, self.config.max_seq_len)?;
        let input = Tensor::from_vec(enc_tokens.clone(), enc_tokens.len(), &self.device)?;
        let embedding = (self.embedding()?.embedding(&input)?.reshape((
            1,
            enc_tokens.len(),
            self.config.d_model,
        ))? * (self.config.d_model as f64).sqrt())?;

        let enc0_norm = self.rms_norm(
            &embedding,
            &self.layer_tensor("encoder/layers/EncoderBlock_0/ZCRMSNorm_0/scale", 0)?,
        )?;
        let encoder_layer0_attention = self.attention(
            &enc0_norm,
            &enc0_norm,
            AttentionWeights {
                prefix: "encoder/layers/EncoderBlock_0/self_attn",
                layer: 0,
                causal: false,
                rope: true,
            },
        )?;
        let encoder_final = self.encode_tokens(&enc_tokens)?;
        let cross_cache = self.build_cross_cache(&encoder_final)?;
        let decoder_logits = self.decode_logits_with_cache(&[EOS_ID], &cross_cache)?;

        Ok(DebugProbe {
            encoder_tokens: enc_tokens.len(),
            embedding: TensorProbe::from_tensor(&embedding, 8)?,
            encoder_layer0_attention: TensorProbe::from_tensor(&encoder_layer0_attention, 8)?,
            encoder_final: TensorProbe::from_tensor(&encoder_final, 8)?,
            decoder_logits0: TensorProbe::from_tensor(&decoder_logits.get(0)?.get(0)?, 8)?,
        })
    }

    fn encoder_layer(&self, layer: usize, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.rms_norm(
            x,
            &self.layer_tensor("encoder/layers/EncoderBlock_0/ZCRMSNorm_0/scale", layer)?,
        )?;
        let attn = self.attention(
            &x_norm,
            &x_norm,
            AttentionWeights {
                prefix: "encoder/layers/EncoderBlock_0/self_attn",
                layer,
                causal: false,
                rope: true,
            },
        )?;
        let gate = self.layer_scalar("encoder/layers/EncoderBlock_0/attn_gate", layer)?;
        Ok(residual.broadcast_add(&(attn * sigmoid(gate) as f64)?)?)
    }

    fn decoder_layer_cached(
        &self,
        layer: usize,
        x: &Tensor,
        cross_cache: &ProjectedKv,
    ) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.rms_norm(
            x,
            &self.layer_tensor("decoder/layers/DecoderBlock_0/ZCRMSNorm_0/scale", layer)?,
        )?;
        let self_attn = self.attention(
            &x_norm,
            &x_norm,
            AttentionWeights {
                prefix: "decoder/layers/DecoderBlock_0/self_attn",
                layer,
                causal: true,
                rope: true,
            },
        )?;
        let self_gate = self.layer_scalar("decoder/layers/DecoderBlock_0/self_attn_gate", layer)?;
        let x = residual.broadcast_add(&(self_attn * sigmoid(self_gate) as f64)?)?;

        let residual = &x;
        let x_norm = self.rms_norm(
            &x,
            &self.layer_tensor("decoder/layers/DecoderBlock_0/ZCRMSNorm_1/scale", layer)?,
        )?;
        let cross_attn = self.attention_with_projected_kv(
            &x_norm,
            cross_cache,
            "decoder/layers/DecoderBlock_0/cross_attn",
            layer,
            false,
        )?;
        let cross_gate =
            self.layer_scalar("decoder/layers/DecoderBlock_0/cross_attn_gate", layer)?;
        Ok(residual.broadcast_add(&(cross_attn * sigmoid(cross_gate) as f64)?)?)
    }

    fn decoder_layer_incremental(
        &self,
        layer: usize,
        x: &Tensor,
        cross_cache: &ProjectedKv,
        self_cache: &mut LayerSelfAttentionCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.rms_norm(
            x,
            &self.layer_tensor("decoder/layers/DecoderBlock_0/ZCRMSNorm_0/scale", layer)?,
        )?;
        let self_attn = self.self_attention_incremental(
            &x_norm,
            self_cache,
            "decoder/layers/DecoderBlock_0/self_attn",
            layer,
        )?;
        let self_gate = self.layer_scalar("decoder/layers/DecoderBlock_0/self_attn_gate", layer)?;
        let x = residual.broadcast_add(&(self_attn * sigmoid(self_gate) as f64)?)?;

        let residual = &x;
        let x_norm = self.rms_norm(
            &x,
            &self.layer_tensor("decoder/layers/DecoderBlock_0/ZCRMSNorm_1/scale", layer)?,
        )?;
        let cross_attn = self.attention_with_projected_kv(
            &x_norm,
            cross_cache,
            "decoder/layers/DecoderBlock_0/cross_attn",
            layer,
            false,
        )?;
        let cross_gate =
            self.layer_scalar("decoder/layers/DecoderBlock_0/cross_attn_gate", layer)?;
        Ok(residual.broadcast_add(&(cross_attn * sigmoid(cross_gate) as f64)?)?)
    }

    fn attention(
        &self,
        q_input: &Tensor,
        kv_input: &Tensor,
        weights: AttentionWeights,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let (_, q_len, _) = q_input.dims3()?;
        let (_, kv_len, _) = kv_input.dims3()?;

        let q = linear(
            q_input,
            &self.layer_tensor(&format!("{}/q_proj/kernel", weights.prefix), weights.layer)?,
        )?;
        let k = linear(
            kv_input,
            &self.layer_tensor(&format!("{}/k_proj/kernel", weights.prefix), weights.layer)?,
        )?;
        let v = linear(
            kv_input,
            &self.layer_tensor(&format!("{}/v_proj/kernel", weights.prefix), weights.layer)?,
        )?;

        let q = q
            .reshape((1, q_len, cfg.num_heads, head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((1, kv_len, cfg.num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((1, kv_len, cfg.num_kv_heads, head_dim))?
            .transpose(1, 2)?;

        let q = self.rms_norm(
            &q,
            &self.layer_tensor(&format!("{}/q_norm/scale", weights.prefix), weights.layer)?,
        )?;
        let k = self.rms_norm(
            &k,
            &self.layer_tensor(&format!("{}/k_norm/scale", weights.prefix), weights.layer)?,
        )?;
        let k = repeat_kv(&k, cfg.num_heads / cfg.num_kv_heads)?;
        let v = repeat_kv(&v, cfg.num_heads / cfg.num_kv_heads)?;

        let (q, k) = if weights.rope {
            (
                apply_rope(&q, cfg.rope_theta)?,
                apply_rope(&k, cfg.rope_theta)?,
            )
        } else {
            (q, k)
        };

        let mut scores = (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
        if weights.causal {
            scores = scores.broadcast_add(&causal_bias(q_len, &self.device)?)?;
        }
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((1, q_len, cfg.d_model))?;
        linear(
            &out,
            &self.layer_tensor(
                &format!("{}/out_proj/kernel", weights.prefix),
                weights.layer,
            )?,
        )
    }

    fn build_cross_cache(&self, encoder_out: &Tensor) -> Result<CrossAttentionCache> {
        let mut layers = Vec::with_capacity(self.config.num_decoder_layers);
        for layer in 0..self.config.num_decoder_layers {
            layers.push(self.project_kv(
                encoder_out,
                "decoder/layers/DecoderBlock_0/cross_attn",
                layer,
                false,
            )?);
        }
        Ok(CrossAttentionCache { layers })
    }

    fn project_kv(
        &self,
        kv_input: &Tensor,
        prefix: &str,
        layer: usize,
        rope: bool,
    ) -> Result<ProjectedKv> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let (_, kv_len, _) = kv_input.dims3()?;
        let k = linear(
            kv_input,
            &self.layer_tensor(&format!("{prefix}/k_proj/kernel"), layer)?,
        )?;
        let v = linear(
            kv_input,
            &self.layer_tensor(&format!("{prefix}/v_proj/kernel"), layer)?,
        )?;
        let k = k
            .reshape((1, kv_len, cfg.num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((1, kv_len, cfg.num_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let k = self.rms_norm(
            &k,
            &self.layer_tensor(&format!("{prefix}/k_norm/scale"), layer)?,
        )?;
        let k = repeat_kv(&k, cfg.num_heads / cfg.num_kv_heads)?;
        let v = repeat_kv(&v, cfg.num_heads / cfg.num_kv_heads)?;
        let k = if rope {
            apply_rope(&k, cfg.rope_theta)?
        } else {
            k
        };
        Ok(ProjectedKv { k, v })
    }

    fn attention_with_projected_kv(
        &self,
        q_input: &Tensor,
        kv: &ProjectedKv,
        prefix: &str,
        layer: usize,
        rope_q: bool,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let (_, q_len, _) = q_input.dims3()?;
        let q = linear(
            q_input,
            &self.layer_tensor(&format!("{prefix}/q_proj/kernel"), layer)?,
        )?
        .reshape((1, q_len, cfg.num_heads, head_dim))?
        .transpose(1, 2)?;
        let q = self.rms_norm(
            &q,
            &self.layer_tensor(&format!("{prefix}/q_norm/scale"), layer)?,
        )?;
        let q = if rope_q {
            apply_rope(&q, cfg.rope_theta)?
        } else {
            q
        };
        let scores = (q.matmul(&kv.k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs
            .matmul(&kv.v)?
            .transpose(1, 2)?
            .reshape((1, q_len, cfg.d_model))?;
        linear(
            &out,
            &self.layer_tensor(&format!("{prefix}/out_proj/kernel"), layer)?,
        )
    }

    fn self_attention_incremental(
        &self,
        q_input: &Tensor,
        cache: &mut LayerSelfAttentionCache,
        prefix: &str,
        layer: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let (_, q_len, _) = q_input.dims3()?;
        let past_len = cache.len();
        let q = linear(
            q_input,
            &self.layer_tensor(&format!("{prefix}/q_proj/kernel"), layer)?,
        )?
        .reshape((1, q_len, cfg.num_heads, head_dim))?
        .transpose(1, 2)?;
        let k = linear(
            q_input,
            &self.layer_tensor(&format!("{prefix}/k_proj/kernel"), layer)?,
        )?
        .reshape((1, q_len, cfg.num_kv_heads, head_dim))?
        .transpose(1, 2)?;
        let v = linear(
            q_input,
            &self.layer_tensor(&format!("{prefix}/v_proj/kernel"), layer)?,
        )?
        .reshape((1, q_len, cfg.num_kv_heads, head_dim))?
        .transpose(1, 2)?;

        let q = self.rms_norm(
            &q,
            &self.layer_tensor(&format!("{prefix}/q_norm/scale"), layer)?,
        )?;
        let k = self.rms_norm(
            &k,
            &self.layer_tensor(&format!("{prefix}/k_norm/scale"), layer)?,
        )?;
        let q = apply_rope_from(&q, cfg.rope_theta, past_len)?;
        let k = apply_rope_from(&k, cfg.rope_theta, past_len)?;
        let k = repeat_kv(&k, cfg.num_heads / cfg.num_kv_heads)?;
        let v = repeat_kv(&v, cfg.num_heads / cfg.num_kv_heads)?;
        cache.push(k, v)?;

        let kv = cache
            .as_projected()
            .context("incremental self-attention cache is empty")?;
        let mut scores = (q.matmul(&kv.k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
        if q_len > 1 {
            scores =
                scores.broadcast_add(&causal_bias_with_past(q_len, past_len, &self.device)?)?;
        }
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs
            .matmul(&kv.v)?
            .transpose(1, 2)?
            .reshape((1, q_len, cfg.d_model))?;
        linear(
            &out,
            &self.layer_tensor(&format!("{prefix}/out_proj/kernel"), layer)?,
        )
    }

    pub fn rms_norm(&self, x: &Tensor, scale: &Tensor) -> Result<Tensor> {
        rms_norm(x, scale)
    }

    fn embedding(&self) -> Result<Tensor> {
        self.tensor("embedding/embedding")
    }

    fn tensor(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .with_context(|| format!("missing tensor {name}"))
    }

    fn layer_tensor(&self, name: &str, layer: usize) -> Result<Tensor> {
        self.tensor(name)?
            .get(layer)
            .with_context(|| format!("slicing {name}[{layer}]"))
    }

    fn layer_scalar(&self, name: &str, layer: usize) -> Result<f32> {
        self.layer_tensor(name, layer)?
            .to_scalar::<f32>()
            .with_context(|| format!("reading scalar {name}[{layer}]"))
    }
}

#[derive(Debug, Clone, Copy)]
struct AttentionWeights {
    prefix: &'static str,
    layer: usize,
    causal: bool,
    rope: bool,
}

#[derive(Debug)]
struct CrossAttentionCache {
    layers: Vec<ProjectedKv>,
}

#[derive(Debug)]
struct DecoderSelfAttentionCache {
    layers: Vec<LayerSelfAttentionCache>,
}

impl DecoderSelfAttentionCache {
    fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerSelfAttentionCache::default())
                .collect(),
        }
    }
}

#[derive(Debug, Default)]
struct LayerSelfAttentionCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl LayerSelfAttentionCache {
    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, k: Tensor, v: Tensor) -> Result<()> {
        let new_len = k.dims()[2];
        self.k = Some(match self.k.take() {
            Some(prev) => Tensor::cat(&[&prev, &k], 2)?,
            None => k,
        });
        self.v = Some(match self.v.take() {
            Some(prev) => Tensor::cat(&[&prev, &v], 2)?,
            None => v,
        });
        self.len += new_len;
        Ok(())
    }

    fn as_projected(&self) -> Option<ProjectedKv> {
        Some(ProjectedKv {
            k: self.k.as_ref()?.clone(),
            v: self.v.as_ref()?.clone(),
        })
    }
}

#[derive(Debug)]
struct ProjectedKv {
    k: Tensor,
    v: Tensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorProbe {
    pub shape: Vec<usize>,
    pub sample: Vec<f32>,
    pub sum: f32,
}

impl TensorProbe {
    fn from_tensor(tensor: &Tensor, sample_len: usize) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let flat = tensor.reshape(tensor.elem_count())?.to_vec1::<f32>()?;
        let sample = flat.iter().copied().take(sample_len).collect::<Vec<_>>();
        let sum = flat.iter().copied().sum::<f32>();
        Ok(Self { shape, sample, sum })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugProbe {
    pub encoder_tokens: usize,
    pub embedding: TensorProbe,
    pub encoder_layer0_attention: TensorProbe,
    pub encoder_final: TensorProbe,
    pub decoder_logits0: TensorProbe,
}

pub fn build_encoder_input(
    tokenizer: &NeedleTokenizer,
    query: &str,
    tools: &str,
    max_enc_len: usize,
) -> Result<Vec<u32>> {
    let mut query_tokens = tokenizer.encode(query)?;
    let mut tool_tokens = tokenizer.encode(tools)?;
    let max_query = max_enc_len.saturating_sub(2);
    if query_tokens.len() > max_query {
        query_tokens.truncate(max_query);
    }
    let remaining = max_enc_len.saturating_sub(query_tokens.len() + 1);
    tool_tokens.truncate(remaining);
    query_tokens.push(TOOLS_ID);
    query_tokens.extend(tool_tokens);
    Ok(query_tokens)
}

pub fn rms_norm(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let denom = ((x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?).sqrt()?;
    let normed = x.broadcast_div(&denom)?;
    let alpha = (scale + 1.0)?;
    Ok(normed.broadcast_mul(&alpha)?)
}

fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() != 3 {
        bail!("linear expects [B,T,D], got {:?}", dims);
    }
    let out_dim = weight.dims()[1];
    Ok(x.reshape((dims[0] * dims[1], dims[2]))?
        .matmul(weight)?
        .reshape((dims[0], dims[1], out_dim))?)
}

pub fn apply_rope(x: &Tensor, theta: f32) -> Result<Tensor> {
    apply_rope_from(x, theta, 0)
}

fn apply_rope_from(x: &Tensor, theta: f32, start_pos: usize) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() != 4 {
        bail!("rope expects [B,H,T,D], got {:?}", dims);
    }
    let seq_len = dims[2];
    let head_dim = dims[3];
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);
    for pos in 0..seq_len {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = (start_pos + pos) as f32 * freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let cos = Tensor::from_vec(cos, (1, 1, seq_len, half), x.device())?;
    let sin = Tensor::from_vec(sin, (1, 1, seq_len, half), x.device())?;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let left = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let right = x2
        .broadcast_mul(&cos)?
        .broadcast_add(&x1.broadcast_mul(&sin)?)?;
    Ok(Tensor::cat(&[left, right], D::Minus1)?)
}

fn repeat_kv(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let heads = x.dims()[1];
    let mut pieces = Vec::with_capacity(heads * repeats);
    for head in 0..heads {
        let h = x.narrow(1, head, 1)?;
        for _ in 0..repeats {
            pieces.push(h.clone());
        }
    }
    let refs = pieces.iter().collect::<Vec<_>>();
    Ok(Tensor::cat(&refs, 1)?)
}

fn causal_bias(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut data: Vec<f32> = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            data.push(if j <= i { 0.0f32 } else { -1.0e30f32 });
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)?)
}

fn causal_bias_with_past(q_len: usize, past_len: usize, device: &Device) -> Result<Tensor> {
    let kv_len = past_len + q_len;
    let mut data: Vec<f32> = Vec::with_capacity(q_len * kv_len);
    for i in 0..q_len {
        let max_key = past_len + i;
        for j in 0..kv_len {
            data.push(if j <= max_key { 0.0f32 } else { -1.0e30f32 });
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, q_len, kv_len), device)?)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (idx, value) in xs.iter().copied().enumerate() {
        if value > best {
            best = value;
            best_idx = idx;
        }
    }
    best_idx
}

pub fn inspect_assets(assets: &Assets) -> Result<String> {
    let model = NeedleModel::load(assets)?;
    let tokenizer = NeedleTokenizer::load(&assets.tokenizer)?;
    Ok(format!(
        "assets: {}\nweights: {}\nconfig: d_model={} enc_layers={} dec_layers={} heads={}/{} no_feedforward={}\ntensors: {} params: {}\ntokenizer_vocab: {} pad_id={} eos_id={}",
        assets.dir.display(),
        assets.weights.display(),
        model.config.d_model,
        model.config.num_encoder_layers,
        model.config.num_decoder_layers,
        model.config.num_heads,
        model.config.num_kv_heads,
        model.config.no_feedforward,
        model.tensor_count(),
        model.parameter_count(),
        tokenizer.vocab_size(),
        PAD_ID,
        EOS_ID
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_matches_zero_center_formula() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0], [2.0, 0.0, 0.0]], &dev)?;
        let scale = Tensor::new(&[0.0f32, 1.0, -0.5], &dev)?;
        let y = rms_norm(&x, &scale)?.to_vec2::<f32>()?;
        let denom0 = ((1.0f32 + 4.0 + 9.0) / 3.0 + 1e-6).sqrt();
        assert!((y[0][0] - 1.0 / denom0).abs() < 1e-5);
        assert!((y[0][1] - 4.0 / denom0).abs() < 1e-5);
        assert!((y[0][2] - 1.5 / denom0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn rope_preserves_position_zero_and_rotates_position_one() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            (1, 1, 2, 4),
            &dev,
        )?;
        let y = apply_rope(&x, 10000.0)?
            .squeeze(0)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        assert!((y[0][0] - 1.0).abs() < 1e-6);
        assert!((y[0][1] - 2.0).abs() < 1e-6);
        assert!((y[0][2] - 3.0).abs() < 1e-6);
        assert!((y[0][3] - 4.0).abs() < 1e-6);
        let c = 1.0f32.cos();
        let s = 1.0f32.sin();
        assert!((y[1][0] - (1.0 * c - 3.0 * s)).abs() < 1e-6);
        assert!((y[1][2] - (3.0 * c + 1.0 * s)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn encoder_input_places_tools_separator() -> Result<()> {
        let path = "needle-weights/tokenizer/needle.model";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping encoder input tokenizer test; local tokenizer is missing");
            return Ok(());
        }
        let tok = NeedleTokenizer::load(path)?;
        let enc = build_encoder_input(&tok, "weather", "[]", 32)?;
        assert!(enc.contains(&TOOLS_ID));
        Ok(())
    }
}
