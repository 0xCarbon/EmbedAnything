use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, rms_norm, Activation, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

/// Granite Vision Configuration - based on SigLIP architecture
#[derive(Debug, Clone, Deserialize)]
pub struct GraniteVisionConfig {
    pub hidden_act: String,
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl Default for GraniteVisionConfig {
    fn default() -> Self {
        Self {
            hidden_act: "gelu_pytorch_tanh".to_string(),
            hidden_size: 1152,
            image_size: 384,
            intermediate_size: 4304,
            layer_norm_eps: 1e-6,
            model_type: "siglip_vision_model".to_string(),
            num_attention_heads: 16,
            num_hidden_layers: 27,
            patch_size: 14,
            torch_dtype: Some("bfloat16".to_string()),
        }
    }
}

/// Granite Text Configuration - based on Granite language model
#[derive(Debug, Clone, Deserialize)]
pub struct GraniteTextConfig {
    pub architectures: Vec<String>,
    pub attention_dropout: f64,
    pub attention_multiplier: f64,
    pub bos_token_id: u32,
    pub embedding_multiplier: f64,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub logits_scaling: f64,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: usize,
    pub pad_token_id: u32,
    pub residual_multiplier: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

impl Default for GraniteTextConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["GraniteForCausalLM".to_string()],
            attention_dropout: 0.1,
            attention_multiplier: 0.015625,
            bos_token_id: 0,
            embedding_multiplier: 12.0,
            eos_token_id: 0,
            hidden_size: 2048,
            intermediate_size: 8192,
            logits_scaling: 8.0,
            max_position_embeddings: 131072,
            model_type: "granite".to_string(),
            num_hidden_layers: 40,
            num_attention_heads: Some(32),
            num_key_value_heads: 8,
            pad_token_id: 0,
            residual_multiplier: 0.22,
            rms_norm_eps: 1e-5,
            rope_theta: 300000.0,
            tie_word_embeddings: true,
            vocab_size: 49156,
            torch_dtype: Some("bfloat16".to_string()),
        }
    }
}

/// Granite Processor Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct GraniteProcessorConfig {
    pub crop_size: ImageSize,
    pub default_to_square: bool,
    pub do_center_crop: bool,
    pub do_convert_rgb: Option<bool>,
    pub do_normalize: bool,
    pub do_pad: bool,
    pub do_rescale: bool,
    pub do_resize: bool,
    pub image_grid_pinpoints: Vec<[usize; 2]>,
    pub image_mean: Vec<f32>,
    pub image_processor_type: String,
    pub image_std: Vec<f32>,
    pub processor_class: String,
    pub resample: u32,
    pub rescale_factor: f64,
    pub size: ImageSize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageSize {
    pub height: usize,
    pub width: usize,
}

impl Default for GraniteProcessorConfig {
    fn default() -> Self {
        Self {
            crop_size: ImageSize { height: 384, width: 384 },
            default_to_square: false,
            do_center_crop: true,
            do_convert_rgb: None,
            do_normalize: true,
            do_pad: true,
            do_rescale: true,
            do_resize: true,
            image_grid_pinpoints: vec![
                [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304],
                [384, 2688], [384, 3072], [384, 3456], [384, 3840],
                [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920],
                [1152, 384], [1152, 768], [1152, 1152],
                [1536, 384], [1536, 768],
                [1920, 384], [1920, 768],
                [2304, 384], [2688, 384], [3072, 384], [3456, 384], [3840, 384]
            ],
            image_mean: vec![0.5, 0.5, 0.5],
            image_processor_type: "LlavaNextImageProcessor".to_string(),
            image_std: vec![0.5, 0.5, 0.5],
            processor_class: "GraniteVisionEmbProcessor".to_string(),
            resample: 3,
            rescale_factor: 0.00392156862745098,
            size: ImageSize { height: 384, width: 384 },
        }
    }
}

/// Overall Granite Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct GraniteConfig {
    pub model_type: String,
    pub vision_config: GraniteVisionConfig,
    pub text_config: GraniteTextConfig,
    pub emb_dim_doc: usize,
    pub emb_dim_query: usize,
    pub image_grid_pinpoints: Vec<[usize; 2]>,
    pub image_seq_length: usize,
    pub image_token_index: u32,
    pub multimodal_projector_bias: bool,
    pub projector_hidden_act: String,
    pub tie_word_embeddings: bool,
    pub use_image_newline_parameter: bool,
    pub vision_feature_layer: Vec<i32>,
    pub vision_feature_select_strategy: String,
    pub base_image_feature_location: String,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

impl Default for GraniteConfig {
    fn default() -> Self {
        Self {
            model_type: "granitevisionemb".to_string(),
            vision_config: GraniteVisionConfig::default(),
            text_config: GraniteTextConfig::default(),
            emb_dim_doc: 128,
            emb_dim_query: 128,
            image_grid_pinpoints: vec![
                [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304],
                [384, 2688], [384, 3072], [384, 3456], [384, 3840],
                [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920],
                [1152, 384], [1152, 768], [1152, 1152],
                [1536, 384], [1536, 768],
                [1920, 384], [1920, 768],
                [2304, 384], [2688, 384], [3072, 384], [3456, 384], [3840, 384]
            ],
            image_seq_length: 576,
            image_token_index: 49155,
            multimodal_projector_bias: true,
            projector_hidden_act: "gelu".to_string(),
            tie_word_embeddings: true,
            use_image_newline_parameter: true,
            vision_feature_layer: vec![-24, -20, -12, -1],
            vision_feature_select_strategy: "full".to_string(),
            base_image_feature_location: "last".to_string(),
            torch_dtype: Some("float32".to_string()),
            transformers_version: Some("4.49.0".to_string()),
        }
    }
}

// Vision Components

/// Granite Vision Patch Embedding - SigLIP style
#[derive(Debug)]
pub struct GraniteVisionPatchEmbed {
    weight: Tensor,
    bias: Option<Tensor>,
    patch_size: usize,
    hidden_size: usize,
    in_channels: usize,
}

impl GraniteVisionPatchEmbed {
    pub fn new(cfg: &GraniteVisionConfig, vb: VarBuilder) -> Result<Self> {
        let in_channels = 3; // RGB
        
        // Load Conv2d weights: [out_channels, in_channels, height, width]
        let weight = vb.get(
            (cfg.hidden_size, in_channels, cfg.patch_size, cfg.patch_size),
            "weight",
        )?;
        
        // Try to load bias (might not exist)
        let bias = vb.get(cfg.hidden_size, "bias").ok();
        
        Ok(Self {
            weight,
            bias,
            patch_size: cfg.patch_size,
            hidden_size: cfg.hidden_size,
            in_channels,
        })
    }
}

impl Module for GraniteVisionPatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Input: (B, C, H, W)
        let (batch_size, _channels, height, width) = xs.dims4()?;
        
        // Extract patches manually using conv2d-like operation
        let height_patches = height / self.patch_size;
        let width_patches = width / self.patch_size;
        
        if height_patches == 0 || width_patches == 0 {
            return Err(candle_core::Error::Msg(format!(
                "Invalid patch dimensions: height_patches={}, width_patches={}",
                height_patches, width_patches
            )));
        }
        
        let mut all_patches = Vec::new();
        
        for b in 0..batch_size {
            let batch_input = xs.get(b)?; // (C, H, W)
            
            for h in 0..height_patches {
                for w in 0..width_patches {
                    // Extract one patch: (C, patch_size, patch_size)
                    let patch = batch_input
                        .narrow(1, h * self.patch_size, self.patch_size)?
                        .narrow(2, w * self.patch_size, self.patch_size)?;
                    
                    // Flatten patch for matrix multiplication
                    let patch_flat = patch.flatten_all()?; // (C * patch_size * patch_size,)
                    
                    // Reshape weights for matrix multiplication
                    let weight_2d = self.weight.reshape((
                        self.hidden_size,
                        self.in_channels * self.patch_size * self.patch_size,
                    ))?;
                    
                    // Apply convolution as matrix multiplication
                    let output_patch = weight_2d.matmul(&patch_flat.unsqueeze(1)?)?.squeeze(1)?; // (hidden_size,)
                    
                    all_patches.push(output_patch);
                }
            }
        }
        
        if all_patches.is_empty() {
            return Err(candle_core::Error::Msg("No patches were created".to_string()));
        }
        
        // Stack all patches and reshape
        let output = Tensor::stack(&all_patches, 0)?; // (B * height_patches * width_patches, hidden_size)
        let output = output.reshape((batch_size, height_patches * width_patches, self.hidden_size))?;
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            let bias_expanded = bias.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(output.shape())?;
            Ok(output.add(&bias_expanded)?)
        } else {
            Ok(output)
        }
    }
}

/// Granite Vision Attention - SigLIP style with separate q/k/v projections
#[derive(Debug)]
pub struct GraniteVisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl GraniteVisionAttention {
    pub fn new(cfg: &GraniteVisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let scale = (head_dim as f64).sqrt().recip();
        
        let attn_vb = vb.pp("self_attn");
        let q_proj = linear(cfg.hidden_size, cfg.hidden_size, attn_vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, cfg.hidden_size, attn_vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, cfg.hidden_size, attn_vb.pp("v_proj"))?;
        let out_proj = linear(cfg.hidden_size, cfg.hidden_size, attn_vb.pp("out_proj"))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale,
        })
    }
}

impl Module for GraniteVisionAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        
        let q = q.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?; // (B, num_heads, N, head_dim)
        let k = k.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?; // (B, num_heads, N, head_dim)
        let v = v.reshape((b, n, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?; // (B, num_heads, N, head_dim)
        
        // Scaled dot-product attention
        let attn = q.contiguous()?.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
        let attn = (attn * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        
        let out = attn.contiguous()?.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.contiguous()?; // (B, N, num_heads, head_dim)
        let out = out.reshape((b, n, self.num_heads * self.head_dim))?;
        
        self.out_proj.forward(&out)
    }
}

/// Granite Vision MLP - SigLIP style with GELU activation
#[derive(Debug)]
pub struct GraniteVisionMLP {
    fc1: Linear,
    fc2: Linear,
    act_fn: Activation,
}

impl GraniteVisionMLP {
    pub fn new(cfg: &GraniteVisionConfig, vb: VarBuilder) -> Result<Self> {
        let mlp_vb = vb.pp("mlp");
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, mlp_vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, mlp_vb.pp("fc2"))?;
        
        // SigLIP uses gelu_pytorch_tanh activation
        let act_fn = match cfg.hidden_act.as_str() {
            "gelu_pytorch_tanh" => Activation::GeluPytorchTanh,
            "gelu" => Activation::Gelu,
            _ => Activation::Gelu, // Default fallback
        };
        
        Ok(Self {
            fc1,
            fc2,
            act_fn,
        })
    }
}

impl Module for GraniteVisionMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.act_fn.forward(&xs)?;
        self.fc2.forward(&xs)
    }
}

/// Granite Vision Encoder Layer - SigLIP style
#[derive(Debug)]
pub struct GraniteVisionEncoderLayer {
    layer_norm1: candle_nn::LayerNorm,
    attn: GraniteVisionAttention,
    layer_norm2: candle_nn::LayerNorm,
    mlp: GraniteVisionMLP,
}

impl GraniteVisionEncoderLayer {
    pub fn new(cfg: &GraniteVisionConfig, vb: VarBuilder) -> Result<Self> {
        let layer_norm1 = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?;
        let attn = GraniteVisionAttention::new(cfg, vb.clone())?;
        let layer_norm2 = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?;
        let mlp = GraniteVisionMLP::new(cfg, vb.clone())?;
        
        Ok(Self {
            layer_norm1,
            attn,
            layer_norm2,
            mlp,
        })
    }
}

impl Module for GraniteVisionEncoderLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.attn.forward(&xs)?;
        let xs = (xs + residual)?;
        
        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

/// Granite Vision Transformer - SigLIP based
#[derive(Debug)]
pub struct GraniteVisionTransformer {
    patch_embed: GraniteVisionPatchEmbed,
    layers: Vec<GraniteVisionEncoderLayer>,
    layer_norm: candle_nn::LayerNorm,
    dtype: DType,
}

impl GraniteVisionTransformer {
    pub fn new(cfg: &GraniteVisionConfig, vb: VarBuilder, dtype: DType) -> Result<Self> {
        let patch_embed = GraniteVisionPatchEmbed::new(cfg, vb.pp("embeddings").pp("patch_embedding"))?;
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(GraniteVisionEncoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        
        // SigLIP doesn't have a final layer norm, but if it exists, use it
        let layer_norm = if let Ok(ln) = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm")) {
            ln
        } else {
            // Create a dummy layer norm if it doesn't exist
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm")).unwrap_or_else(|_| {
                // Create identity layer norm
                candle_nn::LayerNorm::new(
                    Tensor::ones(cfg.hidden_size, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                    Tensor::zeros(cfg.hidden_size, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap(),
                    cfg.layer_norm_eps
                )
            })
        };
        
        Ok(Self {
            patch_embed,
            layers,
            layer_norm,
            dtype,
        })
    }
}

impl Module for GraniteVisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Ensure input is in correct dtype
        let xs = xs.to_dtype(self.dtype)?;
        let xs = self.patch_embed.forward(&xs)?;
        
        // Extract features from multiple layers as specified in config
        // vision_feature_layer: [-24, -20, -12, -1] with 27 total layers
        let target_layers = [3, 7, 15, 26]; // Convert negative indices to positive
        let mut layer_outputs = Vec::new();
        let mut xs = xs;
        
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            // Store outputs from target layers
            if target_layers.contains(&i) {
                layer_outputs.push(xs.clone());
            }
        }
        
        // Apply layer norm to the last layer output
        if let Some(last_output) = layer_outputs.last_mut() {
            *last_output = self.layer_norm.forward(last_output)?;
        }
        
        // Concatenate features from all target layers
        // Each layer output: [batch_size, seq_len, 1152]
        // Result: [batch_size, seq_len, 4608] (4 * 1152)
        Tensor::cat(&layer_outputs, D::Minus1)
    }
}

// Language Model Components

/// RoPE (Rotary Position Embedding) for Granite
#[derive(Debug)]
pub struct GraniteRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl GraniteRotaryEmbedding {
    pub fn new(cfg: &GraniteTextConfig, device: &Device, dtype: DType) -> Result<Self> {
        // Use correct head dimension based on tensor analysis
        let head_dim = 64; // Correct head dimension: 2048/32=64 for q, 512/8=64 for k,v
        
        let max_seq_len = cfg.max_position_embeddings;
        let base = cfg.rope_theta;
        
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
        
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let inv_freq = inv_freq.reshape((1, head_dim / 2))?;
        let freqs = t.broadcast_matmul(&inv_freq)?;
        
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        
        Ok(Self { sin, cos })
    }
    
    pub fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, _position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(D::Minus2)?;
        let cos = self.cos.narrow(0, 0, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, 0, seq_len)?.to_dtype(q.dtype())?;
        
        let q_embed = Self::rotate_half(q, &cos, &sin)?;
        let k_embed = Self::rotate_half(k, &cos, &sin)?;
        
        Ok((q_embed, k_embed))
    }
    
    fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let last_dim = x.dim(D::Minus1)?;
        
        // Split the input into two halves for rotation
        let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
        let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
        
        // Ensure cos and sin have the right shape for broadcasting
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        // Apply RoPE to each half
        let x1_cos = x1.broadcast_mul(&cos)?;
        let x1_sin = x1.broadcast_mul(&sin)?;
        let x2_cos = x2.broadcast_mul(&cos)?;
        let x2_sin = x2.broadcast_mul(&sin)?;
        
        // Rotate: [x1*cos - x2*sin, x2*cos + x1*sin]
        let rotated_1 = (x1_cos - x2_sin)?;
        let rotated_2 = (x2_cos + x1_sin)?;
        
        // Concatenate the rotated halves
        Tensor::cat(&[&rotated_1, &rotated_2], D::Minus1)
    }
}

/// Granite Multi-Head Attention with Grouped Query Attention
#[derive(Debug)]
pub struct GraniteAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: GraniteRotaryEmbedding,
    attention_multiplier: f64,
}

impl GraniteAttention {
    pub fn new(cfg: &GraniteTextConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        // Calculate dimensions from actual tensor shapes
        // q_proj: [2048, 2048] → 32 heads × 64 head_dim  
        // k_proj: [512, 2048] → 8 heads × 64 head_dim
        // v_proj: [512, 2048] → 8 heads × 64 head_dim
        let num_heads = 32;
        let head_dim = 64; // Fixed: must be 64 for all heads
        let kv_head_dim = 64; // Same as head_dim
        let num_key_value_heads = cfg.num_key_value_heads;
        
        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        // For GQA, k and v projections use num_key_value_heads
        let k_proj = linear_no_bias(hidden_size, num_key_value_heads * kv_head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_key_value_heads * kv_head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        
        let rotary_emb = GraniteRotaryEmbedding::new(cfg, device, dtype)?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_key_value_heads,
            head_dim, // Use consistent head_dim for all
            rotary_emb,
            attention_multiplier: cfg.attention_multiplier,
        })
    }
}

impl Module for GraniteAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        
        // Reshape with consistent head dimensions
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?; // Use same head_dim
        let v = v.reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?; // Use same head_dim
        
        let q = q.transpose(1, 2)?; // (B, num_heads, seq_len, head_dim)
        let k = k.transpose(1, 2)?; // (B, num_key_value_heads, seq_len, head_dim)
        let v = v.transpose(1, 2)?; // (B, num_key_value_heads, seq_len, head_dim)
        
        // Create position IDs (simple sequential for now)
        let position_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?
            .to_dtype(xs.dtype())?
            .reshape((1, seq_len))?
            .broadcast_as((b, seq_len))?;
        
        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;
        
        // Expand key and value for grouped query attention
        let num_key_value_groups = self.num_heads / self.num_key_value_heads;
        let k = Self::repeat_kv(&k, num_key_value_groups)?;
        let v = Self::repeat_kv(&v, num_key_value_groups)?;
        
        // Scaled dot-product attention with Granite's attention multiplier
        let scale = (self.head_dim as f64).sqrt().recip() * self.attention_multiplier;
        let attn_weights = q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
        let attn_weights = attn_weights.affine(scale, 0.0)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?; // (B, seq_len, num_heads, head_dim)
        let attn_output = attn_output.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        
        self.o_proj.forward(&attn_output)
    }
}

impl GraniteAttention {
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.contiguous()?);
        }
        
        let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x.unsqueeze(2)?; // (B, n_kv_heads, 1, seq_len, head_dim)
        let x = x.expand((b, n_kv_heads, n_rep, seq_len, head_dim))?;
        x.reshape((b, n_kv_heads * n_rep, seq_len, head_dim))?.contiguous()
    }
}

/// Granite MLP with SwiGLU activation
#[derive(Debug)]
pub struct GraniteMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GraniteMLP {
    pub fn new(cfg: &GraniteTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        let act_fn = Activation::Silu; // SwiGLU uses SiLU activation
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for GraniteMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        let intermediate = gate.mul(&up)?;
        self.down_proj.forward(&intermediate)
    }
}

/// Granite Decoder Layer
#[derive(Debug)]
pub struct GraniteDecoderLayer {
    self_attn: GraniteAttention,
    mlp: GraniteMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_multiplier: f64,
}

impl GraniteDecoderLayer {
    pub fn new(cfg: &GraniteTextConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let self_attn = GraniteAttention::new(cfg, vb.pp("self_attn"), device, dtype)?;
        let mlp = GraniteMLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: cfg.residual_multiplier,
        })
    }
}

impl Module for GraniteDecoderLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs.affine(self.residual_multiplier, 0.0)? + residual)?;
        
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs.affine(self.residual_multiplier, 0.0)? + residual
    }
}

/// Granite Language Model
#[derive(Debug)]
pub struct GraniteLanguageModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<GraniteDecoderLayer>,
    norm: RmsNorm,
    embedding_multiplier: f64,
}

impl GraniteLanguageModel {
    pub fn new(cfg: &GraniteTextConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(GraniteDecoderLayer::new(cfg, vb_layers.pp(i), device, dtype)?);
        }
        
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            embedding_multiplier: cfg.embedding_multiplier,
        })
    }
}

impl Module for GraniteLanguageModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = xs.affine(self.embedding_multiplier, 0.0)?; // Apply embedding multiplier
        
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        
        self.norm.forward(&xs)
    }
}

impl GraniteLanguageModel {
    /// Forward with vision embeddings injected
    pub fn forward_with_vision(&self, input_ids: &Tensor, vision_embeds: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = xs.affine(self.embedding_multiplier, 0.0)?; // Apply embedding multiplier
        
        // If vision embeddings are provided, inject them at appropriate positions
        if let Some(vision_embeds) = vision_embeds {
            // For simplicity, prepend vision embeddings to text embeddings
            // In a full implementation, you'd need to handle vision token positions properly
            xs = Tensor::cat(&[vision_embeds, &xs], 1)?;
        }
        
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        
        self.norm.forward(&xs)
    }
}

/// Main Granite Model combining vision and language components
#[derive(Debug)]
pub struct GraniteModel {
    vision_tower: GraniteVisionTransformer,
    language_model: GraniteLanguageModel,
    multi_modal_projector_1: Linear,
    multi_modal_projector_2: Linear,
    custom_text_proj: Linear,
    #[allow(dead_code)]
    image_newline: Option<Tensor>,
    config: GraniteConfig,
    #[allow(dead_code)]
    dtype: DType,
}

impl GraniteModel {
    pub fn new(config: &GraniteConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let vision_tower = GraniteVisionTransformer::new(
            &config.vision_config,
            vb.pp("model").pp("vision_tower").pp("vision_model"),
            dtype,
        )?;
        
        let language_model = GraniteLanguageModel::new(
            &config.text_config,
            vb.pp("model").pp("language_model").pp("model"),
            device,
            dtype,
        )?;
        
        // Multi-modal projector - 2-layer MLP
        // Input dimension is 4 * vision_hidden_size due to multi-layer feature concatenation
        let multi_modal_projector_1 = linear(
            config.vision_config.hidden_size * 4, // Input: 4608 (4 layers × 1152)
            config.text_config.hidden_size, // Output: 2048
            vb.pp("model").pp("multi_modal_projector").pp("linear_1"),
        )?;
        
        let multi_modal_projector_2 = linear(
            config.text_config.hidden_size, // Input: 2048
            config.text_config.hidden_size, // Output: 2048 (to match text embeddings)
            vb.pp("model").pp("multi_modal_projector").pp("linear_2"),
        )?;
        
        // Custom text projection layer for final embeddings (128-dimensional)
        let custom_text_proj = linear(
            config.text_config.hidden_size,
            config.emb_dim_doc, // 128 dimensions
            vb.pp("custom_text_proj"),
        )?;
        
        // Image newline parameter (optional)
        let image_newline = if config.use_image_newline_parameter {
            Some(vb.get(config.text_config.hidden_size, "model.image_newline")?)
        } else {
            None
        };
        
        Ok(Self {
            vision_tower,
            language_model,
            multi_modal_projector_1,
            multi_modal_projector_2,
            custom_text_proj,
            image_newline,
            config: config.clone(),
            dtype,
        })
    }
    
    /// Forward with text only
    pub fn forward_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.language_model.forward(input_ids)
    }
    
    /// Forward with vision only
    pub fn forward_vision(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_outputs = self.vision_tower.forward(pixel_values)?;
        let projected_1 = self.multi_modal_projector_1.forward(&vision_outputs)?;
        self.multi_modal_projector_2.forward(&projected_1)
    }
    
    /// Forward with both vision and text
    pub fn forward(&self, input_ids: &Tensor, pixel_values: Option<&Tensor>) -> Result<Tensor> {
        let vision_embeds = if let Some(pixel_values) = pixel_values {
            let vision_outputs = self.vision_tower.forward(pixel_values)?;
            let projected_1 = self.multi_modal_projector_1.forward(&vision_outputs)?;
            let projected_2 = self.multi_modal_projector_2.forward(&projected_1)?;
            Some(projected_2)
        } else {
            None
        };
        
        self.language_model.forward_with_vision(input_ids, vision_embeds.as_ref())
    }
    
    /// Get embeddings for retrieval
    pub fn get_embeddings(&self, input_ids: &Tensor, pixel_values: Option<&Tensor>) -> Result<Tensor> {
        let last_hidden_states = self.forward(input_ids, pixel_values)?;
        
        // Apply custom text projection to reduce to embedding dimension (128D)
        let projected = self.custom_text_proj.forward(&last_hidden_states)?;
        
        // For causal models like Granite, use the last token embedding
        // projected shape: [batch_size, seq_len, emb_dim]
        let seq_len = projected.dim(1)?;
        let last_token_embeddings = projected.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        
        // L2 normalization
        let norm = last_token_embeddings.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        last_token_embeddings.broadcast_div(&norm)
    }
    
    pub fn config(&self) -> &GraniteConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    
    #[test]
    fn test_config_deserialization() {
        let config = GraniteConfig::default();
        assert_eq!(config.model_type, "granitevisionemb");
        assert_eq!(config.emb_dim_doc, 128);
        assert_eq!(config.emb_dim_query, 128);
    }
    
    #[test]
    fn test_vision_config() {
        let vision_config = GraniteVisionConfig::default();
        assert_eq!(vision_config.hidden_size, 1152);
        assert_eq!(vision_config.num_attention_heads, 16);
        assert_eq!(vision_config.patch_size, 14);
    }
    
    #[test]
    fn test_text_config() {
        let text_config = GraniteTextConfig::default();
        assert_eq!(text_config.hidden_size, 2048);
        assert_eq!(text_config.num_hidden_layers, 40);
        assert_eq!(text_config.vocab_size, 49156);
    }
}