use candle_core::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, rms_norm, Activation, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;

/// Qwen2.5-VL Vision Configuration - matches Python implementation
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub fullatt_block_indexes: Vec<usize>,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub tokens_per_second: usize,
    pub window_size: usize,
    pub out_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub initializer_range: f64,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
    #[serde(default)]
    pub model_type: Option<String>,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            depth: 32,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            hidden_act: "silu".to_string(),
            hidden_size: 1280,
            in_channels: 3,
            patch_size: 14,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            tokens_per_second: 2,
            window_size: 112,
            out_hidden_size: 2048,
            intermediate_size: 3420,
            num_heads: 16,
            initializer_range: 0.02,
            torch_dtype: Some("float32".to_string()),
            transformers_version: Some("4.53.3".to_string()),
            model_type: Some("qwen2_5_vl".to_string()),
        }
    }
}

/// Qwen2.5-VL Text Configuration - matches Python implementation
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub architectures: Vec<String>,
    pub attention_dropout: f64,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    // Vision-related token IDs - these can be null in the config
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision_token_id: u32,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

fn default_pad_token_id() -> u32 {
    151643
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["Qwen2_5_VLForConditionalGeneration".to_string()],
            attention_dropout: 0.0,
            bos_token_id: 151643,
            eos_token_id: 151645,
            hidden_act: "silu".to_string(),
            hidden_size: 2048,
            initializer_range: 0.02,
            intermediate_size: 11008,
            layer_types: vec!["full_attention".to_string(); 36],
            max_position_embeddings: 128000,
            max_window_layers: 70,
            model_type: "qwen2_5_vl_text".to_string(),
            num_attention_heads: 16,
            num_hidden_layers: 36,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            sliding_window: None,
            tie_word_embeddings: true,
            use_cache: true,
            use_sliding_window: false,
            vocab_size: 151936,
            pad_token_id: 151643,
            // Vision token IDs - matching actual config where some are null
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            vision_token_id: 151654,
            torch_dtype: Some("float32".to_string()),
            transformers_version: Some("4.53.3".to_string()),
            rope_scaling: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    pub do_convert_rgb: bool,
    pub do_normalize: bool,
    pub do_rescale: bool,
    pub do_resize: bool,
    pub image_mean: Vec<f32>,
    pub image_processor_type: String,
    pub image_std: Vec<f32>,
    pub max_pixels: usize,
    pub merge_size: u32,
    pub min_pixels: usize,
    pub patch_size: u32,
    pub processor_class: String,
    pub resample: u32,
    pub rescale_factor: f64,
    pub size: SizeConfig,
    pub temporal_patch_size: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SizeConfig {
    pub longest_edge: usize,
    pub shortest_edge: usize,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            do_convert_rgb: true,
            do_normalize: true,
            do_rescale: true,
            do_resize: true,
            image_mean: vec![0.48145466, 0.4578275, 0.40821073],
            image_processor_type: "Qwen2VLImageProcessor".to_string(),
            image_std: vec![0.26862954, 0.26130258, 0.27577711],
            max_pixels: 12845056,
            merge_size: 2,
            min_pixels: 3136,
            patch_size: 14,
            processor_class: "ColQwen2_5_Processor".to_string(),
            resample: 3,
            rescale_factor: 0.00392156862745098,
            size: SizeConfig {
                longest_edge: 12845056,
                shortest_edge: 3136,
            },
            temporal_patch_size: 2,
        }
    }
}

/// Overall Qwen2.5-VL Configuration - matches Python implementation
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    // Additional fields from the main config - inherit from text_config
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

impl Config {
    // Helper methods to get token IDs from text_config
    pub fn image_token_id(&self) -> u32 {
        self.text_config.image_token_id.unwrap_or(151654)
    }
    
    pub fn video_token_id(&self) -> u32 {
        self.text_config.video_token_id.unwrap_or(151655)
    }
    
    pub fn vision_start_token_id(&self) -> u32 {
        self.text_config.vision_start_token_id
    }
    
    pub fn vision_end_token_id(&self) -> u32 {
        self.text_config.vision_end_token_id
    }
    
    pub fn vision_token_id(&self) -> u32 {
        self.text_config.vision_token_id
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: "qwen2_5_vl".to_string(),
            vision_config: VisionConfig::default(),
            text_config: TextConfig::default(),
            torch_dtype: Some("float32".to_string()),
            transformers_version: Some("4.53.3".to_string()),
        }
    }
}

// Vision Components

/// Qwen2.5-VL Vision Patch Embedding
/// Simplified approach: Load Conv3d weights and reshape input to work with them
#[derive(Debug)]
pub struct VisionPatchEmbed {
    weight: Tensor,
    bias: Option<Tensor>,
    temporal_patch_size: usize,
    patch_size: usize,
    hidden_size: usize,
    in_channels: usize,
}

impl VisionPatchEmbed {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        // Load the Conv3d weights: [out_channels, in_channels, temporal, height, width]
        let weight = vb.get(
            (
                cfg.hidden_size,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "proj.weight",
        )?;
        
        // Try to load bias (might not exist)
        let bias = vb.get(cfg.hidden_size, "proj.bias").ok();
        
        Ok(Self {
            weight,
            bias,
            temporal_patch_size: cfg.temporal_patch_size,
            patch_size: cfg.patch_size,
            hidden_size: cfg.hidden_size,
            in_channels: cfg.in_channels,
        })
    }
}

impl Module for VisionPatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Input: (B, C, T, H, W)
        // We need to simulate Conv3d operation
        let (batch_size, _channels, temporal, height, width) = xs.dims5()?;
        
        // Extract patches manually
        // For single images (temporal=1), use temporal_patch_size=1 to avoid division issues
        let effective_temporal_patch_size = if temporal == 1 { 1 } else { self.temporal_patch_size };
        let temporal_patches = temporal / effective_temporal_patch_size;
        let height_patches = height / self.patch_size;
        let width_patches = width / self.patch_size;
        
        if temporal_patches == 0 || height_patches == 0 || width_patches == 0 {
            return Err(candle_core::Error::Msg(format!(
                "Invalid patch dimensions: temporal_patches={}, height_patches={}, width_patches={}",
                temporal_patches, height_patches, width_patches
            )));
        }
        
        let mut all_patches = Vec::new();
        
        for b in 0..batch_size {
            let batch_input = xs.get(b)?; // (C, T, H, W)
            
            for t in 0..temporal_patches {
                for h in 0..height_patches {
                    for w in 0..width_patches {
                        // Extract one patch: (C, effective_temporal_patch_size, patch_size, patch_size)
                        let patch = batch_input
                            .narrow(1, t * effective_temporal_patch_size, effective_temporal_patch_size)?
                            .narrow(2, h * self.patch_size, self.patch_size)?
                            .narrow(3, w * self.patch_size, self.patch_size)?;
                        
                        // Flatten patch for matrix multiplication
                        let patch_flat = patch.flatten_all()?; // (C * effective_temporal_patch_size * patch_size * patch_size,)
                        
                        // Reshape weights for matrix multiplication based on effective temporal patch size
                        let expected_input_size = self.in_channels * effective_temporal_patch_size * self.patch_size * self.patch_size;
                        let weight_2d = if effective_temporal_patch_size == 1 && self.temporal_patch_size == 2 {
                            // For single temporal frames, we need to adapt the weight matrix
                            // Take only the first half of the temporal dimension in weights
                            let full_weight_2d = self.weight.reshape((
                                self.hidden_size,
                                self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size,
                            ))?;
                            // Extract only the portion corresponding to the first temporal slice
                            let single_temporal_size = self.in_channels * self.patch_size * self.patch_size;
                            full_weight_2d.narrow(1, 0, single_temporal_size)?.contiguous()? // Fix non-contiguous tensor
                        } else {
                            self.weight.reshape((
                                self.hidden_size,
                                expected_input_size,
                            ))?.contiguous()? // Ensure contiguous for matmul
                        };
                        
                        // Apply convolution as matrix multiplication
                        let output_patch = weight_2d.matmul(&patch_flat.contiguous()?.unsqueeze(1)?)?.squeeze(1)?; // (hidden_size,)
                        
                        all_patches.push(output_patch);
                    }
                }
            }
        }
        
        if all_patches.is_empty() {
            return Err(candle_core::Error::Msg("No patches were created".to_string()));
        }
        
        // Stack all patches and reshape
        let output = Tensor::stack(&all_patches, 0)?; // (B * temporal_patches * height_patches * width_patches, hidden_size)
        let output = output.reshape((batch_size, temporal_patches * height_patches * width_patches, self.hidden_size))?;
        
        // Add bias if present
        if let Some(bias) = &self.bias {
            let bias_expanded = bias.unsqueeze(0)?.unsqueeze(0)?.broadcast_as(output.shape())?;
            Ok(output.add(&bias_expanded)?)
        } else {
            Ok(output)
        }
    }
}

/// Qwen2.5-VL Vision Attention
#[derive(Debug)]
pub struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        let qkv = linear(cfg.hidden_size, cfg.hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("proj"))?;
        
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }
}

impl Module for VisionAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let qkv = self.qkv.forward(xs)?;
        let qkv = qkv.reshape((b, n, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, B, num_heads, N, head_dim)
        
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        
        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn = q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
        let attn = (attn * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        
        let out = attn.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.contiguous()?; // (B, N, num_heads, head_dim)
        let out = out.reshape((b, n, self.num_heads * self.head_dim))?;
        
        self.proj.forward(&out)
    }
}

/// Qwen2.5-VL Vision MLP
#[derive(Debug)]
pub struct VisionMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl VisionMLP {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        let act_fn = Activation::Silu;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for VisionMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        let intermediate = (gate * up)?;
        self.down_proj.forward(&intermediate)
    }
}

/// Qwen2.5-VL Vision Block
#[derive(Debug)]
pub struct VisionBlock {
    norm1: RmsNorm,
    attn: VisionAttention,
    norm2: RmsNorm,
    mlp: VisionMLP,
}

impl VisionBlock {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = rms_norm(cfg.hidden_size, cfg.initializer_range, vb.pp("norm1"))?;
        let attn = VisionAttention::new(cfg, vb.pp("attn"))?;
        let norm2 = rms_norm(cfg.hidden_size, cfg.initializer_range, vb.pp("norm2"))?;
        let mlp = VisionMLP::new(cfg, vb.pp("mlp"))?;
        
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }
}

impl Module for VisionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs)?;
        let xs = (xs + residual)?;
        
        let residual = &xs;
        let xs = self.norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

/// Qwen2.5-VL Patch Merger - matches Python implementation
#[derive(Debug)]
pub struct PatchMerger {
    ln_q: RmsNorm,
    mlp: Vec<Linear>,
    spatial_merge_size: usize,
}

impl PatchMerger {
    pub fn new(vision_cfg: &VisionConfig, _text_cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let ln_q = rms_norm(vision_cfg.hidden_size, 1e-6, vb.pp("ln_q"))?;
        
        // Based on the model inspection: 5120 -> 5120 -> 2048
        // This corresponds to 4 * 1280 = 5120 (spatial merge of 2x2 * hidden_size)
        let spatial_merge_unit = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
        let input_dim = vision_cfg.hidden_size * spatial_merge_unit; // 1280 * 4 = 5120
        
        // Sequential MLP: 5120 -> 5120 -> 2048 with GELU activation
        let mlp = vec![
            linear(input_dim, input_dim, vb.pp("mlp").pp("0"))?,
            linear(input_dim, vision_cfg.out_hidden_size, vb.pp("mlp").pp("2"))?,
        ];
        
        Ok(Self { 
            ln_q, 
            mlp, 
            spatial_merge_size: vision_cfg.spatial_merge_size 
        })
    }
}

impl PatchMerger {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Apply layer norm first
        let xs = self.ln_q.forward(xs)?;
        
        // Spatial merge: merge spatial_merge_size x spatial_merge_size patches
        let (b, n, c) = xs.dims3()?;
        let h_w = (n as f64).sqrt() as usize;
        let h = h_w;
        let w = h_w;
        
        // Ensure dimensions are divisible by spatial_merge_size
        if h % self.spatial_merge_size != 0 || w % self.spatial_merge_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "Height ({}) and width ({}) must be divisible by spatial_merge_size ({})", 
                h, w, self.spatial_merge_size
            )));
        }
        
        // Reshape to spatial format: (B, H, W, C)
        let xs = xs.reshape((b, h, w, c))?;
        
        // Merge spatial_merge_size x spatial_merge_size patches
        let new_h = h / self.spatial_merge_size;
        let new_w = w / self.spatial_merge_size;
        let mut merged_patches = Vec::new();
        
        for i in 0..new_h {
            for j in 0..new_w {
                let mut patch_parts = Vec::new();
                for di in 0..self.spatial_merge_size {
                    for dj in 0..self.spatial_merge_size {
                        let patch = xs.i((
                            0..b, 
                            i * self.spatial_merge_size + di..i * self.spatial_merge_size + di + 1,
                            j * self.spatial_merge_size + dj..j * self.spatial_merge_size + dj + 1,
                            0..c
                        ))?.squeeze(1)?.squeeze(1)?;
                        patch_parts.push(patch);
                    }
                }
                let merged_patch = Tensor::cat(&patch_parts, D::Minus1)?;
                merged_patches.push(merged_patch);
            }
        }
        
        let xs = Tensor::stack(&merged_patches, 1)?; // (B, new_h * new_w, spatial_merge_size^2 * C)
        
        // Apply MLP with GELU activation - ensure contiguous tensors
        let xs = self.mlp[0].forward(&xs.contiguous()?)?;
        let xs = Activation::Gelu.forward(&xs)?;
        self.mlp[1].forward(&xs.contiguous()?)
    }
}

/// Qwen2.5-VL Vision Transformer
#[derive(Debug)]
pub struct VisionTransformer {
    patch_embed: VisionPatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    dtype: candle_core::DType,
}

impl VisionTransformer {
    pub fn new(vision_cfg: &VisionConfig, text_cfg: &TextConfig, vb: VarBuilder, dtype: candle_core::DType) -> Result<Self> {
        let patch_embed = VisionPatchEmbed::new(vision_cfg, vb.pp("patch_embed"))?;
        
        let mut blocks = Vec::with_capacity(vision_cfg.depth);
        let vb_blocks = vb.pp("blocks");
        for i in 0..vision_cfg.depth {
            blocks.push(VisionBlock::new(vision_cfg, vb_blocks.pp(i))?);
        }
        
        let merger = PatchMerger::new(vision_cfg, text_cfg, vb.pp("merger"))?;
        
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            dtype,
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Ensure input is in correct dtype
        let xs = xs.to_dtype(self.dtype)?;
        let xs = self.patch_embed.forward(&xs)?;
        
        let mut xs = xs;
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        
        self.merger.forward(&xs)
    }
}

impl VisionTransformer {
    /// Forward method for pre-flattened patches from transformers preprocessing
    pub fn forward_flattened_patches(&self, patches: &Tensor) -> Result<Tensor> {
        // Input: flattened patches (num_patches, channel * temporal_patch_size * patch_size * patch_size)  
        // Need to convert to: (batch_size, num_patches, hidden_size)
        
        let patches = patches.to_dtype(self.dtype)?;
        
        // Convert patch features to hidden dimension using patch embedding weights
        // This replicates what VisionPatchEmbed does but for already-flattened patches
        let patch_feature_dim = patches.dim(1)?;
        let expected_patch_dim = 3 * 2 * 14 * 14; // channel * temporal_patch_size * patch_size * patch_size
        
        if patch_feature_dim != expected_patch_dim {
            return Err(candle_core::Error::Msg(format!(
                "Expected patch feature dim {}, got {}", 
                expected_patch_dim, patch_feature_dim
            )));
        }
        
        // Apply patch embedding projection: (num_patches, patch_feature_dim) -> (num_patches, hidden_size)
        // Use the Conv3d weights from patch_embed but as a 2D linear transformation
        let weight_2d = self.patch_embed.weight.reshape((self.patch_embed.hidden_size, patch_feature_dim))?;
        let projected = patches.matmul(&weight_2d.t()?.contiguous()?)?; // (num_patches, hidden_size)
        
        // Add bias if present
        let mut xs = if let Some(bias) = &self.patch_embed.bias {
            let bias_expanded = bias.unsqueeze(0)?.broadcast_as(projected.shape())?;
            projected.add(&bias_expanded)?
        } else {
            projected
        };
        
        // Add batch dimension: (num_patches, hidden_size) -> (1, num_patches, hidden_size)
        xs = xs.unsqueeze(0)?;
        
        // Process through transformer blocks
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        
        // Apply merger (spatial merge for ColPali)
        let merged = self.merger.forward(&xs)?;
        
        // Remove batch dimension to match expected 2D output: (1, num_patches, hidden_size) -> (num_patches, hidden_size)
        merged.squeeze(0)
    }
}