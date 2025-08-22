use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, rms_norm, Activation, Embedding, Linear, RmsNorm, VarBuilder};
use super::qwen25_vl::TextConfig;

/// RoPE (Rotary Position Embedding) for Qwen2.5
#[derive(Debug)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let base = cfg.rope_theta;
        
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?.to_dtype(dtype)?;
        
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let inv_freq = inv_freq.reshape((1, dim / 2))?;
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
        // x1/x2 shape: [batch, num_heads, seq_len, head_dim/2]
        // cos/sin shape: [seq_len, head_dim/2] -> need [1, 1, seq_len, head_dim/2]
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

/// Qwen2.5 Multi-Head Attention with Grouped Query Attention
#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
}

impl Attention {
    pub fn new(cfg: &TextConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = hidden_size / num_heads;
        
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        
        let rotary_emb = RotaryEmbedding::new(cfg, device, dtype)?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_key_value_heads,
            head_dim,
            rotary_emb,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?;
        
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
        
        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn_weights = q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
        let attn_weights = attn_weights.affine(scale, 0.0)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?; // (B, seq_len, num_heads, head_dim)
        let attn_output = attn_output.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        
        self.o_proj.forward(&attn_output)
    }
}

impl Attention {
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

/// Qwen2.5 MLP
#[derive(Debug)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        let act_fn = Activation::Silu;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.act_fn.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        let intermediate = gate.mul(&up)?;
        self.down_proj.forward(&intermediate)
    }
}

/// Qwen2.5 Decoder Layer
#[derive(Debug)]
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(cfg: &TextConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"), device, dtype)?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl Module for DecoderLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;
        
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

/// Qwen2.5 Language Model
#[derive(Debug)]
pub struct LanguageModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

impl LanguageModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i), device, dtype)?);
        }
        
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

impl Module for LanguageModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        
        self.norm.forward(&xs)
    }
}

impl LanguageModel {
    /// Forward with vision embeddings injected at proper token positions
    pub fn forward_with_vision(&self, input_ids: &Tensor, vision_embeds: Option<&Tensor>) -> Result<Tensor> {
        let text_embeds = self.embed_tokens.forward(input_ids)?;
        
        // If vision embeddings are provided, integrate them properly
        let xs = if let Some(vision_embeds) = vision_embeds {
            self.integrate_vision_embeddings(&text_embeds, vision_embeds, input_ids)?
        } else {
            text_embeds
        };
        
        // Process through transformer layers
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        
        self.norm.forward(&xs)
    }
    
    /// Integrate vision embeddings at proper positions (matching Qwen2.5-VL behavior)
    fn integrate_vision_embeddings(
        &self, 
        text_embeds: &Tensor, 
        vision_embeds: &Tensor, 
        input_ids: &Tensor
    ) -> Result<Tensor> {
        // Implement proper masked_scatter like Python Qwen2_5_VLModel
        // Python: inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        let (batch_size, seq_len, hidden_size) = text_embeds.dims3()?;
        let (num_vision_tokens, _) = vision_embeds.dims2()?;
        
        let image_token_id = 151655u32; // <|image_pad|> token ID
        
        // Create image mask: find positions where input_ids == image_token_id
        let image_mask = self.create_image_token_mask(input_ids, image_token_id)?;
        
        // Count image tokens to verify we have the right number
        let image_token_count = self.count_image_tokens(input_ids, image_token_id)?;
        
        if image_token_count == 0 {
            // No image tokens, return original text embeddings
            return Ok(text_embeds.clone());
        }
        
        // Verify we have the right number of vision embeddings
        if image_token_count != num_vision_tokens {
            eprintln!("Warning: image token count ({}) != vision embeddings ({})", 
                     image_token_count, num_vision_tokens);
        }
        
        // Implement masked_scatter: replace image token positions with vision embeddings
        let mut result = text_embeds.clone();
        
        for b in 0..batch_size {
            let mut vision_idx = 0;
            let batch_input_ids = input_ids.get(b)?; // (seq_len,)
            let input_data: Vec<u32> = batch_input_ids.to_vec1()?;
            
            for seq_idx in 0..seq_len {
                if input_data[seq_idx] == image_token_id && vision_idx < num_vision_tokens {
                    // Replace this position with vision embedding
                    let vision_emb = vision_embeds.get(vision_idx)?; // (hidden_size,)
                    
                    // Create a tensor to replace the embedding at [b, seq_idx, :]
                    let vision_emb_expanded = vision_emb.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, hidden_size)
                    
                    // Use slice assignment to replace the position
                    // result[b:b+1, seq_idx:seq_idx+1, :] = vision_emb_expanded
                    let start_indices = vec![b, seq_idx, 0];
                    let end_indices = vec![b + 1, seq_idx + 1, hidden_size];
                    
                    // Extract the current slice, replace it, then put it back
                    let mut result_data = result.flatten_all()?.to_vec1::<f32>()?;
                    let vision_data = vision_emb.to_vec1::<f32>()?;
                    
                    // Calculate the flat index for this position
                    let flat_idx = (b * seq_len * hidden_size) + (seq_idx * hidden_size);
                    
                    // Replace the data
                    for i in 0..hidden_size {
                        if flat_idx + i < result_data.len() && i < vision_data.len() {
                            result_data[flat_idx + i] = vision_data[i];
                        }
                    }
                    
                    // Reconstruct the tensor
                    result = Tensor::from_vec(result_data, (batch_size, seq_len, hidden_size), text_embeds.device())?;
                    
                    vision_idx += 1;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Create mask for image token positions
    fn create_image_token_mask(&self, input_ids: &Tensor, image_token_id: u32) -> Result<Tensor> {
        // Create boolean mask where True indicates image token positions
        let input_data: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let mask_data: Vec<f32> = input_data.iter()
            .map(|&id| if id == image_token_id { 1.0 } else { 0.0 })
            .collect();
        
        let mask = Tensor::from_vec(mask_data, input_ids.shape(), input_ids.device())?;
        Ok(mask)
    }
    
    /// Count number of image tokens in input_ids
    fn count_image_tokens(&self, input_ids: &Tensor, image_token_id: u32) -> Result<usize> {
        let input_data: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let count = input_data.iter().filter(|&&id| id == image_token_id).count();
        Ok(count)
    }

    
    /// Enhanced forward for proper image token replacement (future enhancement)
    pub fn forward_with_proper_vision_integration(
        &self, 
        input_ids: &Tensor, 
        vision_embeds: Option<&Tensor>,
        image_token_id: Option<u32>
    ) -> Result<Tensor> {
        let text_embeds = self.embed_tokens.forward(input_ids)?;
        
        let xs = if let (Some(vision_embeds), Some(img_token_id)) = (vision_embeds, image_token_id) {
            // Find image token positions and replace with vision embeddings
            self.replace_image_tokens(&text_embeds, vision_embeds, input_ids, img_token_id)?
        } else {
            text_embeds
        };
        
        // Process through transformer layers
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        
        self.norm.forward(&xs)
    }
    
    /// Replace image tokens with vision embeddings (proper Qwen2.5-VL approach)
    fn replace_image_tokens(
        &self,
        text_embeds: &Tensor,
        vision_embeds: &Tensor,
        input_ids: &Tensor,
        _image_token_id: u32
    ) -> Result<Tensor> {
        // This is a placeholder for the full implementation
        // In the real Qwen2.5-VL, vision embeddings replace image token positions
        // For now, fall back to the integration approach
        self.integrate_vision_embeddings(text_embeds, vision_embeds, input_ids)
    }
}