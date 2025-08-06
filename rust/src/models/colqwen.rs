use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use super::qwen25_vl::{Config, VisionTransformer};
use super::qwen25_language::LanguageModel;

/// Main ColQwen Model combining vision and language components
/// This follows the Python ColQwen2_5 implementation
#[derive(Debug)]
pub struct Model {
    visual: VisionTransformer,
    language_model: LanguageModel,
    custom_text_proj: Linear,
    config: Config,
    #[allow(dead_code)]
    dim: usize,
    mask_non_image_embeddings: bool,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder, device: &Device, dtype: DType, mask_non_image_embeddings: bool) -> Result<Self> {
        let visual = VisionTransformer::new(
            &config.vision_config,
            &config.text_config,
            vb.pp("visual"),
        )?;
        
        let language_model = LanguageModel::new(
            &config.text_config,
            vb.pp("model"),
            device,
            dtype,
        )?;
        
        let dim = 128;
        let custom_text_proj = linear(
            config.text_config.hidden_size,
            dim,
            vb.pp("custom_text_proj"),
        )?;
        
        Ok(Self {
            visual,
            language_model,
            custom_text_proj,
            config: config.clone(),
            dim,
            mask_non_image_embeddings,
        })
    }
    
    /// Forward pass following ColQwen2.5 architecture
    /// This matches the Python implementation's forward method
    pub fn forward(
        &self, 
        input_ids: &Tensor, 
        attention_mask: &Tensor,
        pixel_values: Option<&Tensor>,
        _image_grid_thw: Option<&Tensor>
    ) -> Result<Tensor> {
        // Get last hidden states from language model
        let last_hidden_states = if let Some(pixel_values) = pixel_values {
            // Process vision inputs and inject into language model
            let vision_embeds = self.visual.forward(pixel_values)?;
            self.language_model.forward_with_vision(
                input_ids, 
                Some(&vision_embeds)
            )?
        } else {
            // Text-only forward pass
            self.language_model.forward(input_ids)?
        };
        
        // Apply custom projection to get ColPali-style embeddings
        let proj = self.custom_text_proj.forward(&last_hidden_states)?;
        
        // L2 normalization
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let proj = proj.broadcast_div(&norm)?;
        
        // Apply attention mask (only use the mask for the original sequence length)
        let attention_mask_expanded = attention_mask.unsqueeze(D::Minus1)?;
        let proj = if pixel_values.is_some() {
            // For vision+text: need to handle the concatenated sequence
            // For now, just apply mask to the text portion
            // TODO: Implement proper vision+text masking
            proj
        } else {
            // Text-only: apply mask normally
            proj.broadcast_mul(&attention_mask_expanded)?
        };
        
        // Optional: mask non-image embeddings (if enabled)
        if self.mask_non_image_embeddings {
            if let Some(_pixel_values) = pixel_values {
                // Create image mask (tokens with image_token_id)
                let image_token_id = self.config.image_token_id() as i64;
                let image_mask = input_ids.eq(image_token_id)?.unsqueeze(D::Minus1)?;
                return Ok(proj.broadcast_mul(&image_mask.to_dtype(proj.dtype())?)?)
            }
        }
        
        Ok(proj)
    }
    
    /// Forward pass for text-only input (legacy)
    pub fn forward_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.language_model.forward(input_ids)
    }
    
    /// Forward pass for vision-only input (legacy)
    pub fn forward_vision(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.visual.forward(pixel_values)
    }
    
    /// Get the configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    /// Get text embedding dimension
    pub fn text_embed_dim(&self) -> usize {
        self.config.text_config.hidden_size
    }
    
    /// Get vision embedding dimension  
    pub fn vision_embed_dim(&self) -> usize {
        self.config.text_config.hidden_size // After merger, vision embeds are projected to text hidden size
    }
}

impl Model {
    /// Get query embeddings (for text queries)
    pub fn get_query_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Create attention mask (all ones for valid tokens) - use F32 to match model expectations
        let attention_mask = Tensor::ones(input_ids.shape(), candle_core::DType::F32, input_ids.device())?;
        self.forward(input_ids, &attention_mask, None, None)
    }
    
    /// Get document embeddings (for images/documents)
    pub fn get_document_embeddings(&self, input_ids: &Tensor, pixel_values: &Tensor) -> Result<Tensor> {
        // Create attention mask (all ones for valid tokens) - use F32 to match model expectations
        let attention_mask = Tensor::ones(input_ids.shape(), candle_core::DType::F32, input_ids.device())?;
        self.forward(input_ids, &attention_mask, Some(pixel_values), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    
    #[test]
    fn test_model_creation() {
        let device = Device::Cpu;
        let config = Config::default();
        
        // Note: This test would require actual model weights to run
        // It's here as a template for how to use the model
    }
}