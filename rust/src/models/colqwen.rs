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
    dtype: DType,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder, device: &Device, dtype: DType, mask_non_image_embeddings: bool) -> Result<Self> {
        let visual = VisionTransformer::new(
            &config.vision_config,
            &config.text_config,
            vb.pp("visual"),
            dtype,
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
            dtype,
        })
    }
    
    // Note: Removed complex forward method - now using ColPali's simpler pattern
    // where get_query_embeddings and get_document_embeddings call the language model directly
    
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
    /// Get query embeddings (for text queries) - ColPali style without explicit attention masks
    pub fn get_query_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Follow ColPali pattern: direct language model call without attention masks
        let last_hidden_states = self.language_model.forward(input_ids)?;
        
        // Apply custom projection to get ColPali-style embeddings
        let proj = self.custom_text_proj.forward(&last_hidden_states)?;
        
        // L2 normalization
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let proj = proj.broadcast_div(&norm)?;
        
        Ok(proj)
    }
    
    /// Get document embeddings (for images/documents) - simplified approach
    pub fn get_document_embeddings(&self, input_ids: &Tensor, pixel_values: &Tensor) -> Result<Tensor> {
        // Process vision inputs
        let vision_embeds = self.visual.forward(pixel_values)?;
        
        // Forward with vision embeddings (simplified)
        let last_hidden_states = self.language_model.forward_with_vision(input_ids, Some(&vision_embeds))?;
        
        // Apply custom projection to get ColPali-style embeddings
        let proj = self.custom_text_proj.forward(&last_hidden_states)?;
        
        // L2 normalization
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let proj = proj.broadcast_div(&norm)?;
        
        Ok(proj)
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