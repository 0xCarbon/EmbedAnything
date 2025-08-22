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
    /// Get query embeddings (for text queries) - ColPali style with attention mask support
    pub fn get_query_embeddings(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Follow ColPali pattern: direct language model call
        let last_hidden_states = self.language_model.forward(input_ids)?;
        
        // Apply custom projection to get ColPali-style embeddings
        let proj = self.custom_text_proj.forward(&last_hidden_states)?;
        
        // L2 normalization
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let mut proj = proj.broadcast_div(&norm)?;
        
        // Apply attention mask if provided (zeroes out padding tokens)
        if let Some(mask) = attention_mask {
            let expanded_mask = mask.unsqueeze(D::Minus1)?.broadcast_as(proj.shape())?;
            proj = proj.broadcast_mul(&expanded_mask)?;
        }
        
        Ok(proj)
    }
    
    /// Get document embeddings (for images/documents) - full ColPali approach with unpadding
    pub fn get_document_embeddings(
        &self, 
        input_ids: &Tensor, 
        pixel_values: &Tensor, 
        image_grid_thw: Option<&[(u32, u32, u32)]>,
        attention_mask: Option<&Tensor>,
        image_token_id: Option<u32>,
    ) -> Result<Tensor> {
        // Apply unpadding logic matching Python ColQwen2.5 - BEFORE vision processing
        let unpadded_pixel_values = if let Some(grid_thw) = image_grid_thw {
            self.apply_unpadding_to_patches(pixel_values, grid_thw)?
        } else {
            pixel_values.clone()
        };
        
        // Process vision inputs with properly unpadded flattened patches  
        let vision_embeds = self.visual.forward_flattened_patches(&unpadded_pixel_values)?;
        
        // Forward with vision embeddings
        let last_hidden_states = self.language_model.forward_with_vision(input_ids, Some(&vision_embeds))?;
        
        // Apply custom projection to get ColPali-style embeddings
        let proj = self.custom_text_proj.forward(&last_hidden_states)?;
        
        // L2 normalization
        let norm = proj.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let mut proj = proj.broadcast_div(&norm)?;
        
        // Apply attention mask if provided (zeroes out padding tokens)
        if let Some(mask) = attention_mask {
            // If we integrated vision embeddings, we need to expand the attention mask
            let adjusted_mask = if let Some(_) = image_grid_thw {
                // Vision embeddings were integrated, so we need to expand the mask
                // Vision tokens get mask value of 1.0 (always attend), text tokens use original mask
                let (batch_size, original_seq_len) = mask.dims2()?;
                let (_, current_seq_len, _) = proj.dims3()?;
                
                if current_seq_len > original_seq_len {
                    // Calculate number of vision tokens added
                    let num_vision_tokens = current_seq_len - original_seq_len;
                    
                    // Create vision mask (all 1s for vision tokens)
                    let vision_mask = Tensor::ones((batch_size, num_vision_tokens), mask.dtype(), mask.device())?;
                    
                    // Concatenate: [vision_mask, original_mask]
                    Tensor::cat(&[&vision_mask, mask], 1)?
                } else {
                    mask.clone()
                }
            } else {
                mask.clone()
            };
            
            let expanded_mask = adjusted_mask.unsqueeze(D::Minus1)?.broadcast_as(proj.shape())?;
            proj = proj.broadcast_mul(&expanded_mask)?;
        }
        
        // Apply image token masking if requested (only pool image embeddings)
        if self.mask_non_image_embeddings {
            if let Some(img_token_id) = image_token_id {
                let image_mask = self.create_image_token_mask(input_ids, img_token_id)?;
                let expanded_image_mask = image_mask.unsqueeze(D::Minus1)?.broadcast_as(proj.shape())?;
                proj = proj.broadcast_mul(&expanded_image_mask)?;
            }
        }
        
        Ok(proj)
    }
    
    /// Apply unpadding logic to flattened patches matching Python ColQwen2.5
    fn apply_unpadding_to_patches(&self, pixel_values: &Tensor, image_grid_thw: &[(u32, u32, u32)]) -> Result<Tensor> {
        // Python ColQwen2.5: offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # grid_h * grid_w
        // Then: pixel_sequence[:offset] for each image
        
        let mut unpadded_sequences = Vec::new();
        let mut current_idx = 0;
        
        for &(_grid_t, grid_h, grid_w) in image_grid_thw {
            let num_patches = (grid_h * grid_w) as usize; // Matches Python: grid_h * grid_w (NOT including grid_t)
            
            // Extract patches for this image from the flattened patches tensor
            // pixel_values shape: (total_patches, patch_feature_dim)
            let image_patches = pixel_values.narrow(0, current_idx, num_patches)?;
            unpadded_sequences.push(image_patches);
            
            // Move to next image's patches
            current_idx += num_patches;
        }
        
        // Concatenate all unpadded sequences - matches Python torch.cat behavior
        if unpadded_sequences.is_empty() {
            Ok(pixel_values.clone())
        } else {
            Tensor::cat(&unpadded_sequences, 0)
        }
    }
    
    /// Create image token mask for selective embedding pooling
    fn create_image_token_mask(&self, input_ids: &Tensor, image_token_id: u32) -> Result<Tensor> {
        // Create mask where 1.0 = image token, 0.0 = other tokens
        let image_token_tensor = Tensor::new(&[image_token_id], input_ids.device())?
            .to_dtype(input_ids.dtype())?
            .broadcast_as(input_ids.shape())?;
        
        // Compare input_ids with image_token_id
        let mask = input_ids.eq(&image_token_tensor)?.to_dtype(self.dtype)?;
        Ok(mask)
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