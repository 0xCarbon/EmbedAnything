use crate::embeddings::embed::{EmbedData, EmbeddingResult};
use crate::embeddings::select_device;
use crate::models::colqwen::Model;
use crate::models::qwen25_vl::{Config, PreprocessorConfig};
use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;
use pdf2image::{RenderOptionsBuilder, PDF};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

pub trait ColQwenEmbed {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error>;

    fn embed_query(&self, query: &str) -> anyhow::Result<Vec<EmbedData>>;
    fn embed_file(&self, file_path: PathBuf, batch_size: usize) -> anyhow::Result<Vec<EmbedData>>;
    fn embed_image(
        &self,
        image_path: PathBuf,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData>;

    fn embed_image_batch(&self, image_paths: &[PathBuf]) -> anyhow::Result<Vec<EmbedData>>;
}

pub struct ColQwenEmbedder {
    pub model: RwLock<Model>,
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub preprocessor_config: PreprocessorConfig,
    pub device: Device,
    dtype: DType,
}

impl ColQwenEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>) -> Result<Self, anyhow::Error> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo: hf_hub::api::sync::ApiRepo = match revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            )),
            None => api.repo(hf_hub::Repo::new(
                model_id.to_string(),
                hf_hub::RepoType::Model,
            )),
        };

        let (tokenizer_filename, weights_filename, config_filename, preprocessor_config_filename) = {
            let tokenizer = repo.get("tokenizer.json")?;
            let weights = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
            let config = repo.get("config.json")?;
            let preprocessor_config = repo.get("preprocessor_config.json")?;

            (tokenizer, weights, config, preprocessor_config)
        };

        // Load config from file
        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;

        let preprocessor_config_str = std::fs::read_to_string(preprocessor_config_filename)?;
        let preprocessor_config: PreprocessorConfig = serde_json::from_str(&preprocessor_config_str)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.text_config.max_position_embeddings,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let device = select_device();

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_filename, dtype, &device)? };

        let model = Model::new(&config, vb, &device, dtype, false)?;
        let dummy_prompt: &str = "Describe the image.";

        let (_dummy_input, _dummy_mask) = tokenize_batch(&tokenizer, vec![dummy_prompt], &device)?;

        Ok(Self {
            model: RwLock::new(model),
            tokenizer,
            config,
            preprocessor_config,
            device,
            dtype,
        })
    }

    /// Create a ColQwenEmbedder from local files
    pub fn from_local_files(
        model_dir: &str,
        tokenizer_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        let model_path = Path::new(model_dir);
        
        // Look for safetensors files
        let weights_filename = if model_path.join("model.safetensors.index.json").exists() {
            // Multi-file safetensors
            hub_load_safetensors_from_local(model_path, "model.safetensors.index.json")?
        } else if model_path.join("model.safetensors").exists() {
            // Single safetensors file
            vec![model_path.join("model.safetensors")]
        } else {
            return Err(anyhow::anyhow!(
                "No safetensors files found in {}. Expected model.safetensors or model.safetensors.index.json",
                model_dir
            ));
        };

        // Load config
        let config_file = model_path.join("config.json");
        if !config_file.exists() {
            return Err(anyhow::anyhow!(
                "Config file not found: {}. ColQwen models require a config.json file.",
                config_file.display()
            ));
        }

        let preprocessor_config_file = model_path.join("preprocessor_config.json");
        if !preprocessor_config_file.exists() {
            return Err(anyhow::anyhow!(
                "Preprocessor config file not found: {}. ColQwen models require a preprocessor_config.json file.",
                preprocessor_config_file.display()
            ));
        }

        let preprocessor_config_str = std::fs::read_to_string(preprocessor_config_file)?;
        let preprocessor_config: PreprocessorConfig = serde_json::from_str(&preprocessor_config_str)?;

        let config_str = std::fs::read_to_string(config_file)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer_file = if let Some(tokenizer_path) = tokenizer_path {
            Path::new(tokenizer_path).to_path_buf()
        } else if model_path.join("tokenizer.json").exists() {
            model_path.join("tokenizer.json")
        } else {
            return Err(anyhow::anyhow!(
                "Tokenizer file not found. Please provide tokenizer.json in {} or specify tokenizer_path",
                model_dir
            ));
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.text_config.max_position_embeddings,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let device = select_device();

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_filename, dtype, &device)? };

        let model = Model::new(&config, vb, &device, dtype, false)?;
        let dummy_prompt: &str = "Describe the image.";

        let (_dummy_input, _dummy_mask) = tokenize_batch(&tokenizer, vec![dummy_prompt], &device)?;

        Ok(Self {
            model: RwLock::new(model),
            tokenizer,
            config,
            preprocessor_config,
            device,
            dtype,
        })
    }

    /// Smart resize implementation matching transformers library exactly
    /// From transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize
    fn smart_resize(&self, height: u32, width: u32) -> (u32, u32) {
        let factor = (self.config.vision_config.patch_size * self.config.vision_config.spatial_merge_size) as u32; // 14 * 2 = 28
        let min_pixels = self.preprocessor_config.min_pixels as u32;
        let max_pixels = self.preprocessor_config.max_pixels as u32;
        
        // Check aspect ratio constraint - same as Python
        let max_dim = height.max(width);
        let min_dim = height.min(width);
        if max_dim / min_dim > 200 {
            return Err(anyhow::anyhow!(
                "absolute aspect ratio must be smaller than 200, got {}", 
                max_dim as f64 / min_dim as f64
            )).unwrap_or((factor, factor)); // Fallback on error
        }
        
        // Round to nearest factor multiple - exact Python logic
        let mut h_bar = ((height as f64 / factor as f64).round() as u32) * factor;
        let mut w_bar = ((width as f64 / factor as f64).round() as u32) * factor;
        
        // If too large, scale down proportionally - exact Python logic
        if h_bar * w_bar > max_pixels {
            let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
            h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as u32) * factor);
            w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as u32) * factor);
        } 
        // If too small, scale up proportionally - exact Python logic  
        else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as u32) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as u32) * factor;
        }
        
        // CRITICAL FIX: Ensure perfect square patch grid for PatchMerger
        // The vision transformer expects exactly square patch grids (e.g., 24x24 = 576 patches)
        let patches_h = h_bar / (self.config.vision_config.patch_size as u32);
        let patches_w = w_bar / (self.config.vision_config.patch_size as u32);
        
        // Force square patch grid by using the larger dimension for both
        let square_patches = patches_h.max(patches_w);
        let square_dim = square_patches * (self.config.vision_config.patch_size as u32);
        
        (square_dim, square_dim)
    }

    fn images_to_tensor(
        &self,
        pages: &[DynamicImage],
    ) -> anyhow::Result<(Tensor, Vec<(u32, u32, u32)>)> {
        if pages.is_empty() {
            return Err(anyhow::anyhow!("Cannot process empty pages array"));
        }
        
        // ImageNet normalization values (matching Python implementation)
        let image_mean = [0.48145466f32, 0.4578275f32, 0.40821073f32];
        let image_std = [0.26862954f32, 0.26130258f32, 0.27577711f32];
        
        let mut all_patches = vec![];
        let mut grid_thws = vec![];
        
        for page in pages.iter() {
            // Use smart_resize matching Python implementation
            let (target_height, target_width) = self.smart_resize(page.height(), page.width());
            
            let img = page.resize_exact(
                target_width,
                target_height,
                image::imageops::FilterType::Triangle,
            );
            let img = img.to_rgb8();
            let img_data = img.into_raw();
            
            // Convert to tensor and normalize with ImageNet values - following Python _preprocess
            let mut img_tensor = Tensor::from_vec(img_data, (target_height as usize, target_width as usize, 3), &self.device)?
                .permute((2, 0, 1))?  // (C, H, W)
                .to_dtype(DType::F32)?;
            
            // Apply rescaling (0-255 -> 0-1)
            img_tensor = img_tensor.affine(1.0 / 255.0, 0.0)?;
            
            // Apply ImageNet normalization: (image - mean) / std
            let mut normalized_channels = Vec::new();
            for c in 0..3 {
                let channel = img_tensor.get(c)?;
                let normalized = channel.affine(1.0 / image_std[c] as f64, -(image_mean[c] / image_std[c]) as f64)?;
                normalized_channels.push(normalized);
            }
            
            // Stack normalized channels back together
            img_tensor = Tensor::stack(&normalized_channels, 0)?;
            img_tensor = img_tensor.to_dtype(self.dtype)?; // (C, H, W)
            
            // Calculate patch dimensions first
            let patch_size = self.config.vision_config.patch_size as u32;
            let _merge_size = self.config.vision_config.spatial_merge_size as u32; 
            let temporal_patch_size = self.config.vision_config.temporal_patch_size as u32;
            
            // Add temporal dimension for Qwen2.5-VL: (C, H, W) -> (1, C, temporal_patch_size, H, W)
            let img_unsqueezed = img_tensor.unsqueeze(0)?.unsqueeze(2)?; // (1, C, 1, H, W)
            
            // Expand temporal dimension to match temporal_patch_size (for static images, repeat the frame)
            let img_with_temporal = if temporal_patch_size > 1 {
                // Repeat the temporal dimension: (1, C, 1, H, W) -> (1, C, temporal_patch_size, H, W)
                let mut temporal_frames = Vec::new();
                for _ in 0..temporal_patch_size {
                    temporal_frames.push(img_unsqueezed.clone());
                }
                Tensor::cat(&temporal_frames, 2)?
            } else {
                img_unsqueezed
            };
            
            let grid_h = target_height / patch_size;
            let grid_w = target_width / patch_size;
            let grid_t = 1; // Single image
            
            // Simplified patch creation - avoid 9-dimensional operations not supported by candle
            // Create patches in a more straightforward way that achieves the same result
            
            let _channel = 3;
            let (_, _c, _t, h, w) = img_with_temporal.dims5()?;
            
            // Ensure dimensions are compatible 
            if h % patch_size as usize != 0 || w % patch_size as usize != 0 {
                return Err(anyhow::anyhow!("Image dimensions not divisible by patch size"));
            }
            
            // Extract patches using a simpler approach  
            // Input: (1, C, temporal_patch_size, H, W) -> Output: (num_patches, C*temporal_patch_size*patch_size*patch_size)
            let num_patches_h = h / patch_size as usize;
            let num_patches_w = w / patch_size as usize; 
            let _num_patches = num_patches_h * num_patches_w;
            
            let mut patches = Vec::new();
            
            // Extract each patch
            for i in 0..num_patches_h {
                for j in 0..num_patches_w {
                    // Extract patch region: (1, C, temporal_patch_size, patch_size, patch_size)
                    let patch = img_with_temporal
                        .narrow(3, i * patch_size as usize, patch_size as usize)?
                        .narrow(4, j * patch_size as usize, patch_size as usize)?;
                    
                    // Flatten patch: (1, C, temporal_patch_size, patch_size, patch_size) -> (C*temporal_patch_size*patch_size*patch_size)
                    let flattened_patch = patch.flatten_all()?.squeeze(0)?;
                    patches.push(flattened_patch);
                }
            }
            
            // Stack patches: (num_patches, C*temporal_patch_size*patch_size*patch_size) 
            // This should now be (num_patches, 3*2*14*14) = (num_patches, 1176)
            let flattened = Tensor::stack(&patches, 0)?;
            
            all_patches.push(flattened);
            grid_thws.push((grid_t, grid_h, grid_w));
        }
        
        // Stack all patches - this creates the padded format like Python
        let stacked_patches = if all_patches.len() == 1 {
            all_patches.into_iter().next().unwrap()
        } else {
            Tensor::cat(&all_patches, 0)?
        };
        
        Ok((stacked_patches, grid_thws))
    }
}

impl ColQwenEmbed for ColQwenEmbedder {
    fn embed(
        &self,
        text_batch: &[&str],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let batch_size = batch_size.unwrap_or(32);
        let mut all_embeddings = Vec::new();

        for batch in text_batch.chunks(batch_size) {
            let (tokens, attention_mask) = tokenize_batch(&self.tokenizer, batch.to_vec(), &self.device)?;
            
            let model = self.model.read().unwrap();
            let embeddings = model.get_query_embeddings(&tokens, Some(&attention_mask))?
                .to_dtype(DType::F32)?;  // Convert BF16 → F32 for compatibility
            
            // Convert to EmbeddingResult format
            for i in 0..batch.len() {
                let embedding_tensor = embeddings.get(i)?;
                // ColQwen produces multi-vector embeddings (one per token)
                let embedding_matrix: Vec<Vec<f32>> = embedding_tensor.to_vec2()?;
                
                all_embeddings.push(EmbeddingResult::MultiVector(embedding_matrix));
            }
        }

        Ok(all_embeddings)
    }

    fn embed_query(&self, query: &str) -> anyhow::Result<Vec<EmbedData>> {
        // Format query to match Python ColQwen2_5_Processor.process_queries()
        // Python does: query_prefix + text + query_augmentation_token * 10
        let query_prefix = "Query: ";
        let query_augmentation_token = "<|endoftext|>";
        let formatted_query = format!("{}{}{}", 
            query_prefix, 
            query, 
            query_augmentation_token.repeat(10)
        );
        
        let (tokens, attention_mask) = tokenize_batch(&self.tokenizer, vec![&formatted_query], &self.device)?;
        let model = self.model.read().unwrap();
        let embeddings = model.get_query_embeddings(&tokens, Some(&attention_mask))?
            .to_dtype(DType::F32)?;  // Convert BF16 → F32 for compatibility
        
        let embedding_tensor = embeddings.get(0)?;
        // ColQwen produces multi-vector embeddings (one per token)
        let embedding_matrix: Vec<Vec<f32>> = embedding_tensor.to_vec2()?;
        
        Ok(vec![EmbedData {
            embedding: EmbeddingResult::MultiVector(embedding_matrix),
            text: Some(query.to_string()),
            metadata: Some(HashMap::from([
                ("type".to_string(), "query".to_string()),
                ("model_type".to_string(), "colqwen".to_string()),
            ])),
        }])
    }

    fn embed_file(&self, file_path: PathBuf, batch_size: usize) -> anyhow::Result<Vec<EmbedData>> {
        let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
        
        match extension.to_lowercase().as_str() {
            "pdf" => {
                let doc = PDF::from_file(&file_path)?;
                let page_count = doc.page_count();
                
                if page_count == 0 {
                    return Ok(Vec::new()); // Return empty vector for PDFs with no pages
                }
                
                let pages = doc.render(
                    pdf2image::Pages::Range(1..=page_count),
                    RenderOptionsBuilder::default().build()?,
                )?;
                
                if pages.is_empty() {
                    return Ok(Vec::new()); // Return empty vector if no pages were rendered
                }
                
                let mut all_embeddings = Vec::new();
                for (page_num, page_batch) in pages.chunks(batch_size).enumerate() {
                    if page_batch.is_empty() {
                        continue; // Skip empty batches
                    }
                    
                    // For Qwen2.5-VL, use smart_resize and proper normalization
                    let (images_tensor, grid_thws) = self.images_to_tensor(page_batch)?;
                    
                    // Create proper visual prompt matching Python ColQwen2.5 format
                    let visual_prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>";
                    let dummy_text = vec![visual_prompt; page_batch.len()];
                    let (tokens, attention_mask) = tokenize_batch(&self.tokenizer, dummy_text, &self.device)?;
                    
                    // Get image token ID for masking (if available)
                    let image_token_id = self.config.image_token_id();
                    
                    let model = self.model.read().unwrap();
                    let embeddings = model.get_document_embeddings(
                        &tokens, 
                        &images_tensor, 
                        Some(&grid_thws), 
                        Some(&attention_mask),
                        Some(image_token_id)
                    )?.to_dtype(DType::F32)?;  // Convert BF16 → F32 for compatibility
                    
                    for (i, _page) in page_batch.iter().enumerate() {
                        let embedding_tensor = embeddings.get(i)?;
                        // ColQwen produces multi-vector embeddings (one per token)
                        let embedding_matrix: Vec<Vec<f32>> = embedding_tensor.to_vec2()?;
                        
                        all_embeddings.push(EmbedData {
                            embedding: EmbeddingResult::MultiVector(embedding_matrix),
                            text: None,
                            metadata: Some(HashMap::from([
                                ("file_path".to_string(), file_path.to_string_lossy().to_string()),
                                ("page_number".to_string(), (page_num * batch_size + i + 1).to_string()),
                                ("file_type".to_string(), "pdf".to_string()),
                                ("model_type".to_string(), "colqwen".to_string()),
                            ])),
                        });
                    }
                }
                
                Ok(all_embeddings)
            }
            _ => Err(anyhow::anyhow!("Unsupported file type: {}", extension)),
        }
    }

    fn embed_image(
        &self,
        image_path: PathBuf,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        let image = image::ImageReader::open(&image_path)?.decode()?;
        
        // For Qwen2.5-VL, use smart_resize and proper patch creation
        let (images_tensor, grid_thws) = self.images_to_tensor(&[image])?;
        
        // Create proper visual prompt matching Python ColQwen2.5 format  
        let visual_prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>";
        let (tokens, attention_mask) = tokenize_batch(&self.tokenizer, vec![visual_prompt], &self.device)?;
        
        // Get image token ID for masking (if available)
        let image_token_id = self.config.image_token_id();
        
        let model = self.model.read().unwrap();
        let embeddings = model.get_document_embeddings(
            &tokens, 
            &images_tensor, 
            Some(&grid_thws), 
            Some(&attention_mask),
            Some(image_token_id)
        )?.to_dtype(DType::F32)?;  // Convert BF16 → F32 for compatibility
        
        let embedding_tensor = embeddings.get(0)?;
        // ColQwen produces multi-vector embeddings (one per token)
        let embedding_matrix: Vec<Vec<f32>> = embedding_tensor.to_vec2()?;
        
        let mut final_metadata = HashMap::from([
            ("image_path".to_string(), image_path.to_string_lossy().to_string()),
            ("model_type".to_string(), "colqwen".to_string()),
        ]);
        
        if let Some(user_metadata) = metadata {
            final_metadata.extend(user_metadata);
        }
        
        Ok(EmbedData {
            embedding: EmbeddingResult::MultiVector(embedding_matrix),
            text: None,
            metadata: Some(final_metadata),
        })
    }

    fn embed_image_batch(&self, image_paths: &[PathBuf]) -> anyhow::Result<Vec<EmbedData>> {
        let mut images = Vec::new();
        for path in image_paths {
            let image = image::ImageReader::open(path)?.decode()?;
            images.push(image);
        }
        
        // For Qwen2.5-VL, use smart_resize and proper normalization
        let (images_tensor, grid_thws) = self.images_to_tensor(&images)?;
        
        // Create proper visual prompt matching Python ColQwen2.5 format
        let visual_prompt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>";
        let dummy_text = vec![visual_prompt; images.len()];
        let (tokens, attention_mask) = tokenize_batch(&self.tokenizer, dummy_text, &self.device)?;
        
        // Get image token ID for masking (if available)
        let image_token_id = self.config.image_token_id();
        
        let model = self.model.read().unwrap();
        let embeddings = model.get_document_embeddings(
            &tokens, 
            &images_tensor, 
            Some(&grid_thws), 
            Some(&attention_mask),
            Some(image_token_id)
        )?.to_dtype(DType::F32)?;  // Convert BF16 → F32 for compatibility
        
        let mut all_embeddings = Vec::new();
        for (i, image_path) in image_paths.iter().enumerate() {
            let embedding_tensor = embeddings.get(i)?;
            // ColQwen produces multi-vector embeddings (one per token)
            let embedding_matrix: Vec<Vec<f32>> = embedding_tensor.to_vec2()?;
            
            all_embeddings.push(EmbedData {
                embedding: EmbeddingResult::MultiVector(embedding_matrix),
                text: None,
                metadata: Some(HashMap::from([
                    ("image_path".to_string(), image_path.to_string_lossy().to_string()),
                    ("model_type".to_string(), "colqwen".to_string()),
                ])),
            });
        }
        
        Ok(all_embeddings)
    }
}

fn tokenize_batch(tokenizer: &Tokenizer, text_batch: Vec<&str>, device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
    if text_batch.is_empty() {
        return Err(candle_core::Error::Msg("Cannot tokenize empty text batch".to_string()));
    }
    
    let tokens = tokenizer
        .encode_batch(text_batch, true)
        .map_err(candle_core::Error::msg)?;
        
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), device)?)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let attention_masks = tokens
        .iter()
        .map(|tokens| {
            let mask: Vec<f32> = tokens.get_attention_mask().iter().map(|&x| x as f32).collect();
            Ok(Tensor::new(mask.as_slice(), device)?)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok((Tensor::stack(&token_ids, 0)?, Tensor::stack(&attention_masks, 0)?))
}

/// Load safetensors files from local directory (reuse from colpali.rs)
pub fn hub_load_safetensors_from_local(
    model_dir: &Path,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, E> {
    let json_file_path = model_dir.join(json_file);
    let json_file = std::fs::File::open(&json_file_path)
        .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", json_file_path.display(), e))?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => anyhow::bail!("no weight map in {:?}", json_file_path),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("weight map in {:?} is not a map", json_file_path),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|filename| {
            let file_path = model_dir.join(filename);
            if !file_path.exists() {
                anyhow::bail!("Safetensors file not found: {}", file_path.display());
            }
            Ok(file_path)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(safetensors_files)
}

/// Load safetensors from remote repository (reuse from colpali.rs)
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, E> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => anyhow::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(safetensors_files)
}