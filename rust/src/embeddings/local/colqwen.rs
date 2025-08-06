use crate::embeddings::embed::{EmbedData, EmbeddingResult};
use crate::embeddings::select_device;
use crate::models::colqwen::Model;
use crate::models::qwen25_vl::Config;
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

        let (tokenizer_filename, weights_filename, config_filename) = {
            let tokenizer = repo.get("tokenizer.json")?;
            let weights = hub_load_safetensors(&repo, "model.safetensors.index.json")?;
            let config = repo.get("config.json")?;

            (tokenizer, weights, config)
        };

        // Load config from file
        let config_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_str)?;

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

        let _dummy_input: Tensor = tokenize_batch(&tokenizer, vec![dummy_prompt], &device)?;

        Ok(Self {
            model: RwLock::new(model),
            tokenizer,
            config,
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

        let _dummy_input: Tensor = tokenize_batch(&tokenizer, vec![dummy_prompt], &device)?;

        Ok(Self {
            model: RwLock::new(model),
            tokenizer,
            config,
            device,
            dtype,
        })
    }

    fn images_to_tensor(
        &self,
        pages: &[DynamicImage],
        image_size: usize,
    ) -> anyhow::Result<Tensor> {
        if pages.is_empty() {
            return Err(anyhow::anyhow!("Cannot process empty pages array"));
        }
        
        let mut images = vec![];
        for page in pages.iter() {
            let img = page.resize_to_fill(
                image_size as u32,
                image_size as u32,
                image::imageops::FilterType::Triangle,
            );
            let img = img.to_rgb8();
            let img = img.into_raw();
            
            // Create tensor following ColPali's approach but with temporal dimension for Qwen2.5-VL
            let img = Tensor::from_vec(img, (image_size, image_size, 3), &self.device)?
                .permute((2, 0, 1))?  // (C, H, W)
                .to_dtype(self.dtype)?
                .affine(2. / 255., -1.)?; // Normalize to [-1, 1]
            
            // Add temporal dimension: (C, H, W) -> (C, T, H, W) where T=1
            let img = img.unsqueeze(1)?; // (C, 1, H, W)
            images.push(img);
        }
        
        let images = Tensor::stack(&images, 0)?; // (B, C, T, H, W)
        Ok(images)
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
            let tokens = tokenize_batch(&self.tokenizer, batch.to_vec(), &self.device)?;
            
            let model = self.model.read().unwrap();
            let embeddings = model.get_query_embeddings(&tokens)?;
            
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
        let tokens = tokenize_batch(&self.tokenizer, vec![query], &self.device)?;
        let model = self.model.read().unwrap();
        let embeddings = model.get_query_embeddings(&tokens)?;
        
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
                    
                    // For Qwen2.5-VL, use a reasonable image size
                    // The calculated size (112 * 14 = 1568) might be too large, let's use a more reasonable size
                    // Based on typical VLM expectations, 448-896 is common
                    let image_size = 448; // Standard size for many vision-language models
                    let images_tensor = self.images_to_tensor(page_batch, image_size)?;
                    
                    // Create dummy text tokens for image processing
                    let dummy_text = vec!["Describe the image."; page_batch.len()];
                    let tokens = tokenize_batch(&self.tokenizer, dummy_text, &self.device)?;
                    
                    let model = self.model.read().unwrap();
                    let embeddings = model.get_document_embeddings(&tokens, &images_tensor)?;
                    
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
        
        // For Qwen2.5-VL, use a reasonable image size
        // The calculated size (112 * 14 = 1568) might be too large, let's use a more reasonable size
        // Based on typical VLM expectations, 448-896 is common
        let image_size = 448; // Standard size for many vision-language models
        let images_tensor = self.images_to_tensor(&[image], image_size)?;
        
        // Create dummy text token for image processing
        let tokens = tokenize_batch(&self.tokenizer, vec!["Describe the image."], &self.device)?;
        
        let model = self.model.read().unwrap();
        let embeddings = model.get_document_embeddings(&tokens, &images_tensor)?;
        
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
        
        // For Qwen2.5-VL, use a reasonable image size
        // The calculated size (112 * 14 = 1568) might be too large, let's use a more reasonable size
        // Based on typical VLM expectations, 448-896 is common
        let image_size = 448; // Standard size for many vision-language models
        let images_tensor = self.images_to_tensor(&images, image_size)?;
        
        // Create dummy text tokens for image processing
        let dummy_text = vec!["Describe the image."; images.len()];
        let tokens = tokenize_batch(&self.tokenizer, dummy_text, &self.device)?;
        
        let model = self.model.read().unwrap();
        let embeddings = model.get_document_embeddings(&tokens, &images_tensor)?;
        
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

fn tokenize_batch(tokenizer: &Tokenizer, text_batch: Vec<&str>, device: &Device) -> candle_core::Result<Tensor> {
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

    Ok(Tensor::stack(&token_ids, 0)?)
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