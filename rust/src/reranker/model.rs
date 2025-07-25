use std::sync::RwLock;

use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo};
use ndarray::Array2;
use ort::{
    execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session},
};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::local::bert::TokenizerConfig;
use crate::Dtype;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct RerankerResult {
    pub query: String,
    pub documents: Vec<DocumentRank>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DocumentRank {
    pub document: String,
    pub relevance_score: f32,
    pub rank: usize,
}

pub struct Reranker {
    model: RwLock<Session>,
    tokenizer: Tokenizer,
}

impl Reranker {
    pub fn new(model_id: &str, revision: Option<&str>, dtype: Dtype) -> Result<Self, E> {
        let (_, tokenizer_filename, weights_filename, tokenizer_config_filename) = {
            let api = Api::new().unwrap();
            let api = match revision {
                Some(rev) => api.repo(Repo::with_revision(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                    rev.to_string(),
                )),
                None => api.repo(hf_hub::Repo::new(
                    model_id.to_string(),
                    hf_hub::RepoType::Model,
                )),
            };
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let tokenizer_config = api.get("tokenizer_config.json")?;
            let weights = match dtype {
                Dtype::Q4F16 => api.get("onnx/model_q4f16.onnx")?,
                Dtype::F16 => api.get("onnx/model_fp16.onnx")?,
                Dtype::INT8 => api.get("onnx/model_int8.onnx")?,
                Dtype::Q4 => api.get("onnx/model_q4.onnx")?,
                Dtype::UINT8 => api.get("onnx/model_uint8.onnx")?,
                Dtype::BNB4 => api.get("onnx/model_bnb4.onnx")?,
                Dtype::F32 => api.get("onnx/model.onnx")?,
                Dtype::BF16 => api.get("onnx/model_bf16.onnx")?,
                Dtype::QUANTIZED => api.get("onnx/model_quantized.onnx")?,
            };
            (config, tokenizer, weights, tokenizer_config)
        };
        let tokenizer_config = std::fs::read_to_string(tokenizer_config_filename)?;
        let tokenizer_config: TokenizerConfig = serde_json::from_str(&tokenizer_config)?;
        // Set max_length to the minimum of max_length and model_max_length if both are present
        let max_length = match (
            tokenizer_config.max_length,
            tokenizer_config.model_max_length,
        ) {
            (Some(max_len), Some(model_max_len)) => std::cmp::min(max_len, model_max_len),
            (Some(max_len), None) => max_len,
            (None, Some(model_max_len)) => model_max_len,
            (None, None) => 128,
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            max_length,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let cuda = CUDAExecutionProvider::default();

        if !cuda.is_available()? {
            eprintln!("CUDAExecutionProvider is not available");
        } else {
            println!("Session is using CUDAExecutionProvider");
        }

        // Get physical core count (excluding hyperthreading)
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        // For CPU-bound workloads like ONNX inference, it's often better to use
        // physical cores rather than logical cores to avoid context switching overhead
        let optimal_threads = std::cmp::max(1, threads / 2);

        // Allow configuring optimization level via environment variable
        let optimization_level = match std::env::var("ONNX_OPTIMIZATION_LEVEL").as_deref() {
            Ok("0") => GraphOptimizationLevel::Disable,
            Ok("1") => GraphOptimizationLevel::Level1,
            Ok("2") => GraphOptimizationLevel::Level2,
            Ok("3") => GraphOptimizationLevel::Level3,
            _ => GraphOptimizationLevel::Level1, // Default to Level1 for stability
        };

        let model = Session::builder()?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?
            .with_optimization_level(optimization_level)?
            .with_intra_threads(optimal_threads)? // Use optimal thread count
            .with_inter_threads(1)? // Set inter-op parallelism to 1 when using GPU
            .commit_from_file(weights_filename)?;

        Ok(Reranker { model: RwLock::new(model), tokenizer })
    }

    pub fn compute_scores(
        &self,
        queries: Vec<&str>,
        documents: Vec<&str>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, E> {
        let pairs = queries
            .iter()
            .flat_map(|query| documents.iter().map(move |doc| (*query, *doc)))
            .collect::<Vec<_>>();
        let mut scores = Vec::with_capacity(pairs.len());
        let mut model_guard = self.model.write().unwrap();

        for pair in pairs.chunks(batch_size) {
            let input_ids = self.tokenize_batch_ndarray(pair)?;
            let attention_mask = self.get_attention_mask_ndarray(pair)?;
            let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)?;
            let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)?;
            let outputs = model_guard.run(ort::inputs!["input_ids" => input_ids_tensor, "attention_mask" => attention_mask_tensor])?;
            let logits = outputs["logits".to_string()]
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()?;
            scores.extend(
                logits
                    .outer_iter()
                    .flat_map(|row| row.to_vec())
                    .collect::<Vec<_>>(),
            );
        }
        let scores_tensor = Tensor::from_vec(
            scores.clone(),
            (queries.len(), documents.len()),
            &Device::Cpu,
        )?;
        let sigmoid_scores = candle_nn::ops::sigmoid(&scores_tensor).unwrap();
        Ok(sigmoid_scores.to_vec2::<f32>()?)
    }

    pub fn rerank(
        &self,
        queries: Vec<&str>,
        documents: Vec<&str>,
        batch_size: usize,
    ) -> Result<Vec<RerankerResult>, E> {
        let scores = self.compute_scores(queries.clone(), documents.clone(), batch_size)?;
        let mut reranker_results = Vec::new();
        for (i, query) in queries.iter().enumerate() {
            let scores = scores[i].clone();
            let mut indices: Vec<usize> = (0..scores.len()).collect();
            indices.sort_by(|&j, &k| {
                scores[k]
                    .partial_cmp(&scores[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let document_ranks = scores
                .iter()
                .enumerate()
                .map(|(p, score)| DocumentRank {
                    document: documents[p].to_string(),
                    relevance_score: *score,
                    rank: indices.iter().position(|&i| i == p).unwrap() + 1,
                })
                .collect::<Vec<_>>();

            reranker_results.push(RerankerResult {
                query: query.to_string(),
                documents: document_ranks,
            });
        }
        Ok(reranker_results)
    }

    pub fn tokenize_batch_ndarray(&self, pairs: &[(&str, &str)]) -> anyhow::Result<Array2<i64>> {
        let token_ids = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?
            .iter()
            .map(|tokens| {
                tokens
                    .get_ids()
                    .iter()
                    .map(|&id| id as i64)
                    .collect::<Vec<i64>>()
            })
            .collect::<Vec<Vec<i64>>>();

        let token_ids_array = Array2::from_shape_vec(
            (token_ids.len(), token_ids[0].len()),
            token_ids.into_iter().flatten().collect::<Vec<i64>>(),
        )
        .unwrap();
        Ok(token_ids_array)
    }

    pub fn get_attention_mask_ndarray(
        &self,
        pairs: &[(&str, &str)],
    ) -> anyhow::Result<Array2<i64>> {
        let attention_mask = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(E::msg)?
            .iter()
            .map(|tokens| {
                tokens
                    .get_attention_mask()
                    .iter()
                    .map(|&id| id as i64)
                    .collect::<Vec<i64>>()
            })
            .collect::<Vec<Vec<i64>>>();

        let attention_mask_array = Array2::from_shape_vec(
            (attention_mask.len(), attention_mask[0].len()),
            attention_mask.into_iter().flatten().collect::<Vec<i64>>(),
        )
        .unwrap();
        Ok(attention_mask_array)
    }
}
