use embed_anything::embeddings::local::colpali::ColPaliEmbed;
use embed_anything::embeddings::local::colpali::ColPaliEmbedder;
use embed_anything::embeddings::local::colpali_ort::OrtColPaliEmbedder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyResult;
use std::path::PathBuf;

use crate::EmbedData;
#[pyclass]
pub struct ColpaliModel {
    pub model: Box<dyn ColPaliEmbed + Send + Sync>,
}

#[pymethods]
impl ColpaliModel {
    #[new]
    #[pyo3(signature = (model_id, revision=None))]
    pub fn new(model_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let model = ColPaliEmbedder::new(model_id, revision)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None))]
    pub fn from_pretrained(model_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let model = ColPaliEmbedder::new(model_id, revision)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, path_in_repo=None))]
    pub fn from_pretrained_onnx(
        model_id: &str,
        revision: Option<&str>,
        path_in_repo: Option<&str>,
    ) -> PyResult<Self> {
        let model = OrtColPaliEmbedder::new(model_id, revision, path_in_repo)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    /// Create a ColPali model from local ONNX files
    #[staticmethod]
    #[pyo3(signature = (model_path, tokenizer_path, config_path=None))]
    pub fn from_local_files(
        model_path: &str,
        tokenizer_path: &str,
        config_path: Option<&str>,
    ) -> PyResult<Self> {
        let model_path = PathBuf::from(model_path);
        let tokenizer_path = PathBuf::from(tokenizer_path);
        let config_path = config_path.map(PathBuf::from);
        
        let model = OrtColPaliEmbedder::from_local_files(model_path, tokenizer_path, config_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model: Box::new(model),
        })
    }

    pub fn embed_file(&self, file_path: &str, batch_size: usize) -> PyResult<Vec<EmbedData>> {
        let embed_data = self
            .model
            .embed_file(file_path.into(), batch_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embed_data
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }

    pub fn embed_query(&self, query: &str) -> PyResult<Vec<EmbedData>> {
        let embed_data = self
            .model
            .embed_query(query)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(embed_data
            .into_iter()
            .map(|data| EmbedData { inner: data })
            .collect())
    }
}
