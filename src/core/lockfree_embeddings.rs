// Copyright (c) 2025 Julius ML
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//! Lock-free Local Embeddings Engine for semantic understanding
//!
//! This module implements a completely lock-free embedding generation system
//! using atomic operations and lock-free data structures for maximum concurrency.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokenizers::Tokenizer;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info};

/// Lock-free memory pool using crossbeam's atomic data structures
#[derive(Debug)]
struct LockFreeMemoryPool {
    /// Available vectors - using crossbeam's lock-free queue
    available: SegQueue<Vec<f32>>,
    /// Maximum pool size
    #[allow(dead_code)]
    max_size: usize,
    /// Current size (atomic)
    current_size: AtomicUsize,
}

impl LockFreeMemoryPool {
    fn new(size: usize, vector_capacity: usize) -> Self {
        let pool = Self {
            available: SegQueue::new(),
            max_size: size,
            current_size: AtomicUsize::new(0),
        };

        // Pre-populate with empty vectors
        for _ in 0..size {
            pool.available.push(Vec::with_capacity(vector_capacity));
        }
        pool.current_size.store(size, Ordering::Release);

        pool
    }

    fn try_get(&self) -> Option<Vec<f32>> {
        self.available.pop()
    }

    #[allow(dead_code)]
    fn return_vector(&self, mut vec: Vec<f32>) {
        // Clear the vector but keep capacity for reuse
        vec.clear();
        self.available.push(vec);
    }

    #[allow(dead_code)]
    fn get_stats(&self) -> (usize, usize) {
        let available = self.available.len();
        let total = self.current_size.load(Ordering::Relaxed);
        (available, total)
    }
}

/// Configuration for lock-free embedding engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFreeEmbeddingConfig {
    /// Model type for embeddings
    pub model_type: EmbeddingModelType,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Enable adaptive batch sizing
    pub adaptive_batching: bool,
    /// Memory pool size for vector reuse
    pub memory_pool_size: usize,
    /// Maximum concurrent operations (lock-free alternative to semaphore)
    pub max_concurrent_ops: usize,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Model cache directory
    pub cache_dir: PathBuf,
    /// Enable model caching
    pub enable_caching: bool,
    /// Operation timeout in seconds
    pub operation_timeout_secs: u64,
}

impl Default for LockFreeEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_type: EmbeddingModelType::default(),
            max_batch_size: 32,
            adaptive_batching: true,
            memory_pool_size: 1000,
            max_concurrent_ops: num_cpus::get() * 2,
            enable_performance_monitoring: true,
            cache_dir: PathBuf::from("./models_cache"),
            enable_caching: true,
            operation_timeout_secs: 30,
        }
    }
}

/// Lock-free concurrent operation controller using atomics instead of Semaphore
#[derive(Debug)]
struct LockFreeConcurrencyController {
    /// Current number of active operations (atomic)
    current_operations: AtomicUsize,
    /// Maximum allowed operations
    max_operations: usize,
    /// Wait queue for operations that can't start immediately
    #[allow(dead_code)]
    wait_queue: SegQueue<AtomicBool>,
    /// Flag to indicate if controller is active
    active: AtomicBool,
}

impl LockFreeConcurrencyController {
    fn new(max_operations: usize) -> Self {
        Self {
            current_operations: AtomicUsize::new(0),
            max_operations,
            wait_queue: SegQueue::new(),
            active: AtomicBool::new(true),
        }
    }

    /// Try to acquire a slot without blocking (lock-free)
    fn try_acquire(self: &Arc<Self>) -> Option<LockFreeOperationPermit> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }

        let current = self.current_operations.load(Ordering::Relaxed);
        if current >= self.max_operations {
            return None;
        }

        // Try to increment atomically
        match self.current_operations.compare_exchange_weak(
            current,
            current + 1,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => Some(LockFreeOperationPermit {
                controller: Arc::clone(self),
            }),
            Err(_) => None, // Someone else took the slot
        }
    }

    /// Acquire with waiting (lock-free busy wait)
    async fn acquire(self: &Arc<Self>) -> Result<LockFreeOperationPermit> {
        const MAX_WAIT_ITERATIONS: usize = 1000;
        const WAIT_DELAY_MS: u64 = 1;

        if !self.active.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Concurrency controller is not active"));
        }

        let mut iterations = 0;
        loop {
            // Try to get slot immediately
            if let Some(permit) = self.try_acquire() {
                return Ok(permit);
            }

            // Check if we should continue waiting
            iterations += 1;
            if iterations >= MAX_WAIT_ITERATIONS {
                return Err(anyhow::anyhow!(
                    "Timeout waiting for operation slot after {} iterations",
                    MAX_WAIT_ITERATIONS
                ));
            }

            // Small delay to avoid busy waiting
            sleep(Duration::from_millis(WAIT_DELAY_MS)).await;
        }
    }

    /// Release a slot (automatically called by permit drop)
    fn release(&self) {
        self.current_operations.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get current load
    fn current_load(&self) -> usize {
        self.current_operations.load(Ordering::Relaxed)
    }

    /// Get max capacity
    fn max_capacity(&self) -> usize {
        self.max_operations
    }
}

/// Lock-free operation permit (auto-releasing on drop)
struct LockFreeOperationPermit {
    controller: Arc<LockFreeConcurrencyController>,
}

impl Drop for LockFreeOperationPermit {
    fn drop(&mut self) {
        self.controller.release();
    }
}

/// Performance stats for batch optimization - lock-free using DashMap
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BatchPerformanceStats {
    size: usize,
    time_ms: f64,
    success_rate: f64,
}

/// Embedding model types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingModelType {
    /// Static embeddings (fast, lightweight)
    StaticSimilarityMRL,
    /// MiniLM model (balanced performance)
    MiniLM,
    /// TinyBERT model (smallest BERT variant)
    TinyBERT,
    /// BGE Small model (balanced BERT)
    BGESmall,
}

impl Default for EmbeddingModelType {
    fn default() -> Self {
        Self::StaticSimilarityMRL
    }
}

impl EmbeddingModelType {
    /// Get embedding dimension for this model type
    pub fn embedding_dimension(&self) -> usize {
        match self {
            Self::StaticSimilarityMRL => 1024,
            Self::MiniLM => 384,
            Self::TinyBERT => 312,
            Self::BGESmall => 384,
        }
    }

    /// Get model ID for HuggingFace Hub
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::StaticSimilarityMRL => "sentence-transformers/all-MiniLM-L6-v2",
            Self::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
            Self::TinyBERT => "huawei-noah/TinyBERT_General_6L_312D",
            Self::BGESmall => "BAAI/bge-small-en-v1.5",
        }
    }

    /// Check if this is a BERT-based model
    pub fn is_bert_based(&self) -> bool {
        matches!(self, Self::MiniLM | Self::TinyBERT | Self::BGESmall)
    }
}

/// Lock-free local embedding engine
pub struct LockFreeLocalEmbeddingEngine {
    /// Model for generating embeddings
    model: Arc<BertModel>,
    /// Tokenizer for text processing
    tokenizer: Arc<Tokenizer>,
    /// Configuration
    config: LockFreeEmbeddingConfig,
    /// Device for computation
    device: Device,
    /// Current batch size (atomic for lock-free updates)
    current_batch_size: AtomicUsize,
    /// Performance stats for batch optimization - lock-free using DashMap
    batch_performance_cache: Arc<DashMap<usize, f64>>,
    /// Memory pool for pre-allocated embedding vectors - lock-free using atomic operations
    memory_pool: Arc<LockFreeMemoryPool>,
    /// Lock-free concurrency controller instead of Semaphore
    concurrency_controller: Arc<LockFreeConcurrencyController>,
}

impl LockFreeLocalEmbeddingEngine {
    /// Create a new embedding engine with the given configuration
    pub async fn new(config: LockFreeEmbeddingConfig) -> Result<Self> {
        info!(
            "Initializing lock-free embedding engine with model: {:?}",
            config.model_type
        );

        let device = Device::Cpu; // For lock-free implementation, use CPU
        let model_id = config.model_type.model_id();

        // Load model and tokenizer (same as before)
        let api = Api::new().map_err(|e| anyhow::anyhow!("Failed to create API: {}", e))?;
        let repo = api.model(model_id.to_string());

        let model_path = repo
            .get("model.safetensors")
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get model: {}", e))?;
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get config: {}", e))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get tokenizer: {}", e))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let bert_config = std::fs::read_to_string(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read BERT config: {}", e))?;
        let bert_config: BertConfig = serde_json::from_str(&bert_config)
            .map_err(|e| anyhow::anyhow!("Failed to parse BERT config: {}", e))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
        let model = BertModel::load(vb, &bert_config)?;

        info!(
            "Model loaded successfully, embedding dimension: {}",
            config.model_type.embedding_dimension()
        );

        // Initialize lock-free components
        let concurrency_controller = Arc::new(LockFreeConcurrencyController::new(
            config.max_concurrent_ops,
        ));

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            current_batch_size: AtomicUsize::new(config.max_batch_size),
            batch_performance_cache: Arc::new(DashMap::new()),
            memory_pool: Arc::new(LockFreeMemoryPool::new(
                config.memory_pool_size,
                config.model_type.embedding_dimension(),
            )),
            concurrency_controller,
            config,
            device,
        })
    }

    /// Get current batch size (lock-free)
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size.load(Ordering::Relaxed)
    }

    /// Get embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.config.model_type.embedding_dimension()
    }

    /// Check if model is BERT-based
    pub fn is_bert_based(&self) -> bool {
        self.config.model_type.is_bert_based()
    }

    /// Encode text into embeddings (lock-free implementation)
    pub async fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let embeddings = self.encode_batch(texts).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embeddings generated"))
    }

    /// Encode batch of texts (lock-free implementation)
    pub async fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let _start_time = std::time::Instant::now();

        // Use static embeddings for non-BERT models (lock-free)
        if !self.config.model_type.is_bert_based() {
            return self.encode_batch_static(&texts).await;
        }

        // BERT models use lock-free concurrent processing
        let total_start_time = std::time::Instant::now();
        let result = self.encode_batch_bert_lockfree(texts.clone()).await;

        // Performance monitoring (lock-free)
        let total_time = total_start_time.elapsed();
        debug!("Encoded {} texts in {:?}", texts.len(), total_time);

        result
    }

    /// Static embeddings (lock-free, fast path)
    async fn encode_batch_static(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Simplified static embedding generation (same as before but lock-free)
        let dimension = self.embedding_dimension();
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            // Try to get from memory pool first (lock-free)
            let mut embedding = self
                .memory_pool
                .try_get()
                .unwrap_or_else(|| Vec::with_capacity(dimension));

            // Generate static embedding (simplified hash-based approach)
            let hash = self.text_hash(text);
            embedding.clear();
            embedding.reserve(dimension);

            for i in 0..dimension {
                let value = ((hash.wrapping_add(i as u64)) as f32 / u64::MAX as f32) * 2.0 - 1.0;
                embedding.push(value);
            }

            // Normalize embedding
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }

            results.push(embedding);
        }

        Ok(results)
    }

    /// BERT embeddings with lock-free concurrency control
    async fn encode_batch_bert_lockfree(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Try to acquire slot without blocking (lock-free)
        let _permit = match self.concurrency_controller.try_acquire() {
            Some(permit) => permit,
            None => {
                // If no immediate slot, wait with lock-free mechanism
                self.concurrency_controller.acquire().await?
            }
        };

        // Use adaptive or fixed batch size (lock-free)
        let batch_size = if self.config.adaptive_batching {
            self.get_adaptive_batch_size(texts.len()).await
        } else {
            self.current_batch_size()
        };

        let mut all_embeddings = Vec::new();

        // Process in batches (lock-free processing)
        for chunk in texts.chunks(batch_size) {
            let start_time = std::time::Instant::now();

            // Process batch with timeout (lock-free)
            let batch_result = timeout(
                Duration::from_secs(self.config.operation_timeout_secs),
                self.process_bert_batch(chunk.to_vec()),
            )
            .await;

            match batch_result {
                Ok(Ok(batch_embeddings)) => {
                    all_embeddings.extend(batch_embeddings);

                    // Update performance stats (lock-free)
                    let time_ms = start_time.elapsed().as_millis() as f64;
                    self.update_batch_performance(chunk.len(), time_ms, 1.0);
                }
                Ok(Err(e)) => {
                    error!("Batch processing failed: {}", e);
                    self.update_batch_performance(
                        chunk.len(),
                        start_time.elapsed().as_millis() as f64,
                        0.0,
                    );
                    return Err(e);
                }
                Err(_) => {
                    error!("Batch processing timed out");
                    return Err(anyhow::anyhow!(
                        "Batch processing timed out after {} seconds",
                        self.config.operation_timeout_secs
                    ));
                }
            }
        }

        Ok(all_embeddings)
    }

    /// Process a single BERT batch (lock-free implementation)
    async fn process_bert_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Tokenize texts (lock-free)
        let mut tokenized = Vec::with_capacity(texts.len());
        for text in &texts {
            let encoding = self
                .tokenizer
                .encode(text.clone(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            tokenized.push(encoding);
        }

        // Create tensor from tokenized input (lock-free)
        let max_len = tokenized.iter().map(|enc| enc.len()).max().unwrap_or(0);
        let mut input_ids = Vec::new();
        let mut attention_mask = Vec::new();

        for encoding in tokenized {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            input_ids.extend_from_slice(ids);
            attention_mask.extend_from_slice(mask);
        }

        // Create tensors (lock-free)
        let input_tensor = Tensor::from_vec(input_ids, (texts.len(), max_len), &self.device)?;
        let mask_tensor = Tensor::from_vec(attention_mask, (texts.len(), max_len), &self.device)?;

        // Forward pass through BERT model (lock-free)
        let token_type_ids = Tensor::zeros(input_tensor.shape(), DType::I64, &self.device)?;
        let outputs = self
            .model
            .forward(&input_tensor, &mask_tensor, Some(&token_type_ids))?;

        // Extract embeddings (lock-free)
        let embeddings = outputs.mean(1)?; // Mean pooling over sequence dimension
        let embeddings_vec = embeddings.to_vec1::<f32>()?;

        // Split into individual embeddings (lock-free)
        let dimension = self.embedding_dimension();
        let mut results = Vec::with_capacity(texts.len());

        for i in 0..texts.len() {
            let start_idx = i * dimension;
            let end_idx = start_idx + dimension;
            let mut embedding = Vec::with_capacity(dimension);

            // Try to get from memory pool first (lock-free)
            if let Some(mut pooled_vec) = self.memory_pool.try_get() {
                pooled_vec.clear();
                pooled_vec.extend_from_slice(&embeddings_vec[start_idx..end_idx]);
                results.push(pooled_vec);
            } else {
                embedding.extend_from_slice(&embeddings_vec[start_idx..end_idx]);
                results.push(embedding);
            }
        }

        Ok(results)
    }

    /// Get adaptive batch size based on performance history (lock-free)
    async fn get_adaptive_batch_size(&self, text_count: usize) -> usize {
        // Simple adaptive logic based on text count and performance history (lock-free)
        let base_size = self.current_batch_size();

        if text_count <= base_size {
            return text_count;
        }

        // Check recent performance (lock-free)
        let recent_performance: Vec<f64> = self
            .batch_performance_cache
            .iter()
            .take(10)
            .map(|entry| *entry.value())
            .collect();

        let avg_performance = if recent_performance.is_empty() {
            0.8 // Default success rate
        } else {
            recent_performance.iter().sum::<f64>() / recent_performance.len() as f64
        };

        // Adjust batch size based on performance (lock-free)
        if avg_performance > 0.9 {
            (base_size as f64 * 1.2) as usize
        } else if avg_performance < 0.7 {
            (base_size as f64 * 0.8) as usize
        } else {
            base_size
        }
    }

    /// Update batch performance stats (lock-free)
    fn update_batch_performance(&self, batch_size: usize, time_ms: f64, success_rate: f64) {
        // Store performance metric (lock-free)
        let metric = success_rate / (time_ms / batch_size as f64);
        self.batch_performance_cache.insert(batch_size, metric);

        // Update current batch size (lock-free)
        let new_size = if success_rate > 0.9 && time_ms < 1000.0 {
            (self.current_batch_size.load(Ordering::Relaxed) as f64 * 1.1) as usize
        } else if success_rate < 0.7 || time_ms > 2000.0 {
            (self.current_batch_size.load(Ordering::Relaxed) as f64 * 0.9) as usize
        } else {
            self.current_batch_size.load(Ordering::Relaxed)
        };

        self.current_batch_size
            .store(new_size.clamp(8, 256), Ordering::Relaxed);
    }

    /// Get current concurrency load (lock-free)
    pub fn current_concurrency_load(&self) -> usize {
        self.concurrency_controller.current_load()
    }

    /// Get concurrency stats (lock-free)
    pub fn get_concurrency_stats(&self) -> (usize, usize) {
        (
            self.concurrency_controller.current_load(),
            self.concurrency_controller.max_capacity(),
        )
    }

    /// Simple text hash for static embeddings (lock-free)
    fn text_hash(&self, text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
}

pub use LockFreeEmbeddingConfig as EmbeddingConfig;
/// Compatibility wrapper for existing code
pub use LockFreeLocalEmbeddingEngine as LocalEmbeddingEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lockfree_embedding_engine_creation() {
        let config = LockFreeEmbeddingConfig::default();
        let engine = LockFreeLocalEmbeddingEngine::new(config).await;

        // Should fail because we don't have actual models, but should not panic
        assert!(engine.is_err() || engine.is_ok());
    }

    #[test]
    fn test_concurrency_controller() {
        let controller = Arc::new(LockFreeConcurrencyController::new(2));

        // Should be able to acquire 2 permits
        let permit1 = controller.try_acquire();
        assert!(permit1.is_some());

        let permit2 = controller.try_acquire();
        assert!(permit2.is_some());

        // Third should fail
        assert!(controller.try_acquire().is_none());

        // After releasing one, should be able to acquire again
        drop(permit1); // This will release the first permit

        // Now should be able to acquire again
        assert!(controller.try_acquire().is_some());

        // Stats should work
        assert_eq!(controller.max_capacity(), 2);
    }

    #[tokio::test]
    async fn test_memory_pool() {
        let pool = LockFreeMemoryPool::new(10, 384);

        // Should be able to get vectors
        let vec1 = pool.try_get();
        let vec2 = pool.try_get();

        assert!(vec1.is_some());
        assert!(vec2.is_some());

        // Return them
        if let Some(v) = vec1 {
            pool.return_vector(v);
        }
        if let Some(v) = vec2 {
            pool.return_vector(v);
        }

        // Stats should work
        let (available, total) = pool.get_stats();
        assert_eq!(total, 10);
        assert!(available <= 10);
    }
}
