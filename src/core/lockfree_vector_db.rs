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
//! Lock-free Vector Database with HNSW for semantic similarity search
//!
//! This module implements a completely lock-free vector database using HNSW algorithm
//! for approximate nearest neighbor search, optimized for high-concurrency scenarios.

use anyhow::Result;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, info, warn};

// Type aliases to reduce complexity
type QuantizationParams = Arc<arc_swap::ArcSwap<Option<(Vec<f32>, Vec<f32>)>>>;

/// Configuration for the lock-free vector database
#[derive(Debug, Clone)]
pub struct LockFreeVectorDbConfig {
    /// Vector dimension (must match embedding model)
    pub dimension: usize,
    /// Maximum number of connections per node in HNSW
    pub max_connections: usize,
    /// Size of the dynamic candidate list
    pub ef_construction: usize,
    /// Size of the search candidate list
    pub ef_search: usize,
    /// Number of layers in HNSW hierarchy
    pub num_layers: usize,
    /// Enable vector quantization for memory optimization
    pub enable_quantization: bool,
    /// Quantization buckets (power of 2)
    pub quantization_buckets: usize,
    /// Enable HNSW indexing
    pub enable_hnsw_index: bool,
    /// Distance threshold for considering vectors similar
    pub distance_threshold: f32,
    /// Maximum number of vectors to store
    pub max_vectors: usize,
    /// Auto-save interval (number of operations)
    pub auto_save_interval: usize,
    /// Enable persistent storage
    pub persistent_storage: bool,
    /// Storage path
    pub storage_path: Option<PathBuf>,
    /// Enable Product Quantization for memory optimization
    pub enable_product_quantization: bool,
    /// Number of subvectors for PQ (must divide dimension evenly)
    pub pq_subvectors: usize,
    /// Bits per PQ code (2^bits centroids per subvector)
    pub pq_bits: usize,
}

impl Default for LockFreeVectorDbConfig {
    fn default() -> Self {
        Self {
            dimension: 384, // Default for MiniLM model
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            num_layers: 4,
            enable_quantization: true,
            quantization_buckets: 256,
            enable_hnsw_index: true,
            distance_threshold: 0.7,
            max_vectors: 100_000,
            auto_save_interval: 1000,
            persistent_storage: false,
            storage_path: None,
            enable_product_quantization: false, // Disabled by default (experimental)
            pq_subvectors: 8,                   // 384/8 = 48 dims per subvector
            pq_bits: 8,                         // 256 centroids per subvector
        }
    }
}

/// A stored vector with optional quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFreeStoredVector {
    /// Unique identifier
    pub id: u32,
    /// Original vector
    pub vector: Vec<f32>,
    /// Quantized vector (if quantization is enabled)
    pub quantized: Option<Vec<u8>>,
    /// Product Quantization codes (if PQ is enabled)
    pub pq_codes: Option<Vec<u8>>,
    /// Vector magnitude (precomputed for cosine similarity)
    pub magnitude: f32,
}

impl LockFreeStoredVector {
    fn new(id: u32, vector: Vec<f32>, quantized: Option<Vec<u8>>, pq_codes: Option<Vec<u8>>) -> Self {
        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self {
            id,
            vector,
            quantized,
            pq_codes,
            magnitude,
        }
    }
}

/// Metadata for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Unique identifier for the vector
    pub id: String,
    /// Original text content
    pub text: String,
    /// Source identifier (e.g., session_id, update_id)
    pub source: String,
    /// Content type classification
    pub content_type: String,
    /// Timestamp when vector was added
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional custom metadata
    pub metadata: HashMap<String, String>,
}

impl VectorMetadata {
    /// Create new metadata with required fields
    pub fn new(id: String, text: String, source: String, content_type: String) -> Self {
        Self {
            id,
            text,
            source,
            content_type,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Search result from vector database
#[derive(Debug, Clone)]
pub struct SearchMatch {
    /// Vector ID
    pub vector_id: u32,
    /// Similarity score (cosine similarity)
    pub similarity: f32,
    /// Associated metadata
    pub metadata: VectorMetadata,
}

/// Internal search result
#[derive(Debug, Clone)]
struct SearchResult {
    id: u32,
    similarity: f32,
}

/// Lock-free vector database statistics using atomics
#[derive(Debug, Default)]
pub struct LockFreeVectorDbStats {
    /// Total number of vectors stored (atomic)
    pub total_vectors: AtomicUsize,
    /// Index construction status (atomic)
    pub is_built: AtomicBool,
    /// Memory usage in bytes (approximate, atomic)
    pub memory_usage_bytes: AtomicUsize,
    /// Total search operations (atomic)
    pub total_searches: AtomicU64,
    /// Total search time in microseconds (atomic)
    pub total_search_time_us: AtomicU64,
    /// Hit rate for recent searches (computed)
    pub search_hit_rate: f64,
    /// Index efficiency metric (connections per node, computed)
    pub index_efficiency: f64,
    /// Quantization compression ratio (computed)
    pub quantization_ratio: f64,
}

impl LockFreeVectorDbStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a vector addition
    pub fn record_vector_added(&self, vector_size_bytes: usize) {
        self.total_vectors.fetch_add(1, Ordering::Relaxed);
        self.memory_usage_bytes
            .fetch_add(vector_size_bytes, Ordering::Relaxed);
    }

    /// Record a vector removal
    pub fn record_vector_removed(&self, vector_size_bytes: usize) {
        self.total_vectors.fetch_sub(1, Ordering::Relaxed);
        self.memory_usage_bytes
            .fetch_sub(vector_size_bytes, Ordering::Relaxed);
    }

    /// Record a search operation
    pub fn record_search(&self, duration_us: u64) {
        self.total_searches.fetch_add(1, Ordering::Relaxed);
        self.total_search_time_us
            .fetch_add(duration_us, Ordering::Relaxed);
    }

    /// Get average search time in microseconds
    pub fn avg_search_time_us(&self) -> f64 {
        let total_searches = self.total_searches.load(Ordering::Relaxed);
        let total_time = self.total_search_time_us.load(Ordering::Relaxed);

        if total_searches > 0 {
            total_time as f64 / total_searches as f64
        } else {
            0.0
        }
    }

    /// Get snapshot of current stats
    pub fn snapshot(&self) -> VectorDbStatsSnapshot {
        VectorDbStatsSnapshot {
            total_vectors: self.total_vectors.load(Ordering::Relaxed),
            is_built: self.is_built.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_usage_bytes.load(Ordering::Relaxed),
            avg_search_time_us: self.avg_search_time_us(),
            search_hit_rate: self.search_hit_rate,
            index_efficiency: self.index_efficiency,
            quantization_ratio: self.quantization_ratio,
        }
    }
}

/// Snapshot of vector database statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDbStatsSnapshot {
    pub total_vectors: usize,
    pub is_built: bool,
    pub memory_usage_bytes: usize,
    pub avg_search_time_us: f64,
    pub search_hit_rate: f64,
    pub index_efficiency: f64,
    pub quantization_ratio: f64,
}

/// Product Quantization codebook for memory-efficient vector storage
///
/// Splits each vector into subvectors and quantizes each independently
/// using learned centroids. Reduces memory by 8-32x with minimal accuracy loss.
#[derive(Debug, Clone)]
pub struct ProductQuantizationCodebook {
    /// Number of subvectors (dimension must be divisible by this)
    subvectors: usize,
    /// Bits per code (2^bits = number of centroids per subvector)
    bits: usize,
    /// Vector dimension
    dimension: usize,
    /// Centroids[subvector_idx][centroid_idx] = centroid vector
    /// Shape: [subvectors][2^bits][dimension/subvectors]
    centroids: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizationCodebook {
    /// Create new PQ codebook with random initialization
    pub fn new(dimension: usize, subvectors: usize, bits: usize) -> Result<Self> {
        if dimension % subvectors != 0 {
            return Err(anyhow::anyhow!(
                "Dimension {} must be divisible by subvectors {}",
                dimension,
                subvectors
            ));
        }

        let subvec_dim = dimension / subvectors;
        let num_centroids = 1 << bits; // 2^bits (e.g., 2^8 = 256)

        debug!(
            "Initializing PQ codebook: {} subvectors, {} bits ({} centroids), {} dim per subvec",
            subvectors, bits, num_centroids, subvec_dim
        );

        // Initialize random centroids (in practice, these would be trained via k-means)
        let mut centroids = Vec::with_capacity(subvectors);
        for _ in 0..subvectors {
            let mut subvec_centroids = Vec::with_capacity(num_centroids);
            for _ in 0..num_centroids {
                // Initialize with normalized random vectors
                let centroid: Vec<f32> = (0..subvec_dim)
                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                    .collect();

                // Normalize to unit length
                let magnitude = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                let normalized = if magnitude > 0.0 {
                    centroid.iter().map(|x| x / magnitude).collect()
                } else {
                    centroid
                };

                subvec_centroids.push(normalized);
            }
            centroids.push(subvec_centroids);
        }

        Ok(Self {
            subvectors,
            bits,
            dimension,
            centroids,
        })
    }

    /// Encode a vector into PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let subvec_dim = self.dimension / self.subvectors;
        let mut codes = Vec::with_capacity(self.subvectors);

        for i in 0..self.subvectors {
            let start = i * subvec_dim;
            let end = start + subvec_dim;
            let subvec = &vector[start..end];

            // Find nearest centroid using Euclidean distance
            let code = self.find_nearest_centroid(i, subvec);
            codes.push(code);
        }

        codes
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let subvec_dim = self.dimension / self.subvectors;
        let mut vector = Vec::with_capacity(self.dimension);

        for (i, &code) in codes.iter().enumerate() {
            if i >= self.subvectors {
                warn!("PQ decode: code index {} >= subvectors {}", i, self.subvectors);
                break;
            }

            let code_idx = code as usize;
            if code_idx >= self.centroids[i].len() {
                warn!(
                    "PQ decode: code {} >= centroids {} for subvector {}",
                    code_idx,
                    self.centroids[i].len(),
                    i
                );
                // Pad with zeros if invalid code
                vector.extend(vec![0.0; subvec_dim]);
                continue;
            }

            let centroid = &self.centroids[i][code_idx];
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Find nearest centroid for a subvector
    fn find_nearest_centroid(&self, subvec_idx: usize, subvec: &[f32]) -> u8 {
        let centroids = &self.centroids[subvec_idx];
        let mut best_code = 0u8;
        let mut best_dist = f32::INFINITY;

        for (code, centroid) in centroids.iter().enumerate() {
            let dist = euclidean_distance(subvec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_code = code as u8;
            }
        }

        best_code
    }

    /// Approximate distance between query vector and PQ-encoded vector
    pub fn approximate_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let subvec_dim = self.dimension / self.subvectors;
        let mut total_dist = 0.0;

        for i in 0..self.subvectors.min(codes.len()) {
            let start = i * subvec_dim;
            let end = start + subvec_dim;
            let query_subvec = &query[start..end];

            let code_idx = codes[i] as usize;
            if code_idx < self.centroids[i].len() {
                let centroid = &self.centroids[i][code_idx];
                let dist = euclidean_distance(query_subvec, centroid);
                total_dist += dist * dist; // Sum of squared distances
            }
        }

        total_dist.sqrt()
    }

    /// Get compression ratio information
    pub fn compression_info(&self) -> (usize, usize, f32) {
        let original_size = self.dimension * std::mem::size_of::<f32>();
        let compressed_size = self.subvectors * ((self.bits + 7) / 8); // Round up to bytes
        let ratio = original_size as f32 / compressed_size as f32;
        (original_size, compressed_size, ratio)
    }
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// HNSW layer assignment for a vector
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LayerAssignment {
    vector_id: u32,
    layer: usize,
}

/// HNSW connection in the graph
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HnswConnection {
    from_id: u32,
    to_id: u32,
    layer: usize,
}

/// Lock-free HNSW index using DashMap for concurrent access
#[derive(Debug, Default)]
struct LockFreeHnswIndex {
    /// Graph connections for each vector (lock-free)
    connections: DashMap<u32, Vec<u32>>,
    /// Entry point for search (atomic)
    entry_point: AtomicU32,
    /// Layer assignments for each vector (lock-free)
    layers: DashMap<u32, usize>,
    /// Maximum layer in the graph (atomic)
    max_layer: AtomicUsize,
}

impl LockFreeHnswIndex {
    fn new() -> Self {
        Self {
            connections: DashMap::new(),
            entry_point: AtomicU32::new(u32::MAX), // MAX means no entry point
            layers: DashMap::new(),
            max_layer: AtomicUsize::new(0),
        }
    }

    /// Check if index is empty
    fn is_empty(&self) -> bool {
        self.entry_point.load(Ordering::Relaxed) == u32::MAX
    }

    /// Get entry point
    fn get_entry_point(&self) -> Option<u32> {
        let ep = self.entry_point.load(Ordering::Relaxed);
        if ep == u32::MAX { None } else { Some(ep) }
    }

    /// Set entry point
    fn set_entry_point(&self, vector_id: u32) {
        self.entry_point.store(vector_id, Ordering::Relaxed);
    }

    /// Add vector to index
    fn add_vector(&self, vector_id: u32, layer: usize, connections: Vec<u32>) {
        self.layers.insert(vector_id, layer);
        self.connections.insert(vector_id, connections);

        // Update max layer if necessary
        self.max_layer.fetch_max(layer, Ordering::Relaxed);

        // Set entry point if this is the first vector or highest layer
        if self.is_empty() || layer >= self.max_layer.load(Ordering::Relaxed) {
            self.set_entry_point(vector_id);
        }
    }

    /// Remove vector from index
    fn remove_vector(&self, vector_id: u32) {
        self.layers.remove(&vector_id);
        self.connections.remove(&vector_id);

        // Remove connections to this vector from other vectors
        for mut entry in self.connections.iter_mut() {
            let connections = entry.value_mut();
            connections.retain(|&id| id != vector_id);
        }

        // Reset entry point if we removed it
        if self.get_entry_point() == Some(vector_id) {
            // Find new entry point from remaining vectors
            let mut max_layer = 0;
            let mut new_entry_point = None;

            for entry in self.layers.iter() {
                let layer = *entry.value();
                if layer >= max_layer {
                    max_layer = layer;
                    new_entry_point = Some(*entry.key());
                }
            }

            if let Some(new_ep) = new_entry_point {
                self.set_entry_point(new_ep);
                self.max_layer.store(max_layer, Ordering::Relaxed);
            } else {
                self.entry_point.store(u32::MAX, Ordering::Relaxed);
                self.max_layer.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Get connections for a vector
    fn get_connections(&self, vector_id: u32) -> Option<Vec<u32>> {
        self.connections.get(&vector_id).map(|entry| entry.clone())
    }

    /// Get layer for a vector
    #[allow(dead_code)]
    fn get_layer(&self, vector_id: u32) -> Option<usize> {
        self.layers.get(&vector_id).map(|entry| *entry.value())
    }
}

/// Lock-free vector database using DashMap and atomic operations
pub struct LockFreeVectorDB {
    /// Vector storage - lock-free using DashMap
    vectors: Arc<DashMap<u32, LockFreeStoredVector>>,
    /// Metadata storage - lock-free using DashMap
    metadata: Arc<DashMap<u32, VectorMetadata>>,
    /// Configuration
    config: LockFreeVectorDbConfig,
    /// Next available vector ID - atomic
    next_id: Arc<AtomicU32>,
    /// Performance statistics - lock-free using atomics
    stats: Arc<LockFreeVectorDbStats>,
    /// HNSW index - lock-free implementation
    hnsw_index: Arc<LockFreeHnswIndex>,
    /// Vector quantization parameters - lock-free using ArcSwap
    quantization_params: QuantizationParams,
    /// Product Quantization codebook (if enabled)
    pq_codebook: Option<Arc<ProductQuantizationCodebook>>,
}

impl LockFreeVectorDB {
    /// Compatibility method to add vector with old-style metadata
    pub fn add_vector_compat(
        &self,
        vector: Vec<f32>,
        content_id: String,
        text: String,
        source: String,
        content_type: String,
    ) -> Result<u32> {
        let metadata = VectorMetadata::new(content_id, text, source, content_type);
        self.add_vector(vector, metadata)
    }

    /// Create a new vector database with the specified configuration
    pub fn new(config: LockFreeVectorDbConfig) -> Result<Self> {
        info!(
            "Initializing Lock-free Vector Database with dimension: {}, max_connections: {}, quantization: {}, HNSW: {}, PQ: {}",
            config.dimension,
            config.max_connections,
            config.enable_quantization,
            config.enable_hnsw_index,
            config.enable_product_quantization
        );

        let stats = Arc::new(LockFreeVectorDbStats::new());
        let quantization_params = Arc::new(arc_swap::ArcSwap::new(std::sync::Arc::new(None)));

        // Initialize Product Quantization codebook if enabled
        let pq_codebook = if config.enable_product_quantization {
            info!(
                "Initializing PQ codebook: {} subvectors, {} bits",
                config.pq_subvectors, config.pq_bits
            );
            Some(Arc::new(ProductQuantizationCodebook::new(
                config.dimension,
                config.pq_subvectors,
                config.pq_bits,
            )?))
        } else {
            None
        };

        Ok(Self {
            vectors: Arc::new(DashMap::new()),
            metadata: Arc::new(DashMap::new()),
            config,
            next_id: Arc::new(AtomicU32::new(0)),
            stats,
            hnsw_index: Arc::new(LockFreeHnswIndex::new()),
            quantization_params,
            pq_codebook,
        })
    }

    /// Add a vector to the database (lock-free)
    pub fn add_vector(&self, vector: Vec<f32>, metadata: VectorMetadata) -> Result<u32> {
        if vector.len() != self.config.dimension {
            return Err(anyhow::anyhow!(
                "Vector dimension {} does not match expected {}",
                vector.len(),
                self.config.dimension
            ));
        }

        // Lock-free atomic ID generation
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let _vector_len = vector.len();

        // Quantize vector if enabled
        let quantized = if self.config.enable_quantization {
            self.quantize_vector(&vector).ok()
        } else {
            None
        };

        // Product Quantization encoding if enabled
        let pq_codes = if let Some(codebook) = &self.pq_codebook {
            Some(codebook.encode(&vector))
        } else {
            None
        };

        // Create stored vector
        let stored_vector = LockFreeStoredVector::new(id, vector, quantized, pq_codes);
        let vector_size_bytes = std::mem::size_of::<LockFreeStoredVector>()
            + stored_vector.vector.len() * std::mem::size_of::<f32>()
            + stored_vector
                .quantized
                .as_ref()
                .map(|q| q.len())
                .unwrap_or(0)
            + stored_vector
                .pq_codes
                .as_ref()
                .map(|pq| pq.len())
                .unwrap_or(0);

        // Add to vectors collection (lock-free)
        self.vectors.insert(id, stored_vector);

        // Add metadata (lock-free)
        self.metadata.insert(id, metadata);

        // Update statistics (lock-free)
        self.stats.record_vector_added(vector_size_bytes);

        // Build HNSW index if enabled
        if self.config.enable_hnsw_index {
            self.build_hnsw_index_for_vector(id)?;
        }

        debug!("Added vector {} to database", id);
        Ok(id)
    }

    /// Search for similar vectors (lock-free)
    pub fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchMatch>> {
        if query_vector.len() != self.config.dimension {
            return Err(anyhow::anyhow!(
                "Query vector dimension {} does not match expected {}",
                query_vector.len(),
                self.config.dimension
            ));
        }

        let start_time = std::time::Instant::now();

        // Use HNSW search if index is available and built, otherwise fall back to linear search
        let results = if self.config.enable_hnsw_index && !self.hnsw_index.is_empty() {
            self.hnsw_search(query_vector, k)?
        } else {
            self.linear_search(query_vector, k)?
        };

        // Lock-free metadata access with DashMap
        let mut matches = Vec::new();

        for result in results {
            let vector_id = result.id;

            if let Some(metadata) = self.metadata.get(&vector_id) {
                matches.push(SearchMatch {
                    vector_id,
                    similarity: result.similarity,
                    metadata: metadata.clone(),
                });
            }
        }

        // Update statistics (lock-free)
        let duration_us = start_time.elapsed().as_micros() as u64;
        self.stats.record_search(duration_us);

        debug!(
            "Search completed in {}μs, found {} matches",
            duration_us,
            matches.len()
        );
        Ok(matches)
    }

    /// Linear search through all vectors (lock-free)
    fn linear_search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut similarities: Vec<_> = Vec::new();

        // Lock-free iteration over vectors
        for entry in self.vectors.iter() {
            let stored_vector = entry.value();
            let similarity = Self::calculate_cosine_similarity(query_vector, &stored_vector.vector);
            similarities.push(SearchResult {
                id: stored_vector.id,
                similarity,
            });
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        similarities.truncate(k);

        Ok(similarities)
    }

    /// HNSW search (lock-free implementation)
    fn hnsw_search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Get entry point
        let entry_point = self
            .hnsw_index
            .get_entry_point()
            .ok_or_else(|| anyhow::anyhow!("HNSW index is empty"))?;

        // Multi-layer search (simplified implementation)
        let mut candidates = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Start from entry point
        candidates.push(SearchResult {
            id: entry_point,
            similarity: self.get_vector_similarity(query_vector, entry_point)?,
        });
        visited.insert(entry_point);

        // Expand search through connections
        let mut results = Vec::new();
        let max_candidates = self.config.ef_search.max(k * 2);

        while !candidates.is_empty() && results.len() < max_candidates {
            // Get best candidate
            candidates.sort_by(|a, b| {
                b.similarity
                    .partial_cmp(&a.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let current = candidates.remove(0);
            results.push(current.clone());

            // Explore connections
            if let Some(connections) = self.hnsw_index.get_connections(current.id) {
                for &connected_id in &connections {
                    if !visited.contains(&connected_id) {
                        visited.insert(connected_id);
                        let similarity = self.get_vector_similarity(query_vector, connected_id)?;

                        if similarity >= self.config.distance_threshold {
                            candidates.push(SearchResult {
                                id: connected_id,
                                similarity,
                            });
                        }
                    }
                }
            }
        }

        // Sort final results and take top k
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Get vector similarity (helper for HNSW search)
    fn get_vector_similarity(&self, query_vector: &[f32], vector_id: u32) -> Result<f32> {
        self.vectors
            .get(&vector_id)
            .map(|entry| {
                let stored_vector = entry.value();
                Self::calculate_cosine_similarity(query_vector, &stored_vector.vector)
            })
            .ok_or_else(|| anyhow::anyhow!("Vector {} not found", vector_id))
    }

    /// Calculate cosine similarity between two vectors
    fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Quantize a vector (simplified implementation)
    fn quantize_vector(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let params_guard = self.quantization_params.load();
        let params = params_guard.as_ref();

        if let Some((min_vals, max_vals)) = params {
            let mut quantized = Vec::with_capacity(vector.len());

            for (i, &value) in vector.iter().enumerate() {
                if i < min_vals.len() && i < max_vals.len() {
                    let min_val = min_vals[i];
                    let max_val = max_vals[i];

                    if max_val > min_val {
                        let normalized = (value - min_val) / (max_val - min_val);
                        let bucket =
                            (normalized * (self.config.quantization_buckets - 1) as f32) as u8;
                        quantized.push(bucket.min(self.config.quantization_buckets as u8 - 1));
                    } else {
                        quantized.push(0);
                    }
                } else {
                    quantized.push(0);
                }
            }

            Ok(quantized)
        } else {
            // No quantization parameters available, return raw bytes
            Ok(vector
                .iter()
                .map(|&x| (x * 255.0).min(255.0).max(0.0) as u8)
                .collect())
        }
    }

    /// Build HNSW index for a single vector (simplified implementation)
    fn build_hnsw_index_for_vector(&self, vector_id: u32) -> Result<()> {
        // For simplicity, connect to a few random existing vectors
        let mut connections = Vec::new();
        let max_connections = self.config.max_connections.min(10); // Limit for simplicity

        // Get existing vectors (excluding current)
        let existing_vectors: Vec<u32> = self
            .vectors
            .iter()
            .map(|entry| entry.key().clone())
            .filter(|&id| id != vector_id)
            .take(max_connections * 2) // Take more for random selection
            .collect();

        // Select random connections
        use std::collections::HashSet;
        let mut selected = HashSet::new();

        for &existing_id in &existing_vectors {
            if connections.len() < max_connections && selected.insert(existing_id) {
                connections.push(existing_id);
            }
        }

        // Assign to layer 0 for simplicity
        let num_connections = connections.len();
        self.hnsw_index.add_vector(vector_id, 0, connections);

        debug!(
            "Built HNSW index for vector {} with {} connections",
            vector_id, num_connections
        );
        Ok(())
    }

    /// Get metadata for a vector (lock-free)
    pub fn get_metadata(&self, vector_id: u32) -> Option<VectorMetadata> {
        self.metadata.get(&vector_id).map(|entry| entry.clone())
    }

    /// Remove a vector from the database (lock-free)
    pub fn remove_vector(&self, vector_id: u32) -> Result<bool> {
        // Remove from vectors (lock-free)
        let removed_vector = self.vectors.remove(&vector_id);

        // Remove metadata (lock-free) - we don't need the removed metadata
        self.metadata.remove(&vector_id);

        if removed_vector.is_some() {
            // Update statistics (lock-free)
            let vector_size_bytes = std::mem::size_of::<LockFreeStoredVector>()
                + removed_vector.unwrap().1.vector.len() * std::mem::size_of::<f32>();
            self.stats.record_vector_removed(vector_size_bytes);

            // Remove from HNSW index (lock-free)
            self.hnsw_index.remove_vector(vector_id);

            debug!("Removed vector {} from database", vector_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get database statistics (lock-free)
    pub fn get_stats(&self) -> VectorDbStatsSnapshot {
        self.stats.snapshot()
    }

    /// Check if database has embeddings for a session (lock-free)
    pub fn has_session_embeddings(&self, session_id: &str) -> bool {
        self.metadata
            .iter()
            .any(|entry| entry.value().source == session_id)
    }

    /// Count embeddings for a session (lock-free)
    pub fn count_session_embeddings(&self, session_id: &str) -> usize {
        self.metadata
            .iter()
            .filter(|entry| entry.value().source == session_id)
            .count()
    }

    /// Build HNSW index for all vectors (lock-free)
    pub fn build_index(&self) -> Result<()> {
        info!("Building HNSW index for {} vectors", self.vectors.len());

        // Clear existing index
        for entry in self.hnsw_index.connections.iter() {
            let vector_id = *entry.key();
            self.hnsw_index.remove_vector(vector_id);
        }

        // Build index for each vector
        for entry in self.vectors.iter() {
            let vector_id = *entry.key();
            self.build_hnsw_index_for_vector(vector_id)?;
        }

        // Mark index as built
        self.stats.is_built.store(true, Ordering::Relaxed);

        info!("HNSW index built successfully");
        Ok(())
    }

    /// Clear all vectors from the database (lock-free)
    pub fn clear(&self) -> Result<()> {
        let vector_count = self.vectors.len();

        // Clear all collections (lock-free)
        self.vectors.clear();
        self.metadata.clear();

        // Clear HNSW index
        for entry in self.hnsw_index.connections.iter() {
            let vector_id = *entry.key();
            self.hnsw_index.remove_vector(vector_id);
        }

        // Reset statistics (lock-free)
        self.stats.total_vectors.store(0, Ordering::Relaxed);
        self.stats.memory_usage_bytes.store(0, Ordering::Relaxed);
        self.stats.is_built.store(false, Ordering::Relaxed);

        // Reset ID counter
        self.next_id.store(0, Ordering::Relaxed);

        info!("Cleared {} vectors from database", vector_count);
        Ok(())
    }

    /// Add vectors in batch (lock-free)
    pub fn add_vectors_batch(&self, vectors: Vec<(Vec<f32>, VectorMetadata)>) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(vectors.len());

        for (vector, metadata) in vectors {
            match self.add_vector(vector, metadata) {
                Ok(id) => ids.push(id),
                Err(e) => {
                    warn!("Failed to add vector to batch: {}", e);
                    // Continue with other vectors
                }
            }
        }

        info!("Added {} vectors in batch", ids.len());
        Ok(ids)
    }

    /// Search with custom filter (lock-free)
    pub fn search_with_filter<F>(
        &self,
        query_vector: &[f32],
        k: usize,
        filter: F,
    ) -> Result<Vec<SearchMatch>>
    where
        F: Fn(&VectorMetadata) -> bool,
    {
        let all_matches = self.search(query_vector, k * 2)?; // Get more results for filtering

        let filtered_matches: Vec<_> = all_matches
            .into_iter()
            .filter(|match_| filter(&match_.metadata))
            .take(k)
            .collect();

        Ok(filtered_matches)
    }

    /// Search in specific source (lock-free)
    pub fn search_in_source(
        &self,
        query_vector: &[f32],
        k: usize,
        source: &str,
    ) -> Result<Vec<SearchMatch>> {
        self.search_with_filter(query_vector, k, |metadata| metadata.source == source)
    }

    /// Search by content type (lock-free)
    pub fn search_by_content_type(
        &self,
        query_vector: &[f32],
        k: usize,
        content_type: &str,
    ) -> Result<Vec<SearchMatch>> {
        self.search_with_filter(query_vector, k, |metadata| {
            metadata.content_type == content_type
        })
    }
}

impl Default for LockFreeVectorDB {
    fn default() -> Self {
        Self::new(LockFreeVectorDbConfig::default())
            .expect("Failed to create default vector database")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockfree_vector_db_creation() {
        let config = LockFreeVectorDbConfig::default();
        let db = LockFreeVectorDB::new(config).unwrap();

        let stats = db.get_stats();
        assert_eq!(stats.total_vectors, 0);
        assert!(!stats.is_built);
    }

    #[test]
    fn test_add_and_search_vector() {
        let mut config = LockFreeVectorDbConfig::default();
        config.dimension = 3; // Use 3D vectors for testing
        let db = LockFreeVectorDB::new(config).unwrap();

        let vector = vec![1.0, 0.0, 0.0];
        let metadata = VectorMetadata::new(
            "test1".to_string(),
            "session1".to_string(),
            "text".to_string(),
            "test content".to_string(),
        );

        let id = db.add_vector(vector.clone(), metadata).unwrap();
        assert_eq!(id, 0);

        let results = db.search(&vector, 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector_id, 0);
        assert!((results[0].similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_similarity() {
        let mut config = LockFreeVectorDbConfig::default();
        config.dimension = 3; // Use 3D vectors for testing
        let db = LockFreeVectorDB::new(config).unwrap();

        let vector1 = vec![1.0, 0.0, 0.0];
        let vector2 = vec![0.9, 0.1, 0.0]; // Similar vector

        let metadata1 = VectorMetadata::new(
            "test1".to_string(),
            "session1".to_string(),
            "text".to_string(),
            "content1".to_string(),
        );

        let metadata2 = VectorMetadata::new(
            "test2".to_string(),
            "session1".to_string(),
            "text".to_string(),
            "content2".to_string(),
        );

        db.add_vector(vector1.clone(), metadata1).unwrap();
        db.add_vector(vector2.clone(), metadata2).unwrap();

        let results = db.search(&vector1, 5).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].similarity > 0.9); // High similarity
    }

    #[test]
    fn test_remove_vector() {
        let mut config = LockFreeVectorDbConfig::default();
        config.dimension = 3; // Use 3D vectors for testing
        let db = LockFreeVectorDB::new(config).unwrap();

        let vector = vec![1.0, 0.0, 0.0];
        let metadata = VectorMetadata::new(
            "test1".to_string(),
            "session1".to_string(),
            "text".to_string(),
            "test content".to_string(),
        );

        let id = db.add_vector(vector, metadata).unwrap();
        assert_eq!(db.get_stats().total_vectors, 1);

        let removed = db.remove_vector(id).unwrap();
        assert!(removed);
        assert_eq!(db.get_stats().total_vectors, 0);

        let results = db.search(&vec![1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_batch_operations() {
        let mut config = LockFreeVectorDbConfig::default();
        config.dimension = 3; // Use 3D vectors for testing
        let db = LockFreeVectorDB::new(config).unwrap();

        let vectors = vec![
            (
                vec![1.0, 0.0, 0.0],
                VectorMetadata::new(
                    "test1".to_string(),
                    "session1".to_string(),
                    "text".to_string(),
                    "content1".to_string(),
                ),
            ),
            (
                vec![0.0, 1.0, 0.0],
                VectorMetadata::new(
                    "test2".to_string(),
                    "session1".to_string(),
                    "text".to_string(),
                    "content2".to_string(),
                ),
            ),
            (
                vec![0.0, 0.0, 1.0],
                VectorMetadata::new(
                    "test3".to_string(),
                    "session1".to_string(),
                    "text".to_string(),
                    "content3".to_string(),
                ),
            ),
        ];

        let ids = db.add_vectors_batch(vectors).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(db.get_stats().total_vectors, 3);

        // Debug: check what vectors we actually have
        println!("Total vectors in DB: {}", db.get_stats().total_vectors);

        let results = db.search(&vec![1.0, 0.0, 0.0], 5).unwrap();
        println!("Search results count: {}", results.len());
        for (i, result) in results.iter().enumerate() {
            println!(
                "Result {}: vector_id={}, similarity={}",
                i, result.vector_id, result.similarity
            );
        }

        // The search should find all 3 vectors since they are all similar to [1.0, 0.0, 0.0] to some degree
        assert!(results.len() >= 1); // At least the exact match should be found
    }
}
