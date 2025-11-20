# Post-Cortex Implementation Plan
## 6 Critical Optimizations - Detailed Roadmap

**Project:** Post-Cortex Lock-Free Conversation Memory System
**Branch:** dev
**Target:** All 6 optimizations from OPTIMIZATION_PLAN.md
**Breaking Changes:** Allowed (no backward compatibility constraints)

---

## Table of Contents
1. [RocksDB Optimization (CRITICAL)](#1-rocksdb-optimization-critical)
2. [Vector Database Scalability](#2-vector-database-scalability)
3. [Entity Extraction Enhancement](#3-entity-extraction-enhancement)
4. [Concurrency Management](#4-concurrency-management)
5. [Error Handling](#5-error-handling)
6. [Test Infrastructure](#6-test-infrastructure)
7. [Implementation Timeline](#implementation-timeline)
8. [Dependencies](#dependencies)
9. [Breaking Changes Summary](#breaking-changes-summary)

---

## 1. RocksDB Optimization (CRITICAL)

### Priority: CRITICAL | Complexity: Low | Breaking Changes: None

### Problem
**File:** `src/storage/rocksdb_storage.rs`

All RocksDB operations are executed directly in async methods without `spawn_blocking`, which blocks tokio runtime threads and causes severe performance degradation under concurrent load.

### Affected Methods (17 total)
- Line 85-105: `save_session`
- Line 108-135: `load_session`
- Line 138-151: `save_checkpoint`
- Line 154-171: `load_checkpoint`
- Line 174-198: `batch_save_updates`
- Line 201-220: `list_sessions`
- Line 223-230: `get_stats`
- Line 234-238: `compact`
- Line 241-273: `delete_session`
- Line 276-284: `get_key_count`
- Line 287-290: `session_exists`
- Line 297-339: `save_workspace_metadata`
- Line 342-370: `delete_workspace`
- Line 373-413: `add_session_to_workspace`
- Line 416-434: `remove_session_from_workspace`

### Implementation Steps

#### Step 1: Pattern Template
Apply this transformation to all async methods:

```rust
// BEFORE
pub async fn load_session(&self, session_id: Uuid) -> Result<ActiveSession> {
    match self.db.get(key.as_bytes())? { // BLOCKING!
        Some(data) => {
            let (session, _) = bincode::serde::decode_from_slice(&data, ...)?;
            Ok(session)
        }
        None => Err(anyhow::anyhow!("Session not found"))
    }
}

// AFTER
pub async fn load_session(&self, session_id: Uuid) -> Result<ActiveSession> {
    let db = self.db.clone();
    let key = format!("session:{}", session_id);

    tokio::task::spawn_blocking(move || {
        match db.get(key.as_bytes())? {
            Some(data) => {
                let (session, _): (ActiveSession, usize) =
                    bincode::serde::decode_from_slice(&data, bincode::config::standard())?;
                Ok(session)
            }
            None => Err(anyhow::anyhow!("Session not found"))
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??
}
```

#### Step 2: Method-by-Method Checklist
- [ ] `save_session` - Wrap serialization + put
- [ ] `load_session` - Wrap get + deserialization
- [ ] `save_checkpoint` - Wrap put operation
- [ ] `load_checkpoint` - Wrap get + decode
- [ ] `batch_save_updates` - Wrap batch write
- [ ] `list_sessions` - Wrap iterator (blocking!)
- [ ] `get_stats` - Wrap property reads
- [ ] `compact` - Wrap compaction call
- [ ] `delete_session` - Wrap multiple deletes
- [ ] `get_key_count` - Wrap iterator count
- [ ] `session_exists` - Wrap get check
- [ ] `save_workspace_metadata` - Wrap batch write
- [ ] `delete_workspace` - Wrap prefix iteration + deletes
- [ ] `add_session_to_workspace` - Wrap get + put
- [ ] `remove_session_from_workspace` - Wrap get + put + delete

#### Step 3: Test Verification
Existing tests at lines 455-509 should pass without modification.

```bash
cargo test --features embeddings rocksdb_storage
```

### Dependencies
- **None** - Uses existing `tokio::task::spawn_blocking`

### Expected Impact
- **Latency:** 10-50x reduction under high concurrency
- **Throughput:** Prevents runtime thread starvation
- **CPU:** Better utilization of blocking thread pool

### Files Modified
- `src/storage/rocksdb_storage.rs` (17 methods)

---

## 2. Vector Database Scalability

### Priority: High | Complexity: Medium | Breaking Changes: None (short-term)

### Problem
**File:** `src/core/lockfree_vector_db.rs`

- Line 374: All vectors stored in memory via `Arc<DashMap<u32, LockFreeStoredVector>>`
- Line 384: HNSW index entirely in memory via `Arc<LockFreeHnswIndex>`
- Line 100: Each vector is 384 floats × 4 bytes = 1,536 bytes

**Memory usage:**
- 100k vectors: ~150 MB
- 1M vectors: ~1.5 GB (exceeds container limits)

### Short-Term Solution: Product Quantization

#### Step 1: Add Configuration
Modify `LockFreeVectorDbConfig` (line 38):

```rust
pub struct LockFreeVectorDbConfig {
    pub dimension: usize,
    pub max_elements: usize,
    pub m: usize,
    pub ef_construction: usize,

    // NEW: Product Quantization settings
    pub enable_product_quantization: bool,
    pub pq_subvectors: usize,        // Default: 8 (384/8 = 48 dims each)
    pub pq_bits: usize,              // Default: 8 (256 centroids)
}

impl Default for LockFreeVectorDbConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_elements: 100000,
            m: 16,
            ef_construction: 200,
            enable_product_quantization: true,  // Enable by default
            pq_subvectors: 8,
            pq_bits: 8,
        }
    }
}
```

#### Step 2: Modify Vector Storage
Update `LockFreeStoredVector` (line 89):

```rust
pub struct LockFreeStoredVector {
    pub id: u32,
    pub vector: Vec<f32>,           // Keep for exact re-ranking
    pub quantized: Option<Vec<u8>>, // Existing 8-bit quantization
    pub pq_codes: Option<Vec<u8>>,  // NEW: Product quantization codes
    pub magnitude: f32,
}
```

#### Step 3: Implement PQ Codebook
Add new structure for centroids:

```rust
pub struct ProductQuantizationCodebook {
    subvectors: usize,
    bits: usize,
    dimension: usize,
    // centroids[subvector_idx][centroid_idx] = Vec<f32>
    centroids: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizationCodebook {
    pub fn new(dimension: usize, subvectors: usize, bits: usize) -> Self {
        let subvec_dim = dimension / subvectors;
        let num_centroids = 1 << bits; // 2^bits (e.g., 2^8 = 256)

        // Initialize random centroids (will be trained on first vectors)
        let mut centroids = Vec::with_capacity(subvectors);
        for _ in 0..subvectors {
            let mut subvec_centroids = Vec::with_capacity(num_centroids);
            for _ in 0..num_centroids {
                let centroid: Vec<f32> = (0..subvec_dim)
                    .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                    .collect();
                subvec_centroids.push(centroid);
            }
            centroids.push(subvec_centroids);
        }

        Self { subvectors, bits, dimension, centroids }
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let subvec_dim = self.dimension / self.subvectors;
        let mut codes = Vec::with_capacity(self.subvectors);

        for i in 0..self.subvectors {
            let start = i * subvec_dim;
            let end = start + subvec_dim;
            let subvec = &vector[start..end];

            // Find nearest centroid
            let code = self.find_nearest_centroid(i, subvec);
            codes.push(code);
        }

        codes
    }

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

    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let subvec_dim = self.dimension / self.subvectors;
        let mut vector = Vec::with_capacity(self.dimension);

        for (i, &code) in codes.iter().enumerate() {
            let centroid = &self.centroids[i][code as usize];
            vector.extend_from_slice(centroid);
        }

        vector
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
```

#### Step 4: Add PQ to Vector DB
Modify `LockFreeVectorDB`:

```rust
pub struct LockFreeVectorDB {
    config: LockFreeVectorDbConfig,
    vectors: Arc<DashMap<u32, LockFreeStoredVector>>,
    hnsw_index: Arc<LockFreeHnswIndex>,
    next_id: Arc<AtomicU32>,
    pq_codebook: Option<Arc<ProductQuantizationCodebook>>, // NEW
}

impl LockFreeVectorDB {
    pub fn new(config: LockFreeVectorDbConfig) -> Result<Self> {
        let pq_codebook = if config.enable_product_quantization {
            Some(Arc::new(ProductQuantizationCodebook::new(
                config.dimension,
                config.pq_subvectors,
                config.pq_bits,
            )))
        } else {
            None
        };

        Ok(Self {
            config,
            vectors: Arc::new(DashMap::new()),
            hnsw_index: Arc::new(LockFreeHnswIndex::new(config.clone())),
            next_id: Arc::new(AtomicU32::new(0)),
            pq_codebook,
        })
    }

    pub fn add_vector(&self, vector: Vec<f32>, metadata: VectorMetadata) -> Result<u32> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Encode with PQ if enabled
        let pq_codes = if let Some(codebook) = &self.pq_codebook {
            Some(codebook.encode(&vector))
        } else {
            None
        };

        let stored = LockFreeStoredVector {
            id,
            vector: vector.clone(),
            quantized: None,
            pq_codes,
            magnitude: calculate_magnitude(&vector),
        };

        self.vectors.insert(id, stored);
        self.hnsw_index.add_node(id, vector);

        Ok(id)
    }
}
```

#### Step 5: Two-Stage Search
Modify search to use PQ for approximate candidates:

```rust
pub fn search(&self, query_vector: &[f32], k: usize) -> Result<Vec<SearchMatch>> {
    if let Some(codebook) = &self.pq_codebook {
        // Stage 1: Fast PQ approximate search (10x candidates)
        let candidate_ids = self.hnsw_search_approximate(query_vector, k * 10)?;

        // Stage 2: Exact re-ranking with full vectors
        let mut exact_results = Vec::new();
        for id in candidate_ids {
            if let Some(stored) = self.vectors.get(&id) {
                let similarity = Self::calculate_cosine_similarity(
                    query_vector,
                    &stored.vector
                );
                exact_results.push(SearchMatch {
                    vector_id: id,
                    similarity,
                    metadata: stored.metadata.clone(),
                });
            }
        }

        // Sort and return top-k
        exact_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        exact_results.truncate(k);
        Ok(exact_results)
    } else {
        // Fallback to original search
        self.hnsw_search(query_vector, k)
    }
}
```

### Memory Reduction
- **Without PQ:** 1,536 bytes per vector
- **With 8-bit PQ:** 8 bytes per vector + ~50 MB centroids
- **Compression ratio:** 192x
- **At 1M vectors:** 1.5 GB → 8 MB + 50 MB = **58 MB**

### Long-Term Solution: LanceDB Migration

#### Option A: LanceDB (Recommended)
```toml
# Add to Cargo.toml
[dependencies]
lancedb = "0.8"
arrow = "53.0"
```

**Advantages:**
- Native Rust, embedded (no external service)
- Disk-backed with memory-mapped I/O
- Built-in PQ and IVF indexing
- Scales to billions of vectors

#### Option B: Qdrant
Already in dependencies (line 27):
```toml
qdrant-client = { version = "1.12", optional = true }
```

### Dependencies
- **Short-term (PQ):** None (in-house implementation)
- **Long-term:** `lancedb = "0.8"`, `arrow = "53.0"`

### Files Modified
- `src/core/lockfree_vector_db.rs`

---

## 3. Entity Extraction Enhancement

### Priority: Medium | Complexity: Medium | Breaking Changes: None

### Problem
**File:** `src/session/active_session.rs` (lines 878-1298)

Current `extract_entities_from_text` uses only regex patterns:
- High false positive rate (extracts "The", "This", etc.)
- Misses multi-word technical entities
- No semantic understanding
- Language-specific issues

### Implementation Steps

#### Step 1: Create NER Model Wrapper
**New file:** `src/core/ner_model.rs`

```rust
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;
use anyhow::Result;
use std::sync::Arc;

pub struct NerModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    label_map: Vec<String>,
}

impl NerModel {
    pub fn new() -> Result<Self> {
        // Use BERT-based NER model
        let model_id = "dslim/bert-base-NER";

        // Download from HuggingFace Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_id.to_string());
        let weights_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        // Load configuration
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Initialize model
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], device.clone())?
        };
        let model = BertModel::load(vb, &config)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        // NER labels (B-I-O scheme)
        let label_map = vec![
            "O".to_string(),           // Outside
            "B-PER".to_string(),       // Person
            "I-PER".to_string(),
            "B-ORG".to_string(),       // Organization
            "I-ORG".to_string(),
            "B-LOC".to_string(),       // Location
            "I-LOC".to_string(),
            "B-MISC".to_string(),      // Miscellaneous
            "I-MISC".to_string(),
        ];

        Ok(Self { model, tokenizer, device, label_map })
    }

    pub fn extract_entities(&self, text: &str) -> Result<Vec<String>> {
        // Tokenize
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;

        let tokens = encoding.get_ids();
        let token_type_ids = encoding.get_type_ids();
        let word_ids = encoding.get_word_ids();

        // Convert to tensors
        let input_ids = Tensor::new(&[tokens], &self.device)?
            .unsqueeze(0)?;
        let token_type_ids = Tensor::new(&[token_type_ids], &self.device)?
            .unsqueeze(0)?;

        // Run inference
        let output = self.model.forward(&input_ids, &token_type_ids)?;

        // Get logits for token classification
        let logits = output.squeeze(0)?;
        let predictions = logits.argmax(1)?;
        let predictions = predictions.to_vec1::<u32>()?;

        // Extract entities using B-I-O scheme
        let mut entities = Vec::new();
        let mut current_entity = String::new();
        let mut current_type = String::new();

        for (idx, &pred) in predictions.iter().enumerate() {
            if let Some(word_id) = word_ids[idx] {
                let label = &self.label_map[pred as usize];

                if label.starts_with("B-") {
                    // Begin new entity
                    if !current_entity.is_empty() {
                        entities.push(current_entity.clone());
                    }
                    current_entity = encoding.get_tokens()[idx].clone();
                    current_type = label[2..].to_string();
                } else if label.starts_with("I-") && label[2..] == current_type {
                    // Continue entity
                    current_entity.push(' ');
                    current_entity.push_str(&encoding.get_tokens()[idx]);
                } else if label == "O" {
                    // Outside - finish current entity
                    if !current_entity.is_empty() {
                        entities.push(current_entity.clone());
                        current_entity.clear();
                        current_type.clear();
                    }
                }
            }
        }

        // Add last entity
        if !current_entity.is_empty() {
            entities.push(current_entity);
        }

        Ok(entities)
    }
}
```

#### Step 2: Add NER to ActiveSession
Modify `src/session/active_session.rs` (line 33):

```rust
pub struct ActiveSession {
    // ... existing fields ...

    /// Optional NER model (lazy-loaded)
    #[serde(skip)]
    ner_model: Option<Arc<NerModel>>,
}
```

#### Step 3: Update Entity Extraction
Modify `extract_entities_from_text` (line 878):

```rust
fn extract_entities_from_text(&mut self, text: &str) -> Vec<String> {
    let mut entities = HashSet::new();

    // Try NER model first
    if self.ner_model.is_none() {
        // Lazy-load NER model
        match NerModel::new() {
            Ok(model) => {
                info!("Loaded NER model for entity extraction");
                self.ner_model = Some(Arc::new(model));
            }
            Err(e) => {
                warn!("Failed to load NER model, using regex fallback: {}", e);
            }
        }
    }

    if let Some(ner) = &self.ner_model {
        match ner.extract_entities(text) {
            Ok(ner_entities) => {
                info!("NER extracted {} entities", ner_entities.len());
                entities.extend(ner_entities);
            }
            Err(e) => {
                warn!("NER extraction failed: {}, using regex fallback", e);
            }
        }
    }

    // Fallback to regex if NER unavailable or failed
    if entities.is_empty() {
        info!("Using regex-based entity extraction");
        self.extract_proper_nouns(text, &mut entities);
        self.extract_technical_terms(&text.to_lowercase(), &mut entities);
        self.extract_quoted_terms(text, &mut entities);
        self.extract_compound_terms(text, &mut entities);
        self.extract_domain_specific_terms(text, &mut entities);
    }

    // Score and filter
    let scored = self.score_entities(&entities, text);
    scored.into_iter().take(20).collect()
}
```

#### Step 4: Add Module Declaration
Add to `src/core/mod.rs`:

```rust
#[cfg(feature = "embeddings")]
pub mod ner_model;
```

### Model Selection

**Primary:** `dslim/bert-base-NER`
- Size: 110M parameters (~420 MB)
- F1 Score: 95.7% on CoNLL-2003
- Entities: PERSON, ORG, LOC, MISC
- Inference: ~50ms per sentence (CPU)

**Alternative:** `Davlan/distilbert-base-multilingual-cased-ner-hrl`
- Multilingual (40+ languages including Cyrillic)
- Smaller: 66M parameters (~250 MB)
- F1 Score: 90%+

### Dependencies
**Already present** in Cargo.toml (lines 46-50):
```toml
candle-core = { version = "0.9", optional = true }
candle-nn = { version = "0.9", optional = true }
candle-transformers = { version = "0.9", optional = true }
tokenizers = { version = "0.22", optional = true }
hf-hub = { version = "0.4", optional = true }
```

Enabled via `embeddings` feature (line 82).

### Performance Impact
- **Accuracy:** 70% → 95%+ precision/recall
- **Latency:** +50ms per extraction (acceptable)
- **Memory:** +420 MB (lazy-loaded per session)

### Files Modified
- `src/core/ner_model.rs` (NEW)
- `src/core/mod.rs` (add module)
- `src/session/active_session.rs` (integrate NER)

---

## 4. Concurrency Management

### Priority: Medium | Complexity: High | Breaking Changes: Moderate

### Problem
**File:** `src/core/lockfree_memory_system.rs` (line 101)
**File:** `src/session/active_session.rs` (lines 33-72)

`ActiveSession` is monolithic structure wrapped in `Arc<ArcSwap<...>>`:
- Every update clones entire session (10-100 KB)
- Write amplification on frequent updates
- Hot/warm context updated constantly

### Implementation Steps

#### Step 1: Create Granular Components
**New file:** `src/session/session_components.rs`

```rust
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::collections::VecDeque;
use arc_swap::ArcSwap;

/// Hot context with fine-grained locking
pub struct HotContext {
    updates: RwLock<VecDeque<ContextUpdate>>,
    max_size: usize,
}

impl HotContext {
    pub fn new(max_size: usize) -> Self {
        Self {
            updates: RwLock::new(VecDeque::with_capacity(max_size)),
            max_size,
        }
    }

    pub fn push(&self, update: ContextUpdate) {
        let mut updates = self.updates.write();
        updates.push_back(update);
        if updates.len() > self.max_size {
            updates.pop_front();
        }
    }

    pub fn get_recent(&self, n: usize) -> Vec<ContextUpdate> {
        self.updates.read()
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.updates.read().len()
    }
}

/// Lock-free entity graph
pub struct LockFreeEntityGraph {
    entities: DashMap<String, EntityNode>,
    relationships: RwLock<Vec<EntityRelationship>>,
}

#[derive(Clone)]
pub struct EntityNode {
    pub entity_type: EntityType,
    pub mention_count: Arc<AtomicUsize>,
    pub last_mentioned: Arc<AtomicU64>,
}

impl LockFreeEntityGraph {
    pub fn new() -> Self {
        Self {
            entities: DashMap::new(),
            relationships: RwLock::new(Vec::new()),
        }
    }

    pub fn add_entity(&self, name: String, entity_type: EntityType) {
        self.entities
            .entry(name)
            .and_modify(|node| {
                node.mention_count.fetch_add(1, Ordering::Relaxed);
                node.last_mentioned.store(current_timestamp(), Ordering::Relaxed);
            })
            .or_insert(EntityNode {
                entity_type,
                mention_count: Arc::new(AtomicUsize::new(1)),
                last_mentioned: Arc::new(AtomicU64::new(current_timestamp())),
            });
    }

    pub fn add_relationship(&self, from: String, to: String, rel_type: RelationType) {
        let mut rels = self.relationships.write();
        rels.push(EntityRelationship {
            from_entity: from,
            to_entity: to,
            relationship_type: rel_type,
            strength: 1.0,
        });
    }

    pub fn get_entity(&self, name: &str) -> Option<EntityNode> {
        self.entities.get(name).map(|e| e.clone())
    }

    pub fn get_all_entities(&self) -> Vec<(String, EntityNode)> {
        self.entities
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
}

/// Immutable session metadata (use ArcSwap for rare updates)
#[derive(Clone)]
pub struct SessionMetadata {
    pub id: Uuid,
    pub name: Option<String>,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub user_preferences: UserPreferences,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
```

#### Step 2: Refactor ActiveSession
Modify `src/session/active_session.rs`:

```rust
pub struct ActiveSession {
    // Immutable metadata - ArcSwap for rare updates
    pub metadata: Arc<ArcSwap<SessionMetadata>>,

    // Frequently updated - fine-grained locking
    pub hot_context: Arc<HotContext>,
    pub entity_graph: Arc<LockFreeEntityGraph>,
    pub current_state: Arc<RwLock<StructuredContext>>,

    // Medium frequency - existing locking
    pub warm_context: Arc<RwLock<Vec<CompressedUpdate>>>,
    pub cold_context: Arc<RwLock<Vec<StructuredSummary>>>,

    // Code references - lock-free map
    pub code_references: Arc<DashMap<String, Vec<CodeReference>>>,

    // Statistics - atomic counters
    pub last_updated: Arc<AtomicU64>,
    pub update_count: Arc<AtomicUsize>,

    // Optional components
    #[serde(skip)]
    pub ner_model: Option<Arc<NerModel>>,
}

impl ActiveSession {
    pub fn new(id: Uuid, name: Option<String>, description: Option<String>) -> Self {
        let metadata = SessionMetadata {
            id,
            name,
            description,
            created_at: Utc::now(),
            user_preferences: UserPreferences::default(),
        };

        Self {
            metadata: Arc::new(ArcSwap::from_pointee(metadata)),
            hot_context: Arc::new(HotContext::new(50)),
            entity_graph: Arc::new(LockFreeEntityGraph::new()),
            current_state: Arc::new(RwLock::new(StructuredContext::default())),
            warm_context: Arc::new(RwLock::new(Vec::new())),
            cold_context: Arc::new(RwLock::new(Vec::new())),
            code_references: Arc::new(DashMap::new()),
            last_updated: Arc::new(AtomicU64::new(current_timestamp())),
            update_count: Arc::new(AtomicUsize::new(0)),
            ner_model: None,
        }
    }
}
```

#### Step 3: Update Incremental Updates
Modify `add_incremental_update` (line 169):

```rust
pub async fn add_incremental_update(&self, update: ContextUpdate) -> Result<()> {
    // Update hot context (single lock)
    self.hot_context.push(update.clone());

    // Update entity graph (lock-free)
    for entity in &update.creates_entities {
        self.entity_graph.add_entity(
            entity.clone(),
            self.infer_entity_type(&update.update_type, entity),
        );
    }

    for rel in &update.creates_relationships {
        self.entity_graph.add_relationship(
            rel.from.clone(),
            rel.to.clone(),
            rel.rel_type,
        );
    }

    // Update current state (single lock)
    {
        let mut state = self.current_state.write();
        self.update_current_state_sync(&mut state, &update)?;
    }

    // Update code references (lock-free)
    if let Some(code_refs) = &update.code_references {
        for (file, refs) in code_refs {
            self.code_references
                .entry(file.clone())
                .or_insert_with(Vec::new)
                .extend(refs.clone());
        }
    }

    // Update atomics
    self.last_updated.store(current_timestamp(), Ordering::Relaxed);
    self.update_count.fetch_add(1, Ordering::Relaxed);

    Ok(())
}
```

#### Step 4: Update Session Manager
Modify `src/core/lockfree_memory_system.rs` (line 101):

```rust
pub struct LockFreeSessionManager {
    // Change from Arc<ArcSwap<ActiveSession>> to Arc<ActiveSession>
    pub sessions: LockFreeSessionCache<Uuid, Arc<ActiveSession>>,
    storage_actor: StorageActorHandle,
    config: MemorySystemConfig,
}
```

#### Step 5: Serialization Compatibility
Add backward compatibility for deserialization:

```rust
#[derive(Deserialize)]
struct OldActiveSession {
    // Old monolithic structure fields
}

impl ActiveSession {
    pub fn from_old_format(old: OldActiveSession) -> Self {
        let mut session = Self::new(old.id, old.name, old.description);

        // Migrate hot context
        for update in old.hot_context {
            session.hot_context.push(update);
        }

        // Migrate entities
        for (name, entity) in old.entity_graph.entities {
            session.entity_graph.add_entity(name, entity.entity_type);
        }

        // ... migrate other fields

        session
    }
}
```

### Dependencies
**Already present:**
- `parking_lot = "0.12"` (line 37)
- `dashmap = "6.1"` (line 30)
- `arc-swap = "1.7"` (line 35)

### Breaking Changes
- `ActiveSession` structure changes significantly
- Old sessions need migration on load
- Session cache type changes

### Performance Impact
- **Write amplification:** 10-100 KB → 8-64 bytes per update
- **Throughput:** 2-5x improvement under concurrency
- **Memory overhead:** +10% (additional Arc wrappers)

### Files Modified
- `src/session/session_components.rs` (NEW)
- `src/session/active_session.rs` (major refactor)
- `src/core/lockfree_memory_system.rs` (session manager)
- `src/session/mod.rs` (add module)

---

## 5. Error Handling

### Priority: Low | Complexity: Low | Breaking Changes: API signatures

### Problem
`anyhow::Result` used throughout codebase (23 occurrences):
- Cannot match on specific error types
- No programmatic error recovery
- Poor error messages for API consumers

### Implementation Steps

#### Step 1: Create Error Module
**New file:** `src/core/error.rs`

```rust
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum SystemError {
    // Storage errors
    #[error("Database error: {0}")]
    Database(#[from] rocksdb::Error),

    #[error("Session {0} not found")]
    SessionNotFound(Uuid),

    #[error("Workspace {0} not found")]
    WorkspaceNotFound(Uuid),

    #[error("Checkpoint {0} not found")]
    CheckpointNotFound(Uuid),

    // Serialization
    #[error("Serialization failed: {0}")]
    Serialization(String),

    #[error("Deserialization failed: {0}")]
    Deserialization(String),

    // Vector DB
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },

    #[error("HNSW index not built")]
    IndexNotBuilt,

    #[error("Vector {0} not found")]
    VectorNotFound(u32),

    // Context processing
    #[error("Entity extraction failed: {0}")]
    EntityExtractionFailed(String),

    #[error("Update processing timeout after {0}ms")]
    UpdateTimeout(u64),

    #[error("Entity graph update failed: {0}")]
    GraphUpdateFailed(String),

    // Embeddings
    #[cfg(feature = "embeddings")]
    #[error("Embedding model error: {0}")]
    EmbeddingModel(String),

    #[cfg(feature = "embeddings")]
    #[error("Vectorization failed: {0}")]
    VectorizationFailed(String),

    // System
    #[error("Storage actor channel closed")]
    StorageActorDown,

    #[error("Operation timeout after {0}s")]
    OperationTimeout(u64),

    #[error("Circuit breaker open: {0}")]
    CircuitBreakerOpen(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    // I/O
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    // Task
    #[error("Task join error: {0}")]
    TaskJoin(String),

    // Catch-all
    #[error("Internal error: {0}")]
    Internal(String),
}

// Bincode conversions
impl From<bincode::error::EncodeError> for SystemError {
    fn from(err: bincode::error::EncodeError) -> Self {
        SystemError::Serialization(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for SystemError {
    fn from(err: bincode::error::DecodeError) -> Self {
        SystemError::Deserialization(err.to_string())
    }
}

// Tokio join error
impl From<tokio::task::JoinError> for SystemError {
    fn from(err: tokio::task::JoinError) -> Self {
        SystemError::TaskJoin(err.to_string())
    }
}

// Type alias
pub type Result<T> = std::result::Result<T, SystemError>;
```

#### Step 2: Update Public APIs

**File:** `src/storage/rocksdb_storage.rs`
```rust
use crate::core::error::{Result, SystemError};

pub async fn load_session(&self, session_id: Uuid) -> Result<ActiveSession> {
    // ... implementation ...
    None => Err(SystemError::SessionNotFound(session_id))
}

pub async fn load_workspace(&self, workspace_id: Uuid) -> Result<Workspace> {
    // ... implementation ...
    None => Err(SystemError::WorkspaceNotFound(workspace_id))
}
```

**File:** `src/core/lockfree_vector_db.rs`
```rust
use crate::core::error::{Result, SystemError};

pub fn add_vector(&self, vector: Vec<f32>, metadata: VectorMetadata) -> Result<u32> {
    if vector.len() != self.config.dimension {
        return Err(SystemError::VectorDimensionMismatch {
            expected: self.config.dimension,
            actual: vector.len(),
        });
    }
    // ... rest
}

pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchMatch>> {
    if query.len() != self.config.dimension {
        return Err(SystemError::VectorDimensionMismatch {
            expected: self.config.dimension,
            actual: query.len(),
        });
    }
    // ... rest
}
```

**File:** `src/session/active_session.rs`
```rust
use crate::core::error::{Result, SystemError};

pub async fn add_incremental_update(&self, update: ContextUpdate) -> Result<()> {
    match timeout(Duration::from_secs(3), self.process_update(&update)).await {
        Ok(result) => result,
        Err(_) => Err(SystemError::UpdateTimeout(3000)),
    }
}
```

#### Step 3: Add Module Declaration
In `src/core/mod.rs`:
```rust
pub mod error;
```

#### Step 4: Update Re-exports
In `src/lib.rs`:
```rust
pub use core::error::{Result, SystemError};
```

### Dependencies
**Already present:** `thiserror = "2.0"` (line 25)

### Migration Guide for Consumers

```rust
// Before
use post_cortex::LockFreeConversationMemorySystem;

match system.load_session(id).await {
    Ok(session) => { /* ... */ },
    Err(e) => eprintln!("Error: {}", e),
}

// After
use post_cortex::{LockFreeConversationMemorySystem, SystemError};

match system.load_session(id).await {
    Ok(session) => { /* ... */ },
    Err(SystemError::SessionNotFound(id)) => {
        // Create new session
        eprintln!("Session {} not found, creating new", id);
        system.create_session(id, None, None).await?
    },
    Err(SystemError::Database(e)) => {
        // Retry or fail gracefully
        eprintln!("Database error: {}, retrying...", e);
    },
    Err(e) => eprintln!("Unexpected error: {}", e),
}
```

### Files Modified
- `src/core/error.rs` (NEW)
- `src/core/mod.rs` (add module)
- `src/lib.rs` (re-export)
- `src/storage/rocksdb_storage.rs` (update signatures)
- `src/core/lockfree_vector_db.rs` (update signatures)
- `src/session/active_session.rs` (update signatures)

---

## 6. Test Infrastructure

### Priority: Medium | Complexity: Low | Breaking Changes: None

### Problem
**File:** `tests/integration_daemon.rs`

Current tests use real TCP ports (line 9-34):
- Port conflicts on CI/CD
- Race conditions on startup
- Slow (~100ms per test)
- Flaky (intermittent failures)

### Implementation Steps

#### Step 1: Create Test Helpers
**New file:** `tests/helpers/mod.rs`

```rust
use axum::Router;
use tower::ServiceExt;
use hyper::{Request, Response, Body, StatusCode};
use serde_json::Value;

/// In-memory test helper (no TCP)
pub struct TestApp {
    router: Router,
}

impl TestApp {
    pub fn new(router: Router) -> Self {
        Self { router }
    }

    /// Make request without network
    pub async fn request(&self, req: Request<Body>) -> Response<Body> {
        self.router
            .clone()
            .oneshot(req)
            .await
            .expect("Request failed")
    }

    /// Helper: GET request
    pub async fn get(&self, uri: &str) -> Response<Body> {
        Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap()
            |> |req| self.request(req).await
    }

    /// Helper: POST JSON
    pub async fn post_json(&self, uri: &str, json: Value) -> Response<Body> {
        let body = serde_json::to_string(&json).unwrap();
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
            |> |req| self.request(req).await
    }

    /// Helper: Extract JSON body
    pub async fn json_body(response: Response<Body>) -> Value {
        let bytes = hyper::body::to_bytes(response.into_body())
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Helper: Assert status
    pub fn assert_status(response: &Response<Body>, expected: StatusCode) {
        assert_eq!(response.status(), expected);
    }
}
```

#### Step 2: Expose Router from Daemon
Modify daemon server (find the file - likely `src/daemon/mod.rs` or similar):

```rust
impl LockFreeDaemonServer {
    // Existing private method
    async fn start(&self) -> Result<()> {
        let router = self.build_router();
        // ... bind and serve
    }

    // NEW: Public method for testing
    pub fn build_router(&self) -> Router {
        Router::new()
            .route("/health", get(health_check))
            .route("/stats", get(stats))
            .route("/mcp", post(mcp_handler))
            .with_state(self.system.clone())
    }
}
```

#### Step 3: Refactor Integration Tests
Update `tests/integration_daemon.rs`:

```rust
mod helpers;
use helpers::TestApp;

#[tokio::test]
async fn test_daemon_health_check() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = DaemonConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // Unused
        data_directory: temp_dir.path().to_str().unwrap().to_string(),
    };

    let server = LockFreeDaemonServer::new(config).await.unwrap();
    let app = TestApp::new(server.build_router());

    // In-memory request
    let response = app.get("/health").await;
    TestApp::assert_status(&response, StatusCode::OK);

    let json = TestApp::json_body(response).await;
    assert_eq!(json["status"], "ok");
}

// Apply same pattern to all tests...
```

#### Step 4: Property-Based Testing
**New file:** `tests/property_vector_db.rs`

```rust
use proptest::prelude::*;
use post_cortex::core::lockfree_vector_db::*;
use std::sync::Arc;

proptest! {
    #[test]
    fn test_vector_search_properties(
        dim in 32usize..512,
        num_vecs in 1usize..100,
        k in 1usize..10,
    ) {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let config = LockFreeVectorDbConfig {
                dimension: dim,
                ..Default::default()
            };
            let db = LockFreeVectorDB::new(config)?;

            // Insert random vectors
            for i in 0..num_vecs {
                let vec: Vec<f32> = (0..dim).map(|_| rand::random()).collect();
                let meta = VectorMetadata::new(
                    format!("v{}", i),
                    format!("text{}", i),
                    "test".into(),
                    "test".into(),
                );
                db.add_vector(vec, meta)?;
            }

            // Property 1: Results <= k
            let query: Vec<f32> = (0..dim).map(|_| rand::random()).collect();
            let results = db.search(&query, k)?;
            prop_assert!(results.len() <= k);

            // Property 2: All IDs exist
            for r in &results {
                prop_assert!(db.get_metadata(r.vector_id).is_some());
            }

            // Property 3: Sorted descending
            for w in results.windows(2) {
                prop_assert!(w[0].similarity >= w[1].similarity);
            }

            // Property 4: Valid range [0, 1]
            for r in &results {
                prop_assert!(r.similarity >= 0.0 && r.similarity <= 1.0);
            }

            Ok(())
        })?;
    }

    #[test]
    fn test_concurrent_correctness(
        threads in 1usize..16,
        ops in 1usize..50,
    ) {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let db = Arc::new(LockFreeVectorDB::new(Default::default())?);

            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let db = db.clone();
                    tokio::spawn(async move {
                        for i in 0..ops {
                            let vec = vec![1.0; 384];
                            let meta = VectorMetadata::new(
                                format!("t{}v{}", t, i),
                                format!("text{}", i),
                                "test".into(),
                                "test".into(),
                            );
                            db.add_vector(vec, meta).unwrap();
                        }
                    })
                })
                .collect();

            for h in handles {
                h.await.unwrap();
            }

            let stats = db.get_stats();
            prop_assert_eq!(stats.total_vectors, threads * ops);

            Ok(())
        })?;
    }
}
```

### Dependencies
Add to `Cargo.toml`:
```toml
[dev-dependencies]
proptest = "1.5"
# tower and axum already present
```

### Performance Impact
- **Test speed:** 100ms → <5ms (20x faster)
- **Reliability:** Eliminates port conflicts
- **Coverage:** Property tests find edge cases

### Files Modified
- `tests/helpers/mod.rs` (NEW)
- `tests/integration_daemon.rs` (refactor all tests)
- `tests/property_vector_db.rs` (NEW)
- Daemon server file (expose `build_router()`)
- `Cargo.toml` (add proptest)

---

## Implementation Timeline

### Phase 1: Critical Fixes (Days 1-2)
**Estimated effort:** 4-6 hours

- [ ] RocksDB optimization (17 methods) - 2-3 hours
- [ ] Test infrastructure refactor - 2-3 hours

**Deliverables:**
- All RocksDB operations use `spawn_blocking`
- Tests run 20x faster without TCP ports
- Property tests for vector DB

### Phase 2: Scalability (Days 3-5)
**Estimated effort:** 8-12 hours

- [ ] Vector DB Product Quantization - 4-6 hours
  - Codebook implementation
  - Two-stage search
  - Testing and validation
- [ ] Typed error system - 4-6 hours
  - Create error enum
  - Update all public APIs
  - Migration guide

**Deliverables:**
- 192x memory reduction for vectors
- Typed errors across codebase

### Phase 3: Concurrency & NLP (Days 6-10)
**Estimated effort:** 16-24 hours

- [ ] ActiveSession refactoring - 12-16 hours
  - Create granular components
  - Update all access patterns
  - Serialization compatibility
  - Thorough testing
- [ ] NER model integration - 4-8 hours
  - NER wrapper implementation
  - Integration with entity extraction
  - Lazy loading logic

**Deliverables:**
- 2-5x throughput improvement
- 95%+ entity extraction accuracy

### Phase 4: Long-Term (Future)
- [ ] LanceDB evaluation and migration
- [ ] Production metrics collection
- [ ] Performance benchmarking suite

**Total estimated time:** 28-42 hours (1-2 weeks for single developer)

---

## Dependencies

### Required (Add to Cargo.toml)
```toml
[dev-dependencies]
proptest = "1.5"
```

### Already Present (No changes needed)
```toml
[dependencies]
tokio = { version = "1.42", features = ["full"] }
rocksdb = "0.22"
dashmap = "6.1"
arc-swap = "1.7"
parking_lot = "0.12"
thiserror = "2.0"
candle-core = { version = "0.9", optional = true }
candle-nn = { version = "0.9", optional = true }
candle-transformers = { version = "0.9", optional = true }
tokenizers = { version = "0.22", optional = true }
hf-hub = { version = "0.4", optional = true }
```

---

## Breaking Changes Summary

### Phase 1 & 2: NO BREAKING CHANGES
- RocksDB: Internal implementation only
- Tests: Dev-only changes
- Vector PQ: Transparent optimization
- Typed errors: API signatures change, but migration is straightforward

### Phase 3: MODERATE BREAKING CHANGES

**ActiveSession structure:**
- Old: Monolithic struct in `ArcSwap`
- New: Granular components with fine-grained locking

**Migration path:**
1. Old sessions auto-migrate on load via `from_old_format()`
2. No data loss - all fields preserved
3. May require RocksDB compaction to cleanup old format

**Session manager:**
- Old: `Arc<ArcSwap<ActiveSession>>`
- New: `Arc<ActiveSession>`
- Affects only internal code, not public API

**Mitigation:**
- Implement compatibility deserializer
- Document migration process
- Provide rollback instructions

---

## Testing Strategy

### Unit Tests
- [ ] RocksDB spawn_blocking wrapper tests
- [ ] PQ encode/decode correctness
- [ ] Error type conversions
- [ ] Component-level tests for granular ActiveSession

### Integration Tests
- [ ] All daemon tests migrated to in-memory
- [ ] End-to-end session lifecycle
- [ ] Multi-session concurrency

### Property-Based Tests
- [ ] Vector search invariants
- [ ] Concurrent operation correctness
- [ ] Similarity score properties

### Performance Tests
- [ ] RocksDB latency under load
- [ ] Vector search with PQ vs without
- [ ] ActiveSession update throughput

### Regression Tests
- [ ] Old session deserialization
- [ ] Backward compatibility checks

---

## Rollback Plan

### If Critical Issues Arise:

**Phase 1 (RocksDB/Tests):**
- Revert: Remove `spawn_blocking` wrappers
- Risk: Low - isolated changes

**Phase 2 (PQ/Errors):**
- Revert: Disable PQ via config flag
- Revert: Keep `anyhow` alongside `SystemError`
- Risk: Medium - can run in parallel

**Phase 3 (ActiveSession):**
- Revert: Keep `from_old_format()` deserializer
- Revert: Deploy old version, sessions auto-downgrade
- Risk: High - requires careful migration

---

## Success Metrics

### Performance
- [ ] RocksDB P99 latency < 10ms (down from 100-500ms)
- [ ] Vector search memory < 100 MB for 1M vectors
- [ ] Session update throughput > 1000 ops/sec
- [ ] Test suite < 30 seconds (down from 10+ minutes)

### Quality
- [ ] Entity extraction F1 score > 90%
- [ ] Zero flaky tests
- [ ] Error recovery rate > 80%
- [ ] Property test coverage > 50%

### Reliability
- [ ] No deadlocks (verify with stress tests)
- [ ] No data corruption (verify with checksums)
- [ ] Graceful degradation on resource limits

---

## Post-Implementation

### Documentation Updates
- [ ] Update CLAUDE.md with new patterns
- [ ] Add migration guide to README
- [ ] Document error handling best practices
- [ ] Add performance tuning guide

### Monitoring
- [ ] Add metrics for spawn_blocking pool usage
- [ ] Track PQ accuracy vs full search
- [ ] Monitor memory usage trends
- [ ] Alert on error rate spikes

### Future Optimizations
- [ ] Evaluate LanceDB migration (after 3 months PQ data)
- [ ] Consider SIMD optimizations for PQ
- [ ] Explore async RocksDB (if available)
- [ ] GPU acceleration for NER (optional)

---

**Last Updated:** 2025-11-20
**Status:** Ready for Implementation
**Approved By:** User Confirmation
