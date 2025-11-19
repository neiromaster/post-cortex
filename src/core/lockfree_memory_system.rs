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
use crate::core::lockfree_cache::{LockFreeCache, LockFreeSessionCache};
use crate::core::lockfree_performance::LockFreePerformanceMonitor;
use crate::session::active_session::ActiveSession;
use crate::storage::rocksdb_storage::RealRocksDBStorage;
use crate::workspace::LockFreeWorkspaceManager;
use anyhow;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender, channel, unbounded_channel};
use tokio::sync::{OnceCell, oneshot};

use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

// Conditional imports for embeddings feature
#[cfg(feature = "embeddings")]
use crate::core::content_vectorizer::{ContentVectorizer, ContentVectorizerConfig};
#[cfg(feature = "embeddings")]
use crate::core::embeddings::{EmbeddingConfig, EmbeddingModelType};
#[cfg(feature = "embeddings")]
use crate::core::vector_db::VectorDbConfig;

/// Configuration holder for lazy embedding initialization
#[cfg(feature = "embeddings")]
pub struct EmbeddingConfigHolder {
    pub model_type: EmbeddingModelType,
    pub vector_dimension: usize,
    pub max_vectors_per_session: usize,
    pub data_directory: String,
    pub cross_session_search_enabled: bool,
}

/// Completely lock-free conversation memory system using actors and channels
pub struct LockFreeConversationMemorySystem {
    /// Session management - lock-free
    pub session_manager: LockFreeSessionManager,

    /// Context processing - lock-free
    pub context_processor: LockFreeIncrementalContextProcessor,

    /// Graph management - lock-free
    pub graph_manager: LockFreeSimpleGraphManager,

    /// Workspace management - lock-free
    pub workspace_manager: Arc<LockFreeWorkspaceManager>,

    /// Storage actor handle
    pub storage_actor: StorageActorHandle,

    /// System configuration - immutable after creation
    pub config: SystemConfig,

    /// Performance monitoring - lock-free
    pub performance_monitor: Arc<LockFreePerformanceMonitor>,

    /// Circuit breaker state - all atomic
    pub circuit_breaker: Arc<LockFreeCircuitBreaker>,

    /// Global system metrics - atomic
    pub system_metrics: Arc<LockFreeSystemMetrics>,

    /// Embeddings and semantic search components (optional, lazy-initialized)
    #[cfg(feature = "embeddings")]
    pub content_vectorizer: Arc<OnceCell<Arc<crate::core::content_vectorizer::ContentVectorizer>>>,

    #[cfg(feature = "embeddings")]
    pub semantic_query_engine:
        Arc<OnceCell<Arc<crate::core::semantic_query_engine::SemanticQueryEngine>>>,

    /// Embedding configuration for lazy initialization
    #[cfg(feature = "embeddings")]
    pub embedding_config_holder: Arc<EmbeddingConfigHolder>,
}

/// Lock-free session manager using `DashMap` and atomic operations
pub struct LockFreeSessionManager {
    /// Session cache - completely lock-free
    pub sessions: LockFreeSessionCache<Uuid, Arc<ArcSwap<ActiveSession>>>,

    /// Storage communication
    storage_actor: StorageActorHandle,

    /// Configuration
    #[allow(dead_code)]
    config: SystemConfig,

    /// Performance monitoring
    performance_monitor: Arc<LockFreePerformanceMonitor>,

    /// Session metrics - all atomic
    pub session_count: Arc<AtomicUsize>,
    pub total_session_operations: Arc<AtomicU64>,
    pub session_cache_hits: Arc<AtomicU64>,
    pub session_cache_misses: Arc<AtomicU64>,
    pub active_sessions: DashMap<Uuid, Arc<AtomicU64>>, // last_access_timestamp
}

/// Lock-free incremental context processor
pub struct LockFreeIncrementalContextProcessor {
    #[allow(dead_code)]
    config: SystemConfig,
    #[allow(dead_code)]
    performance_monitor: Arc<LockFreePerformanceMonitor>,

    /// Processing metrics - atomic
    pub contexts_processed: Arc<AtomicU64>,
    pub processing_errors: Arc<AtomicU64>,
    pub total_processing_time_ns: Arc<AtomicU64>,
    pub avg_processing_time_ns: Arc<AtomicU64>,
}

/// Lock-free graph manager
pub struct LockFreeSimpleGraphManager {
    #[allow(dead_code)]
    config: SystemConfig,
    #[allow(dead_code)]
    performance_monitor: Arc<LockFreePerformanceMonitor>,

    /// Graph metrics - atomic
    pub entities_count: Arc<AtomicUsize>,
    pub relationships_count: Arc<AtomicUsize>,
    pub graph_operations: Arc<AtomicU64>,
    pub graph_updates: Arc<AtomicU64>,
}

/// Lock-free circuit breaker using only atomics
#[derive(Debug)]
pub struct LockFreeCircuitBreaker {
    pub is_open: AtomicBool,
    pub failure_count: AtomicU64,
    pub last_failure_timestamp: AtomicU64,
    pub success_count: AtomicU64,
    pub last_success_timestamp: AtomicU64,

    // Configuration
    pub failure_threshold: u64,
    pub timeout_seconds: u64,
}

/// Global system metrics - all atomic
#[derive(Debug)]
pub struct LockFreeSystemMetrics {
    pub total_requests: AtomicU64,
    pub successful_requests: AtomicU64,
    pub failed_requests: AtomicU64,
    pub active_operations: AtomicUsize,
    pub storage_operations: AtomicU64,
    pub cache_operations: AtomicU64,
    pub start_timestamp: AtomicU64,
}

/// Storage actor for handling all storage operations asynchronously
pub struct StorageActor {
    storage: RealRocksDBStorage,
    receiver: UnboundedReceiver<StorageMessage>,
    performance_monitor: Arc<LockFreePerformanceMonitor>,
    operation_count: AtomicU64,
}

/// Handle for communicating with storage actor
#[derive(Clone)]
pub struct StorageActorHandle {
    sender: UnboundedSender<StorageMessage>,
    operation_timeout: Duration,
}

/// Messages for storage actor
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum StorageMessage {
    LoadSession {
        session_id: Uuid,
        response_tx: Sender<Result<Option<ActiveSession>, String>>,
    },
    SaveSession {
        session: Box<ActiveSession>,
        response_tx: Sender<Result<(), String>>,
    },
    DeleteSession {
        session_id: Uuid,
        response_tx: Sender<Result<bool, String>>,
    },
    ListSessions {
        response_tx: Sender<Result<Vec<Uuid>, String>>,
    },
    GetStats {
        response_tx: Sender<StorageStats>,
    },
    SaveCheckpoint {
        checkpoint: crate::storage::rocksdb_storage::SessionCheckpoint,
        response_tx: Sender<Result<(), String>>,
    },
    LoadCheckpoint {
        checkpoint_id: Uuid,
        response_tx: Sender<Result<crate::storage::rocksdb_storage::SessionCheckpoint, String>>,
    },
    SaveWorkspaceMetadata {
        workspace_id: Uuid,
        name: String,
        description: String,
        session_ids: Vec<Uuid>,
        response_tx: Sender<Result<(), String>>,
    },
    DeleteWorkspace {
        workspace_id: Uuid,
        response_tx: Sender<Result<(), String>>,
    },
    AddSessionToWorkspace {
        workspace_id: Uuid,
        session_id: Uuid,
        role: crate::workspace::SessionRole,
        response_tx: Sender<Result<(), String>>,
    },
    RemoveSessionFromWorkspace {
        workspace_id: Uuid,
        session_id: Uuid,
        response_tx: Sender<Result<(), String>>,
    },
    Shutdown,
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_operations: u64,
    pub load_operations: u64,
    pub save_operations: u64,
    pub delete_operations: u64,
    pub avg_operation_time_ns: u64,
    pub last_operation_timestamp: u64,
}

/// System configuration
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct SystemConfig {
    pub max_hot_context_size: usize,
    pub max_warm_context_size: usize,
    pub context_compression_threshold: usize,
    pub session_timeout_minutes: u64,
    pub storage_timeout_seconds: u64,
    pub cache_capacity: usize,
    pub enable_performance_monitoring: bool,
    pub circuit_breaker_failure_threshold: u64,
    pub circuit_breaker_timeout_seconds: u64,
    pub data_directory: String,
    // Entity extraction configuration
    pub max_extracted_entities: usize,
    pub max_referenced_entities: usize,
    pub enable_smart_entity_ranking: bool,
    // Embeddings and vectorization configuration
    pub enable_embeddings: bool,
    pub embeddings_model_type: String,
    pub vector_dimension: usize,
    pub max_vectors_per_session: usize,
    pub semantic_search_threshold: f32,
    pub auto_vectorize_on_update: bool,
    pub cross_session_search_enabled: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            max_hot_context_size: 50,
            max_warm_context_size: 200,
            context_compression_threshold: 1000,
            session_timeout_minutes: 30,
            storage_timeout_seconds: 10,
            cache_capacity: 1000,
            enable_performance_monitoring: true,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_timeout_seconds: 300, // 5 minutes
            data_directory: "./post_cortex_data".to_string(),
            // Entity extraction defaults
            max_extracted_entities: 15,
            max_referenced_entities: 15,
            enable_smart_entity_ranking: true, // Enable by default for better quality
            // Embeddings defaults
            enable_embeddings: true, // Enabled for semantic search functionality
            embeddings_model_type: "MiniLM".to_string(),
            vector_dimension: 384, // MiniLM uses 384-dimensional embeddings
            max_vectors_per_session: 1000,
            semantic_search_threshold: 0.7,
            auto_vectorize_on_update: true,
            cross_session_search_enabled: true,
        }
    }
}

impl LockFreeConversationMemorySystem {
    /// Create system from config (backward compatibility)
    ///
    /// # Errors
    /// Returns an error if storage initialization fails or if system setup encounters any issues
    pub async fn new(config: SystemConfig) -> Result<Self, String> {
        // Create storage from config
        let storage = RealRocksDBStorage::new(&config.data_directory)
            .await
            .map_err(|e| format!("Failed to initialize storage: {e}"))?;

        Self::new_with_storage(storage, config).await
    }

    /// Create system with existing storage
    pub async fn new_with_storage(
        storage: RealRocksDBStorage,
        config: SystemConfig,
    ) -> Result<Self, String> {
        let performance_monitor = Arc::new(LockFreePerformanceMonitor::new(None));

        // Create storage actor
        let storage_actor = StorageActor::spawn(storage, Arc::clone(&performance_monitor)).await?;

        // Create session cache
        let session_cache = LockFreeCache::new(config.cache_capacity, "session_cache".to_string())
            .map_err(|e| format!("Failed to create session cache: {e}"))?;

        let session_manager = LockFreeSessionManager {
            sessions: session_cache,
            storage_actor: storage_actor.clone(),
            config: config.clone(),
            performance_monitor: Arc::clone(&performance_monitor),
            session_count: Arc::new(AtomicUsize::new(0)),
            total_session_operations: Arc::new(AtomicU64::new(0)),
            session_cache_hits: Arc::new(AtomicU64::new(0)),
            session_cache_misses: Arc::new(AtomicU64::new(0)),
            active_sessions: DashMap::new(),
        };

        let context_processor = LockFreeIncrementalContextProcessor {
            config: config.clone(),
            performance_monitor: Arc::clone(&performance_monitor),
            contexts_processed: Arc::new(AtomicU64::new(0)),
            processing_errors: Arc::new(AtomicU64::new(0)),
            total_processing_time_ns: Arc::new(AtomicU64::new(0)),
            avg_processing_time_ns: Arc::new(AtomicU64::new(0)),
        };

        let graph_manager = LockFreeSimpleGraphManager {
            config: config.clone(),
            performance_monitor: Arc::clone(&performance_monitor),
            entities_count: Arc::new(AtomicUsize::new(0)),
            relationships_count: Arc::new(AtomicUsize::new(0)),
            graph_operations: Arc::new(AtomicU64::new(0)),
            graph_updates: Arc::new(AtomicU64::new(0)),
        };

        let circuit_breaker = Arc::new(LockFreeCircuitBreaker::new(
            config.circuit_breaker_failure_threshold,
            config.circuit_breaker_timeout_seconds,
        ));

        let system_metrics = Arc::new(LockFreeSystemMetrics::new());

        // Create workspace manager
        let workspace_manager = Arc::new(LockFreeWorkspaceManager::new());

        // Prepare embeddings configuration for lazy initialization
        #[cfg(feature = "embeddings")]
        let (content_vectorizer, semantic_query_engine, embedding_config_holder) =
            if config.enable_embeddings {
                info!("Embeddings enabled - will lazy-initialize on first use");

                // Parse embedding model type
                let model_type = match config.embeddings_model_type.as_str() {
                    "StaticSimilarityMRL" => EmbeddingModelType::StaticSimilarityMRL,
                    "MiniLM" => EmbeddingModelType::MiniLM,
                    "TinyBERT" => EmbeddingModelType::TinyBERT,
                    "BGESmall" => EmbeddingModelType::BGESmall,
                    _ => {
                        warn!(
                            "Unknown embedding model type: {}, defaulting to StaticSimilarityMRL",
                            config.embeddings_model_type
                        );
                        EmbeddingModelType::StaticSimilarityMRL
                    }
                };

                // Store configuration for lazy initialization
                let embedding_config_holder = Arc::new(EmbeddingConfigHolder {
                    model_type,
                    vector_dimension: config.vector_dimension,
                    max_vectors_per_session: config.max_vectors_per_session,
                    data_directory: config.data_directory.clone(),
                    cross_session_search_enabled: config.cross_session_search_enabled,
                });

                (
                    Arc::new(OnceCell::new()),
                    Arc::new(OnceCell::new()),
                    embedding_config_holder,
                )
            } else {
                info!("Embeddings disabled in configuration");
                (
                    Arc::new(OnceCell::new()),
                    Arc::new(OnceCell::new()),
                    Arc::new(EmbeddingConfigHolder {
                        model_type: EmbeddingModelType::StaticSimilarityMRL,
                        vector_dimension: 1024,
                        max_vectors_per_session: 10000,
                        data_directory: config.data_directory.clone(),
                        cross_session_search_enabled: false,
                    }),
                )
            };

        #[cfg(not(feature = "embeddings"))]
        let (_content_vectorizer, _semantic_query_engine) = {
            if config.enable_embeddings {
                warn!("Embeddings requested but 'embeddings' feature not enabled");
            }
        };

        Ok(Self {
            session_manager,
            context_processor,
            graph_manager,
            workspace_manager,
            storage_actor,
            config,
            performance_monitor,
            circuit_breaker,
            system_metrics,
            #[cfg(feature = "embeddings")]
            content_vectorizer,
            #[cfg(feature = "embeddings")]
            semantic_query_engine,
            #[cfg(feature = "embeddings")]
            embedding_config_holder,
        })
    }

    /// Create a new session with name and description
    pub async fn create_session(
        &self,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<Uuid, String> {
        let _timer = self.performance_monitor.start_timer("create_session");
        self.system_metrics
            .total_requests
            .fetch_add(1, Ordering::Relaxed);

        let session_id = Uuid::new_v4();

        // Create session with provided name and description
        let session_name = name.or_else(|| {
            Some(format!(
                "Session {}",
                session_id
                    .to_string()
                    .split('-')
                    .next()
                    .unwrap_or("unknown")
            ))
        });

        let session_description =
            description.or_else(|| Some("New conversation session".to_string()));

        let session = ActiveSession::new(session_id, session_name, session_description);
        let session_arc = Arc::new(ArcSwap::new(Arc::new(session.clone())));

        // Save to storage first
        match self.storage_actor.save_session(session).await {
            Ok(_) => {
                // Add to cache and active sessions only if storage save succeeded
                self.session_manager
                    .sessions
                    .put(session_id, Arc::clone(&session_arc));

                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("System time before UNIX epoch")
                    .as_secs();
                self.session_manager
                    .active_sessions
                    .insert(session_id, Arc::new(AtomicU64::new(now)));
                self.session_manager
                    .session_count
                    .fetch_add(1, Ordering::Relaxed);

                self.system_metrics
                    .successful_requests
                    .fetch_add(1, Ordering::Relaxed);
                info!("Created new session: {}", session_id);

                Ok(session_id)
            }
            Err(e) => {
                self.system_metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
                Err(format!("Failed to save session to storage: {e}"))
            }
        }
    }

    /// Get existing session by ID
    pub async fn get_session(
        &self,
        session_id: Uuid,
    ) -> Result<Arc<ArcSwap<ActiveSession>>, String> {
        let _timer = self.performance_monitor.start_timer("get_session");

        if let Some(session_arc) = self.session_manager.sessions.get(&session_id) {
            // Update access time
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("System time before UNIX epoch")
                .as_secs();
            if let Some(access_time) = self.session_manager.active_sessions.get(&session_id) {
                access_time.store(now, Ordering::Relaxed);
            }
            return Ok(session_arc);
        }

        // Try loading from storage via actor
        match self.storage_actor.load_session(session_id).await {
            Ok(Some(session)) => {
                let session_arc = Arc::new(ArcSwap::new(Arc::new(session)));
                self.session_manager
                    .sessions
                    .put(session_id, Arc::clone(&session_arc));

                // Add to active sessions
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("System time before UNIX epoch")
                    .as_secs();
                self.session_manager
                    .active_sessions
                    .insert(session_id, Arc::new(AtomicU64::new(now)));

                Ok(session_arc)
            }
            Ok(None) => Err(format!("Session {session_id} not found")),
            Err(e) => Err(format!("Storage error: {e}")),
        }
    }

    // MCP Compatibility methods

    /// Get storage actor handle for compatibility (replaces storage.read().await)
    pub fn get_storage(&self) -> &StorageActorHandle {
        &self.storage_actor
    }

    /// Update session metadata compatibility method
    pub async fn update_session_metadata(
        &self,
        session_id: Uuid,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<(), String> {
        let session_arc = self.get_session(session_id).await?;

        // Update session atomically
        let current_session = session_arc.load();
        let mut new_session = (**current_session).clone();

        if let Some(new_name) = name {
            new_session.name = Some(new_name);
        }
        if let Some(new_desc) = description {
            new_session.description = Some(new_desc);
        }
        new_session.last_updated = chrono::Utc::now();

        session_arc.store(Arc::new(new_session));
        Ok(())
    }

    /// Find sessions by name or description compatibility method
    pub async fn find_sessions_by_name_or_description(
        &self,
        query: &str,
    ) -> Result<Vec<Uuid>, String> {
        // Simple implementation - get all sessions and filter
        let session_ids = self.list_sessions().await?;
        let mut matching_sessions = Vec::new();

        for session_id in session_ids {
            if let Ok(session_arc) = self.get_session(session_id).await {
                let session = session_arc.load();

                let name_match = session
                    .name
                    .as_ref()
                    .map(|n| n.to_lowercase().contains(&query.to_lowercase()))
                    .unwrap_or(false);

                let desc_match = session
                    .description
                    .as_ref()
                    .map(|d| d.to_lowercase().contains(&query.to_lowercase()))
                    .unwrap_or(false);

                if name_match || desc_match {
                    matching_sessions.push(session_id);
                }
            }
        }

        Ok(matching_sessions)
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Result<Vec<Uuid>, String> {
        let _timer = self.performance_monitor.start_timer("list_sessions");

        // Get sessions from storage via actor
        match self.storage_actor.list_sessions().await {
            Ok(session_ids) => {
                self.system_metrics
                    .successful_requests
                    .fetch_add(1, Ordering::Relaxed);
                Ok(session_ids)
            }
            Err(e) => {
                self.system_metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
                Err(format!("Failed to list sessions: {e}"))
            }
        }
    }

    /// Add incremental update - compatibility wrapper
    #[instrument(skip(self, session_id, description))]
    pub async fn add_incremental_update(
        &self,
        session_id: Uuid,
        description: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, String> {
        tracing::info!("add_incremental_update START for session {}", session_id);
        let _timer = self
            .performance_monitor
            .start_timer("add_incremental_update");
        self.system_metrics
            .total_requests
            .fetch_add(1, Ordering::Relaxed);
        self.system_metrics
            .active_operations
            .fetch_add(1, Ordering::Relaxed);

        tracing::info!("Metrics updated, checking circuit breaker...");

        // Circuit breaker check - atomic
        if self.circuit_breaker.is_open() {
            warn!("Circuit breaker is open - rejecting request");
            self.system_metrics
                .active_operations
                .fetch_sub(1, Ordering::Relaxed);
            return Err("System temporarily unavailable - circuit breaker open".to_string());
        }

        tracing::info!("Calling internal implementation...");
        let result = self
            .add_incremental_update_internal(session_id, description, metadata)
            .await;
        tracing::info!("Internal implementation returned: {:?}", result.is_ok());

        // Update circuit breaker based on result
        match &result {
            Ok(_) => {
                self.circuit_breaker.record_success();
                self.system_metrics
                    .successful_requests
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.circuit_breaker.record_failure();
                self.system_metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        self.system_metrics
            .active_operations
            .fetch_sub(1, Ordering::Relaxed);
        result
    }

    async fn add_incremental_update_internal(
        &self,
        session_id: Uuid,
        mut description: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, String> {
        tracing::info!("Internal: Processing session {}", session_id);
        // Content size limit to prevent timeouts
        if description.len() > 2000 {
            description.truncate(1800);
            description.push_str("... (truncated)");
            debug!("Truncated long description to prevent timeout");
        }

        tracing::info!("Internal: Getting or creating session...");
        // Get or create session - lock-free
        let session_arc = self
            .session_manager
            .get_or_create_session(session_id)
            .await?;
        tracing::info!("Internal: Session obtained successfully");

        // Update session atomically using ArcSwap
        let update_result = {
            let start = std::time::Instant::now();

            // Load current session
            tracing::info!("DEBUG: About to load current session");
            let current_session = session_arc.load();
            tracing::info!("DEBUG: Loaded current session, about to clone");
            let mut new_session = (**current_session).clone();
            tracing::info!("DEBUG: Cloned session, about to call add_context_update_to_session");

            // Add context update
            let result = Self::add_context_update_to_session(
                &mut new_session,
                description.clone(),
                metadata.clone(),
            )
            .await;

            tracing::info!("DEBUG: add_context_update_to_session returned, about to store");
            // Store updated session atomically
            session_arc.store(Arc::new(new_session));
            tracing::info!("DEBUG: Stored updated session");

            // Update processing metrics
            let processing_time_ns = start.elapsed().as_nanos() as u64;
            let total_time = self
                .context_processor
                .total_processing_time_ns
                .fetch_add(processing_time_ns, Ordering::Relaxed)
                + processing_time_ns;
            let processed_count = self
                .context_processor
                .contexts_processed
                .fetch_add(1, Ordering::Relaxed)
                + 1;

            // Update cached average
            let avg_time = total_time / processed_count;
            self.context_processor
                .avg_processing_time_ns
                .store(avg_time, Ordering::Relaxed);

            result
        };

        match update_result {
            Ok(update_id) => {
                // Update graph metrics
                self.graph_manager
                    .graph_operations
                    .fetch_add(1, Ordering::Relaxed);
                self.graph_manager
                    .graph_updates
                    .fetch_add(1, Ordering::Relaxed);

                // Save updated session to storage
                let current_session = session_arc.load();
                if let Err(save_error) = self
                    .storage_actor
                    .save_session((**current_session).clone())
                    .await
                {
                    warn!("Failed to persist session {}: {}", session_id, save_error);
                    // Continue anyway - session is still updated in memory
                } else {
                    debug!("Session {} persisted to storage", session_id);
                }

                // Auto-vectorize if embeddings are enabled
                #[cfg(feature = "embeddings")]
                {
                    if let Err(e) = self.auto_vectorize_if_enabled(session_id).await {
                        // Don't fail the main operation if auto-vectorization fails
                        debug!(
                            "Auto-vectorization warning for session {}: {}",
                            session_id, e
                        );
                    }
                }

                info!(
                    "Added incremental update to session {}: {} chars",
                    session_id,
                    description.len()
                );
                Ok(update_id)
            }
            Err(e) => {
                self.context_processor
                    .processing_errors
                    .fetch_add(1, Ordering::Relaxed);
                warn!("Failed to add incremental update: {}", e);
                Err(e)
            }
        }
    }

    /// Get conversation context - lock-free
    pub async fn get_conversation_context(&self, session_id: Uuid) -> Result<String, String> {
        let _timer = self
            .performance_monitor
            .start_timer("get_conversation_context");
        self.system_metrics
            .total_requests
            .fetch_add(1, Ordering::Relaxed);

        // Try cache first - lock-free
        if let Some(session_arc) = self.session_manager.sessions.get(&session_id) {
            self.session_manager
                .session_cache_hits
                .fetch_add(1, Ordering::Relaxed);

            // Load session atomically
            let session = session_arc.load();
            let context = Self::get_session_context_summary(&session);

            // Update access timestamp
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("System time before UNIX epoch")
                .as_secs();
            if let Some(access_time) = self.session_manager.active_sessions.get(&session_id) {
                access_time.store(now, Ordering::Relaxed);
            }

            self.system_metrics
                .successful_requests
                .fetch_add(1, Ordering::Relaxed);
            return Ok(context);
        }

        // Cache miss - load from storage via actor
        self.session_manager
            .session_cache_misses
            .fetch_add(1, Ordering::Relaxed);

        match self.storage_actor.load_session(session_id).await {
            Ok(Some(session)) => {
                // Cache the loaded session using ArcSwap
                let session_arc = Arc::new(ArcSwap::new(Arc::new(session)));
                self.session_manager
                    .sessions
                    .put(session_id, Arc::clone(&session_arc));

                // Add to active sessions
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("System time before UNIX epoch")
                    .as_secs();
                self.session_manager
                    .active_sessions
                    .insert(session_id, Arc::new(AtomicU64::new(now)));

                let context = Self::get_session_context_summary(&session_arc.load());
                self.system_metrics
                    .successful_requests
                    .fetch_add(1, Ordering::Relaxed);
                Ok(context)
            }
            Ok(None) => {
                self.system_metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
                Err("Session not found".to_string())
            }
            Err(e) => {
                self.system_metrics
                    .failed_requests
                    .fetch_add(1, Ordering::Relaxed);
                Err(format!("Storage error: {e}"))
            }
        }
    }

    /// Get system health - all atomic reads
    pub fn get_system_health(&self) -> LockFreeSystemHealth {
        let _perf_snapshot = self.performance_monitor.get_snapshot();
        let cache_stats = self.session_manager.sessions.get_stats();
        let circuit_breaker_stats = self.circuit_breaker.get_stats();

        LockFreeSystemHealth {
            total_requests: self.system_metrics.total_requests.load(Ordering::Relaxed),
            successful_requests: self
                .system_metrics
                .successful_requests
                .load(Ordering::Relaxed),
            failed_requests: self.system_metrics.failed_requests.load(Ordering::Relaxed),
            active_operations: self
                .system_metrics
                .active_operations
                .load(Ordering::Relaxed),
            active_sessions: self.session_manager.active_sessions.len(),
            circuit_breaker_open: circuit_breaker_stats.is_open,
            circuit_breaker_failures: circuit_breaker_stats.failure_count,
            cache_hit_rate: cache_stats.hit_rate,
            contexts_processed: self
                .context_processor
                .contexts_processed
                .load(Ordering::Relaxed),
            processing_errors: self
                .context_processor
                .processing_errors
                .load(Ordering::Relaxed),
            storage_operations: self
                .system_metrics
                .storage_operations
                .load(Ordering::Relaxed),
            uptime_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("System time before UNIX epoch")
                .as_secs()
                .saturating_sub(self.system_metrics.start_timestamp.load(Ordering::Relaxed)),
        }
    }

    /// Cleanup expired sessions - background task
    pub async fn cleanup_expired_sessions(&self) {
        let _timer = self
            .performance_monitor
            .start_timer("cleanup_expired_sessions");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time before UNIX epoch")
            .as_secs();
        let timeout_seconds = self.config.session_timeout_minutes * 60;

        let mut expired_sessions = Vec::new();

        // Find expired sessions
        for entry in self.session_manager.active_sessions.iter() {
            let last_access = entry.value().load(Ordering::Relaxed);
            if now.saturating_sub(last_access) > timeout_seconds {
                expired_sessions.push(*entry.key());
            }
        }

        // Remove expired sessions
        let expired_count = expired_sessions.len();
        for session_id in expired_sessions {
            self.session_manager.active_sessions.remove(&session_id);
            self.session_manager.sessions.remove(&session_id);
            debug!("Cleaned up expired session: {}", session_id);
        }

        if expired_count > 0 {
            info!("Cleaned up {} expired sessions", expired_count);
        }
    }
}

impl LockFreeSessionManager {
    /// Get or create session - lock-free with actor communication
    pub async fn get_or_create_session(
        &self,
        session_id: Uuid,
    ) -> Result<Arc<ArcSwap<ActiveSession>>, String> {
        let _timer = self
            .performance_monitor
            .start_timer("get_or_create_session");
        self.total_session_operations
            .fetch_add(1, Ordering::Relaxed);

        // Try cache first
        if let Some(session_arc) = self.sessions.get(&session_id) {
            self.session_cache_hits.fetch_add(1, Ordering::Relaxed);

            // Update access time
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if let Some(access_time) = self.active_sessions.get(&session_id) {
                access_time.store(now, Ordering::Relaxed);
            }

            return Ok(session_arc);
        }

        // Cache miss

        self.session_cache_misses.fetch_add(1, Ordering::Relaxed);

        // Try loading from storage via actor
        let storage_result: Result<Option<ActiveSession>, String> = match self
            .storage_actor
            .load_session(session_id)
            .await
        {
            Ok(Some(session)) => Ok(Some(session)),
            Ok(None) => Ok(None), // Session not found is OK
            Err(e) => {
                tracing::error!(
                    "Storage error loading session {}: {}. Treating as not found and creating new session.",
                    session_id,
                    e
                );
                Ok(None) // Treat storage errors as session not found
            }
        };

        match storage_result {
            Ok(Some(session)) => {
                let session_arc = Arc::new(ArcSwap::new(Arc::new(session)));
                self.sessions.put(session_id, Arc::clone(&session_arc));

                // Add to active sessions
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                self.active_sessions
                    .insert(session_id, Arc::new(AtomicU64::new(now)));

                Ok(session_arc)
            }
            Ok(None) => {
                // Session not found - create new one
                tracing::info!(
                    "Creating new session {} since not found in storage",
                    session_id
                );
                let new_session = ActiveSession::new(session_id, None, None);
                let session_arc = Arc::new(ArcSwap::new(Arc::new(new_session.clone())));

                // Save to storage first, then cache it
                match self.storage_actor.save_session(new_session).await {
                    Ok(_) => {
                        // Cache it only if storage save succeeded
                        self.sessions.put(session_id, Arc::clone(&session_arc));

                        // Add to active sessions
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        self.active_sessions
                            .insert(session_id, Arc::new(AtomicU64::new(now)));

                        self.session_count.fetch_add(1, Ordering::Relaxed);
                        Ok(session_arc)
                    }
                    Err(e) => Err(format!("Failed to save new session to storage: {e}")),
                }
            }
            Err(e) => Err(format!("Storage error: {e}")),
        }
    }

    /// Get session metrics - all atomic reads
    pub fn get_metrics(&self) -> LockFreeSessionManagerMetrics {
        let cache_stats = self.sessions.get_stats();

        LockFreeSessionManagerMetrics {
            session_count: self.session_count.load(Ordering::Relaxed),
            active_sessions: self.active_sessions.len(),
            total_operations: self.total_session_operations.load(Ordering::Relaxed),
            cache_hits: self.session_cache_hits.load(Ordering::Relaxed),
            cache_misses: self.session_cache_misses.load(Ordering::Relaxed),
            cache_hit_rate: cache_stats.hit_rate,
            cache_size: cache_stats.current_size,
            cache_capacity: cache_stats.capacity,
        }
    }
}

impl LockFreeCircuitBreaker {
    pub fn new(failure_threshold: u64, timeout_seconds: u64) -> Self {
        Self {
            is_open: AtomicBool::new(false),
            failure_count: AtomicU64::new(0),
            last_failure_timestamp: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_success_timestamp: AtomicU64::new(0),
            failure_threshold,
            timeout_seconds,
        }
    }

    pub fn is_open(&self) -> bool {
        if !self.is_open.load(Ordering::Relaxed) {
            return false;
        }

        // Check if timeout has passed
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last_failure = self.last_failure_timestamp.load(Ordering::Relaxed);

        if now.saturating_sub(last_failure) > self.timeout_seconds {
            // Reset circuit breaker
            self.is_open.store(false, Ordering::Relaxed);
            self.failure_count.store(0, Ordering::Relaxed);
            debug!("Circuit breaker reset after timeout");
            false
        } else {
            true
        }
    }

    pub fn record_failure(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.last_failure_timestamp.store(now, Ordering::Relaxed);

        if failures >= self.failure_threshold {
            self.is_open.store(true, Ordering::Relaxed);
            warn!("Circuit breaker opened after {} failures", failures);
        }
    }

    pub fn record_success(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.last_success_timestamp.store(now, Ordering::Relaxed);

        // Gradually reduce failure count on success
        let current_failures = self.failure_count.load(Ordering::Relaxed);
        if current_failures > 0 {
            self.failure_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn get_stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            is_open: self.is_open.load(Ordering::Relaxed),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            last_failure_timestamp: self.last_failure_timestamp.load(Ordering::Relaxed),
            last_success_timestamp: self.last_success_timestamp.load(Ordering::Relaxed),
        }
    }
}

impl Default for LockFreeSystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl LockFreeSystemMetrics {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            active_operations: AtomicUsize::new(0),
            storage_operations: AtomicU64::new(0),
            cache_operations: AtomicU64::new(0),
            start_timestamp: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            ),
        }
    }
}

impl StorageActorHandle {
    pub async fn load_session(&self, session_id: Uuid) -> Result<Option<ActiveSession>, String> {
        tracing::info!(
            "StorageHandle: Sending LoadSession message for {}",
            session_id
        );
        let (response_tx, mut response_rx) = channel::<Result<Option<ActiveSession>, String>>(1);

        self.sender
            .send(StorageMessage::LoadSession {
                session_id,
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tracing::info!(
            "StorageHandle: Waiting for response with {}s timeout...",
            self.operation_timeout.as_secs()
        );
        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn save_session(&self, session: ActiveSession) -> Result<(), String> {
        let (response_tx, mut response_rx) = channel::<Result<(), String>>(1);

        self.sender
            .send(StorageMessage::SaveSession {
                session: Box::new(session),
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn list_sessions(&self) -> Result<Vec<Uuid>, String> {
        let (response_tx, mut response_rx) = channel::<Result<Vec<Uuid>, String>>(1);

        self.sender
            .send(StorageMessage::ListSessions { response_tx })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn save_checkpoint(
        &self,
        checkpoint: &crate::storage::rocksdb_storage::SessionCheckpoint,
    ) -> Result<(), String> {
        let (response_tx, mut response_rx) = channel::<Result<(), String>>(1);

        self.sender
            .send(StorageMessage::SaveCheckpoint {
                checkpoint: checkpoint.clone(),
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn load_checkpoint(
        &self,
        checkpoint_id: uuid::Uuid,
    ) -> Result<crate::storage::rocksdb_storage::SessionCheckpoint, String> {
        let (response_tx, mut response_rx) =
            channel::<Result<crate::storage::rocksdb_storage::SessionCheckpoint, String>>(1);

        self.sender
            .send(StorageMessage::LoadCheckpoint {
                checkpoint_id,
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn save_workspace_metadata(
        &self,
        workspace_id: Uuid,
        name: &str,
        description: &str,
        session_ids: &Vec<Uuid>,
    ) -> Result<(), String> {
        let (response_tx, mut response_rx) = channel::<Result<(), String>>(1);

        self.sender
            .send(StorageMessage::SaveWorkspaceMetadata {
                workspace_id,
                name: name.to_string(),
                description: description.to_string(),
                session_ids: session_ids.clone(),
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn delete_workspace(&self, workspace_id: Uuid) -> Result<(), String> {
        let (response_tx, mut response_rx) = channel::<Result<(), String>>(1);

        self.sender
            .send(StorageMessage::DeleteWorkspace {
                workspace_id,
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn add_session_to_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
        role: crate::workspace::SessionRole,
    ) -> Result<(), String> {
        let (response_tx, mut response_rx) = channel::<Result<(), String>>(1);

        self.sender
            .send(StorageMessage::AddSessionToWorkspace {
                workspace_id,
                session_id,
                role,
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }

    pub async fn remove_session_from_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
    ) -> Result<(), String> {
        let (response_tx, mut response_rx) = channel::<Result<(), String>>(1);

        self.sender
            .send(StorageMessage::RemoveSessionFromWorkspace {
                workspace_id,
                session_id,
                response_tx,
            })
            .map_err(|_| "Storage actor unavailable".to_string())?;

        tokio::time::timeout(self.operation_timeout, response_rx.recv())
            .await
            .map_err(|_| "Storage operation timed out".to_string())?
            .ok_or_else(|| "Storage actor response channel closed".to_string())?
    }
}

impl StorageActor {
    pub async fn spawn(
        storage: RealRocksDBStorage,
        performance_monitor: Arc<LockFreePerformanceMonitor>,
    ) -> Result<StorageActorHandle, String> {
        let (sender, receiver) = unbounded_channel();

        let actor = Self {
            storage,
            receiver,
            performance_monitor,
            operation_count: AtomicU64::new(0),
        };

        // Create confirmation channel for startup synchronization
        let (startup_tx, startup_rx) = oneshot::channel();

        // Spawn async actor task with startup confirmation
        tokio::spawn(async move {
            // Send confirmation that actor is ready
            let _ = startup_tx.send(());
            actor.run_async().await;
        });

        // Wait for actor to be ready before returning handle
        startup_rx
            .await
            .map_err(|_| "Storage actor failed to start".to_string())?;

        Ok(StorageActorHandle {
            sender,
            operation_timeout: Duration::from_secs(60),
        })
    }

    async fn run_async(mut self) {
        info!("Storage actor started (async)");

        while let Some(message) = self.receiver.recv().await {
            match message {
                StorageMessage::Shutdown => {
                    info!("Storage actor shutting down");
                    break;
                }
                msg => self.handle_message_async(msg).await,
            }
        }

        info!("Storage actor stopped");
    }

    async fn handle_message_async(&self, message: StorageMessage) {
        tracing::info!(
            "StorageActor: Handling message: {:?}",
            std::mem::discriminant(&message)
        );
        let _timer = self.performance_monitor.start_timer("storage_operation");
        self.operation_count.fetch_add(1, Ordering::Relaxed);

        match message {
            StorageMessage::LoadSession {
                session_id,
                response_tx,
            } => {
                tracing::info!("StorageActor: Loading session {}", session_id);
                let result = match self.storage.load_session(session_id).await {
                    Ok(session) => Ok(Some(session)),
                    Err(_) => Ok(None), // Session not found is OK, not an error
                };
                let _ = response_tx.send(result).await;
            }
            StorageMessage::SaveSession {
                session,
                response_tx,
            } => {
                let result = self
                    .storage
                    .save_session(&session)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::DeleteSession {
                session_id,
                response_tx,
            } => {
                let result = self
                    .storage
                    .delete_session(session_id)
                    .await
                    .map(|_| true)
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::ListSessions { response_tx } => {
                let result = self
                    .storage
                    .list_sessions()
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::GetStats { response_tx } => {
                let stats = StorageStats {
                    total_operations: self.operation_count.load(Ordering::Relaxed),
                    load_operations: 0, // TODO: implement detailed counters
                    save_operations: 0,
                    delete_operations: 0,
                    avg_operation_time_ns: 0,
                    last_operation_timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("System time before UNIX epoch")
                        .as_secs(),
                };
                let _ = response_tx.send(stats).await;
            }
            StorageMessage::SaveCheckpoint {
                checkpoint,
                response_tx,
            } => {
                let result = self
                    .storage
                    .save_checkpoint(&checkpoint)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::LoadCheckpoint {
                checkpoint_id,
                response_tx,
            } => {
                let result = self
                    .storage
                    .load_checkpoint(checkpoint_id)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::SaveWorkspaceMetadata {
                workspace_id,
                name,
                description,
                session_ids,
                response_tx,
            } => {
                let result = self
                    .storage
                    .save_workspace_metadata(workspace_id, &name, &description, &session_ids)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::DeleteWorkspace {
                workspace_id,
                response_tx,
            } => {
                let result = self
                    .storage
                    .delete_workspace(workspace_id)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::AddSessionToWorkspace {
                workspace_id,
                session_id,
                role,
                response_tx,
            } => {
                let result = self
                    .storage
                    .add_session_to_workspace(workspace_id, session_id, role)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::RemoveSessionFromWorkspace {
                workspace_id,
                session_id,
                response_tx,
            } => {
                let result = self
                    .storage
                    .remove_session_from_workspace(workspace_id, session_id)
                    .await
                    .map_err(|e| e.to_string());
                let _ = response_tx.send(result).await;
            }
            StorageMessage::Shutdown => {} // Handled in main loop
        }
    }
}

// Health and metrics structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFreeSystemHealth {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_operations: usize,
    pub active_sessions: usize,
    pub circuit_breaker_open: bool,
    pub circuit_breaker_failures: u64,
    pub cache_hit_rate: f64,
    pub contexts_processed: u64,
    pub processing_errors: u64,
    pub storage_operations: u64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockFreeSessionManagerMetrics {
    pub session_count: usize,
    pub active_sessions: usize,
    pub total_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub cache_size: usize,
    pub cache_capacity: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    pub is_open: bool,
    pub failure_count: u64,
    pub success_count: u64,
    pub last_failure_timestamp: u64,
    pub last_success_timestamp: u64,
}

/// Extensions for ActiveSession to work with lock-free system
impl LockFreeConversationMemorySystem {
    async fn add_context_update_to_session(
        session: &mut ActiveSession,
        description: String,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, String> {
        use crate::core::context_update::{ContextUpdate, UpdateContent, UpdateType};

        tracing::info!(
            "add_context_update_to_session: Starting with description: '{}'",
            description
        );
        // Try to deserialize complete ContextUpdate from metadata
        let update = if let Some(metadata) = metadata {
            if let Ok(context_update) = serde_json::from_value::<ContextUpdate>(metadata) {
                tracing::info!(
                    "Using provided ContextUpdate with ID: {}",
                    context_update.id
                );
                context_update
            } else {
                tracing::info!("Failed to deserialize ContextUpdate, creating new one");
                let update_id = Uuid::new_v4();
                ContextUpdate {
                    id: update_id,
                    update_type: UpdateType::ConceptDefined,
                    content: UpdateContent {
                        title: "Incremental Update".to_string(),
                        description,
                        details: Vec::new(),
                        examples: Vec::new(),
                        implications: Vec::new(),
                    },
                    timestamp: chrono::Utc::now(),
                    related_code: None,
                    parent_update: None,
                    user_marked_important: false,
                    creates_entities: Vec::new(),
                    creates_relationships: Vec::new(),
                    references_entities: Vec::new(),
                }
            }
        } else {
            tracing::info!("No metadata provided, creating new ContextUpdate");
            let update_id = Uuid::new_v4();
            ContextUpdate {
                id: update_id,
                update_type: UpdateType::ConceptDefined,
                content: UpdateContent {
                    title: "Incremental Update".to_string(),
                    description,
                    details: Vec::new(),
                    examples: Vec::new(),
                    implications: Vec::new(),
                },
                timestamp: chrono::Utc::now(),
                related_code: None,
                parent_update: None,
                user_marked_important: false,
                creates_entities: Vec::new(),
                creates_relationships: Vec::new(),
                references_entities: Vec::new(),
            }
        };

        tracing::info!("Using ContextUpdate with ID: {}", update.id);
        tracing::info!(
            "Update content - title: '{}', description: '{}', has_code: {}",
            update.content.title,
            update.content.description,
            update.related_code.is_some()
        );

        // Use the proper add_incremental_update method that handles entity extraction
        tracing::info!("Calling session.add_incremental_update...");
        let update_id = update.id;
        match session.add_incremental_update(update).await {
            Ok(_) => {
                tracing::info!(
                    "add_context_update_to_session: Successfully added update {}",
                    update_id
                );
                Ok(update_id.to_string())
            }
            Err(e) => {
                tracing::error!(
                    "add_context_update_to_session: Failed to add incremental update: {e}"
                );
                Err(format!("Failed to add update: {e}"))
            }
        }
    }

    fn get_session_context_summary(session: &ActiveSession) -> String {
        let mut summary = Vec::new();

        // Recent updates from hot context
        for update in session.hot_context.iter().rev().take(10) {
            summary.push(format!("- {}", update.content.description));
        }

        if summary.is_empty() {
            "No context available".to_string()
        } else {
            summary.join("\n")
        }
    }

    // Embeddings and Semantic Search Methods

    /// Lazy-initialize content vectorizer on first use
    #[cfg(feature = "embeddings")]
    async fn ensure_vectorizer_initialized(&self) -> Result<Arc<ContentVectorizer>, String> {
        self.content_vectorizer
            .get_or_try_init(|| async {
                info!("Lazy-initializing content vectorizer...");

                let embedding_config = EmbeddingConfig {
                    model_type: self.embedding_config_holder.model_type.clone(),
                    max_batch_size: 32,
                    ..Default::default()
                };

                let vector_db_config = VectorDbConfig {
                    dimension: self.embedding_config_holder.vector_dimension,
                    max_vectors: self.embedding_config_holder.max_vectors_per_session,
                    ..Default::default()
                };

                let vectorizer_config = ContentVectorizerConfig {
                    embedding_config,
                    vector_db_config,
                    enable_cross_session_search: self
                        .embedding_config_holder
                        .cross_session_search_enabled,
                    ..Default::default()
                };

                ContentVectorizer::new(vectorizer_config)
                    .await
                    .map(Arc::new)
                    .map_err(|e| format!("Failed to initialize content vectorizer: {}", e))
            })
            .await
            .map(|arc| Arc::clone(arc))
    }

    /// Vectorize a session's content (requires embeddings feature)
    #[cfg(feature = "embeddings")]
    pub async fn vectorize_session(&self, session_id: Uuid) -> Result<usize, String> {
        let _timer = self.performance_monitor.start_timer("vectorize_session");

        // Lazy-initialize vectorizer if needed
        let vectorizer = self.ensure_vectorizer_initialized().await?;

        // Load session
        let session_result = self.get_session_internal(session_id).await?;
        let session = session_result.load();

        // Vectorize content
        match vectorizer.vectorize_session(&session).await {
            Ok(count) => {
                info!("Vectorized {} items for session {}", count, session_id);
                Ok(count)
            }
            Err(e) => {
                warn!("Failed to vectorize session {session_id}: {e}");
                Err(format!("Vectorization failed: {e}"))
            }
        }
    }

    /// Auto-vectorize only the latest update (incremental vectorization)
    /// This is much more efficient than re-vectorizing the entire session
    #[cfg(feature = "embeddings")]
    pub async fn auto_vectorize_if_enabled(&self, session_id: Uuid) -> Result<(), String> {
        if self.config.enable_embeddings && self.config.auto_vectorize_on_update {
            // Lazy-initialize vectorizer if needed
            match self.ensure_vectorizer_initialized().await {
                Ok(vectorizer) => {
                    // Load session
                    match self.get_session_internal(session_id).await {
                        Ok(session_arc) => {
                            let session = session_arc.load();

                            // Vectorize only the latest update (incremental)
                            match vectorizer.vectorize_latest_update(&session).await {
                                Ok(count) => {
                                    info!(
                                        "Incrementally vectorized {} update(s) for session {}",
                                        count, session_id
                                    );

                                    // Clear query cache to invalidate stale search results
                                    if let Err(e) = vectorizer.clear_query_cache().await {
                                        warn!(
                                            "Failed to clear query cache after vectorization: {}",
                                            e
                                        );
                                    } else {
                                        debug!(
                                            "Query cache cleared after incremental vectorization"
                                        );
                                    }
                                }
                                Err(e) => {
                                    // Don't fail the main operation if auto-vectorization fails
                                    warn!(
                                        "Incremental vectorization failed for session {}: {}",
                                        session_id, e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            warn!(
                                "Failed to load session {} for vectorization: {}",
                                session_id, e
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to initialize vectorizer: {}", e);
                }
            }
        }
        Ok(())
    }

    /// Perform semantic search across all sessions
    #[cfg(feature = "embeddings")]
    pub async fn semantic_search_global(
        &self,
        query: &str,
        limit: Option<usize>,
        date_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    ) -> Result<Vec<crate::core::content_vectorizer::SemanticSearchResult>, String> {
        let _timer = self
            .performance_monitor
            .start_timer("semantic_search_global");

        // Lazy-initialize vectorizer if needed
        let vectorizer = self.ensure_vectorizer_initialized().await?;

        match vectorizer
            .semantic_search(query, limit.unwrap_or(20), None, date_range)
            .await
        {
            Ok(results) => Ok(results),
            Err(e) => Err(format!("Semantic search failed: {e}")),
        }
    }

    /// Perform semantic search within a specific session
    #[cfg(feature = "embeddings")]
    pub async fn semantic_search_session(
        &self,
        session_id: Uuid,
        query: &str,
        limit: Option<usize>,
        date_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    ) -> Result<Vec<crate::core::content_vectorizer::SemanticSearchResult>, String> {
        let _timer = self
            .performance_monitor
            .start_timer("semantic_search_session");

        // Lazy-initialize vectorizer if needed
        let vectorizer = self.ensure_vectorizer_initialized().await?;

        // Auto-load session if not already loaded (get_session_internal handles this)
        // This ensures the session is in memory before vectorization
        let _session = self.get_session_internal(session_id).await?;

        // Auto-vectorize if session hasn't been vectorized yet
        if !vectorizer.is_session_vectorized(session_id) {
            info!(
                "Session {} not vectorized, auto-vectorizing before search",
                session_id
            );
            if let Err(e) = self.vectorize_session(session_id).await {
                warn!(
                    "Auto-vectorization failed for session {}: {}",
                    session_id, e
                );
                // Continue anyway - search might still work with partial data
            }
        }

        match vectorizer
            .semantic_search(query, limit.unwrap_or(20), Some(session_id), date_range)
            .await
        {
            Ok(results) => Ok(results),
            Err(e) => Err(format!("Session semantic search failed: {e}")),
        }
    }

    /// Find related content across sessions
    #[cfg(feature = "embeddings")]
    pub async fn find_related_content(
        &self,
        session_id: Uuid,
        topic: &str,
        limit: Option<usize>,
    ) -> Result<Vec<crate::core::content_vectorizer::SemanticSearchResult>, String> {
        let _timer = self.performance_monitor.start_timer("find_related_content");

        // Lazy-initialize vectorizer if needed
        let vectorizer = self.ensure_vectorizer_initialized().await?;

        // Auto-load session if not already loaded
        let session_result = self.get_session_internal(session_id).await?;
        let session = session_result.load();

        // Auto-vectorize if session hasn't been vectorized yet
        if !vectorizer.is_session_vectorized(session_id) {
            info!(
                "Session {} not vectorized, auto-vectorizing before related content search",
                session_id
            );
            if let Err(e) = self.vectorize_session(session_id).await {
                warn!(
                    "Auto-vectorization failed for session {}: {}",
                    session_id, e
                );
                // Continue anyway - search might still work with partial data
            }
        }

        match vectorizer
            .find_related_content(&session, topic, limit.unwrap_or(10))
            .await
        {
            Ok(results) => Ok(results),
            Err(e) => Err(format!("Related content search failed: {e}")),
        }
    }

    /// Get vectorization statistics
    #[cfg(feature = "embeddings")]
    pub fn get_vectorization_stats(
        &self,
    ) -> Result<std::collections::HashMap<String, usize>, String> {
        // Check if vectorizer has been initialized
        if let Some(vectorizer) = self.content_vectorizer.get() {
            Ok(vectorizer.get_vectorization_stats())
        } else {
            Err("Embeddings not initialized yet (call any vectorization method first)".to_string())
        }
    }

    /// Check if embeddings are enabled and initialized
    pub fn embeddings_enabled(&self) -> bool {
        self.config.enable_embeddings && cfg!(feature = "embeddings") && {
            #[cfg(feature = "embeddings")]
            {
                self.content_vectorizer.get().is_some()
            }
            #[cfg(not(feature = "embeddings"))]
            {
                false
            }
        }
    }

    /// Enable embeddings at runtime (requires restart to initialize components)
    pub async fn enable_embeddings_config(&mut self) -> Result<(), String> {
        if !cfg!(feature = "embeddings") {
            return Err("Embeddings feature not compiled in".to_string());
        }

        self.config.enable_embeddings = true;
        Ok(())
    }

    /// Configure embedding model type
    pub async fn set_embedding_model(&mut self, model_type: String) -> Result<(), String> {
        self.config.embeddings_model_type = model_type;
        Ok(())
    }

    /// Clear query cache to prevent stale vector IDs after restart
    ///
    /// This should be called on daemon startup to ensure cached query results
    /// don't reference vector IDs from before the restart, which would cause
    /// incorrect similarity calculations.
    pub async fn clear_query_cache(&self) -> Result<(), String> {
        #[cfg(feature = "embeddings")]
        {
            if let Some(vectorizer) = self.content_vectorizer.get() {
                vectorizer
                    .clear_query_cache()
                    .await
                    .map_err(|e| format!("Failed to clear query cache: {}", e))?;
                info!("Query cache cleared successfully");
            }
        }
        Ok(())
    }
}

/// Wrapper to make ArcSwap compatible with RwLock API for MCP compatibility
pub struct RwLockCompatWrapper<T> {
    inner: Arc<ArcSwap<T>>,
}

impl<T> RwLockCompatWrapper<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(ArcSwap::new(Arc::new(value))),
        }
    }

    pub fn from_arc_swap(arc_swap: Arc<ArcSwap<T>>) -> Self {
        Self { inner: arc_swap }
    }

    /// Read access - returns a guard that derefs to T
    pub async fn read(&self) -> RwLockReadGuard<T> {
        RwLockReadGuard {
            inner: Arc::clone(&*self.inner.load()),
        }
    }

    /// Write access - returns a guard for atomic updates
    pub async fn write(&self) -> RwLockWriteGuard<T>
    where
        T: Clone,
    {
        RwLockWriteGuard {
            arc_swap: Arc::clone(&self.inner),
            current: Arc::clone(&*self.inner.load()),
        }
    }
}

/// Read guard that provides deref access to the value
pub struct RwLockReadGuard<T> {
    inner: Arc<T>,
}

impl<T> std::ops::Deref for RwLockReadGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Write guard that enables atomic updates
pub struct RwLockWriteGuard<T>
where
    T: Clone,
{
    arc_swap: Arc<ArcSwap<T>>,
    current: Arc<T>,
}

impl<T> std::ops::Deref for RwLockWriteGuard<T>
where
    T: Clone,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.current
    }
}

// Note: DerefMut is intentionally NOT implemented for RwLockWriteGuard
// to prevent unsafe mutable access patterns. Use the update() method instead.

impl<T> RwLockWriteGuard<T>
where
    T: Clone,
{
    /// Update the value atomically
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T),
    {
        let mut new_value = (*self.current).clone();
        f(&mut new_value);
        self.arc_swap.store(Arc::new(new_value));
    }

    /// Replace the entire value atomically
    pub fn replace(&self, new_value: T) {
        self.arc_swap.store(Arc::new(new_value));
    }
}

/// Update get_session to return RwLock-compatible wrapper
/// Comprehensive API compatibility layer for MCP tools
impl LockFreeConversationMemorySystem {
    /// Storage compatibility - returns a wrapper that behaves like Arc<RwLock<Storage>>
    pub fn storage(&self) -> StorageCompatibilityWrapper<'_> {
        StorageCompatibilityWrapper {
            actor: &self.storage_actor,
        }
    }

    /// Get session with RwLock-like API for backwards compatibility
    pub async fn get_session_compat(
        &self,
        session_id: Uuid,
    ) -> Result<SessionCompatibilityWrapper, anyhow::Error> {
        let session_arc = self
            .get_session_internal(session_id)
            .await
            .map_err(anyhow::Error::msg)?;

        Ok(SessionCompatibilityWrapper::new(session_arc))
    }

    async fn get_session_internal(
        &self,
        session_id: Uuid,
    ) -> Result<Arc<ArcSwap<ActiveSession>>, String> {
        if let Some(session_arc) = self.session_manager.sessions.get(&session_id) {
            // Update access time
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("System time before UNIX epoch")
                .as_secs();
            if let Some(access_time) = self.session_manager.active_sessions.get(&session_id) {
                access_time.store(now, Ordering::Relaxed);
            }
            return Ok(session_arc);
        }

        // Try loading from storage via actor
        match self.storage_actor.load_session(session_id).await {
            Ok(Some(session)) => {
                let session_arc = Arc::new(ArcSwap::new(Arc::new(session)));
                self.session_manager
                    .sessions
                    .put(session_id, Arc::clone(&session_arc));

                // Add to active sessions
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("System time before UNIX epoch")
                    .as_secs();
                self.session_manager
                    .active_sessions
                    .insert(session_id, Arc::new(AtomicU64::new(now)));

                Ok(session_arc)
            }
            Ok(None) => Err(format!("Session {session_id} not found")),
            Err(e) => Err(format!("Storage error: {e}")),
        }
    }
}

/// Storage compatibility wrapper
pub struct StorageCompatibilityWrapper<'a> {
    actor: &'a StorageActorHandle,
}

impl<'a> StorageCompatibilityWrapper<'a> {
    pub async fn read(&self) -> &StorageActorHandle {
        self.actor
    }

    pub async fn write(&self) -> &StorageActorHandle {
        self.actor
    }

    // Workspace storage proxy methods
    pub async fn save_workspace_metadata(
        &self,
        workspace_id: Uuid,
        name: &str,
        description: &str,
        session_ids: &Vec<Uuid>,
    ) -> Result<(), String> {
        self.actor
            .save_workspace_metadata(workspace_id, name, description, session_ids)
            .await
    }

    pub async fn delete_workspace(&self, workspace_id: Uuid) -> Result<(), String> {
        self.actor.delete_workspace(workspace_id).await
    }

    pub async fn add_session_to_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
        role: crate::workspace::SessionRole,
    ) -> Result<(), String> {
        self.actor
            .add_session_to_workspace(workspace_id, session_id, role)
            .await
    }

    pub async fn remove_session_from_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
    ) -> Result<(), String> {
        self.actor
            .remove_session_from_workspace(workspace_id, session_id)
            .await
    }
}

/// Session compatibility wrapper that mimics Arc<RwLock<ActiveSession>>
pub struct SessionCompatibilityWrapper {
    inner: Arc<ArcSwap<ActiveSession>>,
}

impl SessionCompatibilityWrapper {
    pub fn new(inner: Arc<ArcSwap<ActiveSession>>) -> Self {
        Self { inner }
    }

    pub async fn read(&self) -> SessionReadGuard {
        SessionReadGuard {
            inner: Arc::clone(&*self.inner.load()),
        }
    }

    pub async fn write(&self) -> SessionWriteGuard {
        SessionWriteGuard {
            arc_swap: Arc::clone(&self.inner),
            current: Arc::clone(&*self.inner.load()),
        }
    }
}

/// Session read guard
pub struct SessionReadGuard {
    inner: Arc<ActiveSession>,
}

impl std::ops::Deref for SessionReadGuard {
    type Target = ActiveSession;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Session write guard that enables atomic updates
pub struct SessionWriteGuard {
    arc_swap: Arc<ArcSwap<ActiveSession>>,
    current: Arc<ActiveSession>,
}

impl std::ops::Deref for SessionWriteGuard {
    type Target = ActiveSession;

    fn deref(&self) -> &Self::Target {
        &self.current
    }
}

impl SessionWriteGuard {
    /// Update the session atomically - required for modifications
    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut ActiveSession),
    {
        let mut new_session = (*self.current).clone();
        f(&mut new_session);
        self.arc_swap.store(Arc::new(new_session));
    }
}

/// Convenience macro for timing lock-free operations
#[macro_export]
macro_rules! time_lockfree_memory_operation {
    ($system:expr, $operation:expr, $code:block) => {{
        let _timer = $system.performance_monitor.start_timer($operation);
        $code
    }};
}
