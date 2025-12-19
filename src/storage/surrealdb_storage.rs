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

//! SurrealDB storage backend for post-cortex.
//!
//! This module provides a SurrealDB-based storage implementation with:
//! - Native graph support for entity relationships
//! - HNSW vector indexing for embeddings
//! - Efficient session and workspace management
//!
//! Enable with: `cargo build --features surrealdb-storage`

use crate::core::context_update::{
    ContextUpdate, EntityData, EntityRelationship, EntityType, RelationType,
};
use crate::core::lockfree_vector_db::{SearchMatch, VectorMetadata};
use crate::core::structured_context::StructuredContext;
use crate::graph::entity_graph::{EntityNetwork, SimpleEntityGraph};
use crate::session::active_session::{ActiveSession, CompressedUpdate, StructuredSummary, UserPreferences, ChangeRecord, CodeReference};
use crate::storage::rocksdb_storage::{SessionCheckpoint, StoredWorkspace};
use crate::storage::traits::{GraphStorage, Storage, VectorStorage};
use crate::workspace::SessionRole;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use surrealdb::engine::any::{Any, connect};
use surrealdb::opt::auth::Root;
use surrealdb::Surreal;
use surrealdb::types::SurrealValue;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Embedding dimension (must match the embedding model)
const EMBEDDING_DIMENSION: usize = 384;

/// SurrealDB storage implementation supporting both local (RocksDB) and remote (WebSocket) backends
#[derive(Clone)]
pub struct SurrealDBStorage {
    db: Arc<Surreal<Any>>,
    #[allow(dead_code)]
    namespace: String,
    #[allow(dead_code)]
    database: String,
}

// ============================================================================
// SurrealDB Record Types
// ============================================================================

/// Session record for SurrealDB - hybrid storage (scalars native, complex as JSON)
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct SessionRecord {
    session_id: String,
    name: Option<String>,
    description: Option<String>,
    created_at: String,
    last_updated: String,
    // User preferences as JSON (queryable object)
    user_preferences: JsonValue,
    // Tiered context as JSON arrays (queryable)
    hot_context: JsonValue,
    warm_context: JsonValue,
    cold_context: JsonValue,
    // Structured context as JSON (queryable object)
    current_state: JsonValue,
    // Code integration as JSON
    code_references: JsonValue,
    change_history: JsonValue,
    // Configuration (native scalars)
    max_extracted_entities: u32,
    max_referenced_entities: u32,
    enable_smart_entity_ranking: bool,
    // Metrics (native scalars)
    total_entity_truncations: u32,
    total_entities_truncated: u32,
    // Vectorization tracking
    vectorized_update_ids: Vec<String>,
}

/// Context update record for SurrealDB - hybrid storage
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct ContextUpdateRecord {
    update_id: String,
    session_id: String,
    timestamp: String,
    update_type: String,
    // Full update as JSON (queryable object!)
    update_data: JsonValue,
}

/// Entity record for SurrealDB graph nodes
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct EntityRecord {
    session_id: String,
    name: String,
    entity_type: String,
    first_mentioned: String,
    last_mentioned: String,
    mention_count: u32,
    importance_score: f32,
    description: Option<String>,
}

/// Embedding record for vector storage
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct EmbeddingRecord {
    content_id: String,
    session_id: String,
    vector: Vec<f32>,
    text: String,
    content_type: String,
    timestamp: String,
    metadata: HashMap<String, String>,
}

/// Workspace record for SurrealDB
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct WorkspaceRecord {
    workspace_id: String,
    name: String,
    description: String,
    created_at: u64,
}

/// Workspace-Session association record
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct WorkspaceSessionRecord {
    workspace_id: String,
    session_id: String,
    role: String,
    added_at: u64,
}

/// Checkpoint record for SurrealDB - hybrid storage
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct CheckpointRecord {
    checkpoint_id: String,
    session_id: String,
    created_at: String,
    // Complete context snapshot as JSON (queryable)
    structured_context: JsonValue,
    recent_updates: JsonValue,
    code_references: JsonValue,
    change_history: JsonValue,
    // Metadata (native scalars)
    total_updates: u32,
    context_quality_score: f32,
    compression_ratio: f32,
}

impl SurrealDBStorage {
    /// Create a new SurrealDB storage instance with remote WebSocket connection
    ///
    /// # Arguments
    /// * `endpoint` - WebSocket endpoint (e.g., "localhost:8000" or "ws://localhost:8000")
    /// * `username` - Database username (optional, for authentication)
    /// * `password` - Database password (optional, for authentication)
    ///
    /// # Example
    /// ```bash
    /// docker run -d -p 8000:8000 surrealdb/surrealdb:latest start --user root --pass root
    /// ```
    /// ```rust
    /// let storage = SurrealDBStorage::new("localhost:8000", Some("root"), Some("root")).await?;
    /// ```
    pub async fn new(
        endpoint: &str,
        username: Option<&str>,
        password: Option<&str>,
    ) -> Result<Self> {
        // Normalize endpoint (add ws:// if not present and not using other engines)
        let endpoint = if endpoint.contains("://") {
            endpoint.to_string()
        } else {
            format!("ws://{}", endpoint)
        };

        info!(
            "SurrealDBStorage: Connecting to remote SurrealDB at {}",
            endpoint
        );

        // Connect to SurrealDB via WebSocket
        let db = connect(&endpoint).await?;

        // Authenticate if credentials provided
        if let (Some(user), Some(pass)) = (username, password) {
            db.signin(Root {
                username: user.to_string(),
                password: pass.to_string(),
            })
            .await?;
            info!("SurrealDBStorage: Authenticated as {}", user);
        }

        let namespace = "post_cortex".to_string();
        let database = "main".to_string();

        // Select namespace and database
        db.use_ns(&namespace).use_db(&database).await?;

        let storage = Self {
            db: Arc::new(db),
            namespace,
            database,
        };

        // Initialize schema
        storage.initialize_schema().await?;

        info!("SurrealDBStorage: Remote instance initialized successfully");

        Ok(storage)
    }

    /// Select all records from a table
    async fn select_all<T: for<'de> Deserialize<'de> + SurrealValue>(&self, table: &str) -> surrealdb::Result<Vec<T>> {
        self.db.select(table).await
    }

    /// Select a single record by ID (table, id)
    async fn select_one<T: for<'de> Deserialize<'de> + SurrealValue>(&self, table: &str, id: &str) -> surrealdb::Result<Option<T>> {
        self.db.select((table, id)).await
    }

    /// Delete a record and return it (table, id)
    async fn delete<T: for<'de> Deserialize<'de> + SurrealValue>(&self, table: &str, id: &str) -> surrealdb::Result<Option<T>> {
        self.db.delete((table, id)).await
    }

    /// Initialize the database schema
    async fn initialize_schema(&self) -> Result<()> {
        info!("SurrealDBStorage: Initializing schema...");

        // First, remove old binary 'data' fields if they exist (schema migration)
        let cleanup = r#"
            -- Remove old binary data fields from previous schema versions
            REMOVE FIELD IF EXISTS data ON session;
            REMOVE FIELD IF EXISTS data ON context_update;
            REMOVE FIELD IF EXISTS data ON checkpoint;
        "#;

        // Ignore cleanup errors (fields may not exist)
        let _ = self.db.query(cleanup).await;

        // Define tables with schema - using SCHEMALESS for complex nested JSON storage
        // This allows storing arbitrary JSON structures while still being queryable
        let schema = r#"
            -- Sessions table (SCHEMALESS to allow complex nested JSON)
            DEFINE TABLE IF NOT EXISTS session SCHEMALESS;
            DEFINE INDEX IF NOT EXISTS idx_session_id ON session FIELDS session_id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_session_created ON session FIELDS created_at;

            -- Context updates table (SCHEMALESS for complex nested JSON)
            DEFINE TABLE IF NOT EXISTS context_update SCHEMALESS;
            DEFINE INDEX IF NOT EXISTS idx_update_id ON context_update FIELDS update_id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_update_session ON context_update FIELDS session_id;
            DEFINE INDEX IF NOT EXISTS idx_update_timestamp ON context_update FIELDS timestamp;

            -- Entities table (graph nodes)
            DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS session_id ON entity TYPE string;
            DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
            DEFINE FIELD IF NOT EXISTS entity_type ON entity TYPE string;
            DEFINE FIELD IF NOT EXISTS first_mentioned ON entity TYPE string;
            DEFINE FIELD IF NOT EXISTS last_mentioned ON entity TYPE string;
            DEFINE FIELD IF NOT EXISTS mention_count ON entity TYPE int;
            DEFINE FIELD IF NOT EXISTS importance_score ON entity TYPE float;
            DEFINE FIELD IF NOT EXISTS description ON entity TYPE option<string>;
            DEFINE INDEX IF NOT EXISTS idx_entity_session_name ON entity FIELDS session_id, name UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_entity_session ON entity FIELDS session_id;
            DEFINE INDEX IF NOT EXISTS idx_entity_type ON entity FIELDS entity_type;
            DEFINE INDEX IF NOT EXISTS idx_entity_importance ON entity FIELDS importance_score;

            -- Relation tables for graph edges (schemaless for flexibility)
            DEFINE TABLE IF NOT EXISTS required_by SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS leads_to SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS related_to SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS conflicts_with SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS depends_on SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS implements SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS caused_by SCHEMALESS;
            DEFINE TABLE IF NOT EXISTS solves SCHEMALESS;

            -- Embeddings table with vector support
            DEFINE TABLE IF NOT EXISTS embedding SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS content_id ON embedding TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON embedding TYPE string;
            DEFINE FIELD IF NOT EXISTS vector ON embedding TYPE array<float>;
            DEFINE FIELD IF NOT EXISTS text ON embedding TYPE string;
            DEFINE FIELD IF NOT EXISTS content_type ON embedding TYPE string;
            DEFINE FIELD IF NOT EXISTS timestamp ON embedding TYPE string;
            DEFINE FIELD IF NOT EXISTS metadata ON embedding TYPE object;
            DEFINE INDEX IF NOT EXISTS idx_embedding_content ON embedding FIELDS content_id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_embedding_session ON embedding FIELDS session_id;
            DEFINE INDEX IF NOT EXISTS idx_embedding_type ON embedding FIELDS content_type;

            -- Workspaces table
            DEFINE TABLE IF NOT EXISTS workspace SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS workspace_id ON workspace TYPE string;
            DEFINE FIELD IF NOT EXISTS name ON workspace TYPE string;
            DEFINE FIELD IF NOT EXISTS description ON workspace TYPE string;
            DEFINE FIELD IF NOT EXISTS created_at ON workspace TYPE int;
            DEFINE INDEX IF NOT EXISTS idx_workspace_id ON workspace FIELDS workspace_id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_workspace_name ON workspace FIELDS name;

            -- Workspace-Session associations
            DEFINE TABLE IF NOT EXISTS workspace_session SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS workspace_id ON workspace_session TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON workspace_session TYPE string;
            DEFINE FIELD IF NOT EXISTS role ON workspace_session TYPE string;
            DEFINE FIELD IF NOT EXISTS added_at ON workspace_session TYPE int;
            DEFINE INDEX IF NOT EXISTS idx_ws_workspace ON workspace_session FIELDS workspace_id;
            DEFINE INDEX IF NOT EXISTS idx_ws_session ON workspace_session FIELDS session_id;
            DEFINE INDEX IF NOT EXISTS idx_ws_unique ON workspace_session FIELDS workspace_id, session_id UNIQUE;

            -- Checkpoints table (SCHEMALESS for complex nested JSON)
            DEFINE TABLE IF NOT EXISTS checkpoint SCHEMALESS;
            DEFINE INDEX IF NOT EXISTS idx_checkpoint_id ON checkpoint FIELDS checkpoint_id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_checkpoint_session ON checkpoint FIELDS session_id;
        "#;

        self.db.query(schema).await?;

        info!("SurrealDBStorage: Schema initialized successfully");

        Ok(())
    }

    /// Get the relation table name for a relation type
    fn relation_table_name(relation_type: &RelationType) -> &'static str {
        match relation_type {
            RelationType::RequiredBy => "required_by",
            RelationType::LeadsTo => "leads_to",
            RelationType::RelatedTo => "related_to",
            RelationType::ConflictsWith => "conflicts_with",
            RelationType::DependsOn => "depends_on",
            RelationType::Implements => "implements",
            RelationType::CausedBy => "caused_by",
            RelationType::Solves => "solves",
        }
    }

    /// Create entity ID from session and name
    fn entity_id(session_id: Uuid, name: &str) -> String {
        format!("{}_{}", session_id, name.replace(' ', "_").to_lowercase())
    }

    /// Parse RFC3339 datetime string, fallback to current time on error
    fn parse_datetime(s: &str) -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339(s)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now())
    }

    /// Parse relation type from string
    #[allow(dead_code)]
    fn parse_relation_type(s: &str) -> RelationType {
        match s {
            "required_by" => RelationType::RequiredBy,
            "leads_to" => RelationType::LeadsTo,
            "related_to" => RelationType::RelatedTo,
            "conflicts_with" => RelationType::ConflictsWith,
            "depends_on" => RelationType::DependsOn,
            "implements" => RelationType::Implements,
            "caused_by" => RelationType::CausedBy,
            "solves" => RelationType::Solves,
            _ => RelationType::RelatedTo,
        }
    }

    /// Parse entity type from string
    fn parse_entity_type(s: &str) -> EntityType {
        match s.to_lowercase().as_str() {
            "technology" => EntityType::Technology,
            "concept" => EntityType::Concept,
            "problem" => EntityType::Problem,
            "solution" => EntityType::Solution,
            "decision" => EntityType::Decision,
            "codecomponent" | "code_component" => EntityType::CodeComponent,
            _ => EntityType::Concept,
        }
    }
}

// ============================================================================
// Storage Trait Implementation
// ============================================================================

#[async_trait]
impl Storage for SurrealDBStorage {
    async fn save_session(&self, session: &ActiveSession) -> Result<()> {
        debug!(
            "SurrealDBStorage: Saving session with ID: {}",
            session.id()
        );

        // Convert complex types to JSON for queryable storage
        let hot_context_vec: Vec<ContextUpdate> = session.hot_context.iter();

        let record = SessionRecord {
            session_id: session.id().to_string(),
            name: session.name().clone(),
            description: session.description().clone(),
            created_at: session.created_at().to_rfc3339(),
            last_updated: Utc::now().to_rfc3339(),
            user_preferences: serde_json::to_value(session.user_preferences())?,
            hot_context: serde_json::to_value(&hot_context_vec)?,
            warm_context: serde_json::to_value(session.warm_context.as_ref())?,
            cold_context: serde_json::to_value(session.cold_context.as_ref())?,
            current_state: serde_json::to_value(session.current_state.as_ref())?,
            code_references: serde_json::to_value(session.code_references.as_ref())?,
            change_history: serde_json::to_value(session.change_history.as_ref())?,
            max_extracted_entities: session.max_extracted_entities as u32,
            max_referenced_entities: session.max_referenced_entities as u32,
            enable_smart_entity_ranking: session.enable_smart_entity_ranking,
            total_entity_truncations: session.total_entity_truncations as u32,
            total_entities_truncated: session.total_entities_truncated as u32,
            vectorized_update_ids: session.vectorized_update_ids
                .iter()
                .map(|id| id.to_string())
                .collect(),
        };

        // Upsert session record
        let _: Option<SessionRecord> = self
            .db
            .upsert(("session", session.id().to_string()))
            .content(record)
            .await?;

        // Save entities to native graph table
        for entity in session.entity_graph.get_all_entities() {
            if let Err(e) = self.upsert_entity(session.id(), &entity).await {
                warn!("Failed to save entity '{}': {}", entity.name, e);
            }
        }

        // Save relationships via native RELATE
        for rel in session.entity_graph.get_all_relationships() {
            if let Err(e) = self.create_relationship(session.id(), &rel).await {
                warn!("Failed to save relationship '{}' -> '{}': {}", rel.from_entity, rel.to_entity, e);
            }
        }

        debug!("SurrealDBStorage: Session saved successfully with native graph");

        Ok(())
    }

    async fn load_session(&self, session_id: Uuid) -> Result<ActiveSession> {
        debug!(
            "SurrealDBStorage: Loading session with ID: {}",
            session_id
        );

        // Load session record
        let record: Option<SessionRecord> = self
            .db
            .select(("session", session_id.to_string()))
            .await?;

        let r = record.ok_or_else(|| anyhow::anyhow!("Session not found"))?;

        // Load entities from native table and build graph
        let entities = self.list_entities(session_id).await.unwrap_or_default();
        let mut entity_graph = SimpleEntityGraph::new();
        for entity in &entities {
            entity_graph.add_or_update_entity(
                entity.name.clone(),
                entity.entity_type.clone(),
                entity.last_mentioned,
                entity.description.as_deref().unwrap_or(""),
            );
        }

        // Load relationships and add to graph
        for entity in &entities {
            let related = self.find_related_entities(session_id, &entity.name).await.unwrap_or_default();
            for related_name in related {
                entity_graph.add_relationship(EntityRelationship {
                    from_entity: entity.name.clone(),
                    to_entity: related_name,
                    relation_type: RelationType::RelatedTo,
                    context: String::new(),
                });
            }
        }

        // Load incremental updates from native table
        let incremental_updates = self.load_session_updates(session_id).await.unwrap_or_default();

        // Parse JSON fields back to types
        let user_preferences: UserPreferences = serde_json::from_value(r.user_preferences)?;
        let hot_context: Vec<ContextUpdate> = serde_json::from_value(r.hot_context)?;
        let warm_context: Vec<CompressedUpdate> = serde_json::from_value(r.warm_context)?;
        let cold_context: Vec<StructuredSummary> = serde_json::from_value(r.cold_context)?;
        let current_state: StructuredContext = serde_json::from_value(r.current_state)?;
        let code_references: HashMap<String, Vec<CodeReference>> = serde_json::from_value(r.code_references)?;
        let change_history: Vec<ChangeRecord> = serde_json::from_value(r.change_history)?;

        // Parse vectorized_update_ids from strings to Uuids
        let vectorized_ids: Vec<Uuid> = r.vectorized_update_ids
            .iter()
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect();

        // Reconstruct ActiveSession from components
        let session = ActiveSession::from_components(
            session_id,
            r.name,
            r.description,
            DateTime::parse_from_rfc3339(&r.created_at)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            DateTime::parse_from_rfc3339(&r.last_updated)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            user_preferences,
            hot_context,
            warm_context,
            cold_context,
            current_state,
            incremental_updates,
            code_references,
            change_history,
            entity_graph,
            r.max_extracted_entities as usize,
            r.max_referenced_entities as usize,
            r.enable_smart_entity_ranking,
            r.total_entity_truncations as usize,
            r.total_entities_truncated as usize,
            vectorized_ids,
        );

        debug!("SurrealDBStorage: Session loaded and reconstructed successfully");
        Ok(session)
    }

    async fn delete_session(&self, session_id: Uuid) -> Result<()> {
        debug!(
            "SurrealDBStorage: Deleting session with ID: {}",
            session_id
        );

        // Delete session
        let _: Option<SessionRecord> = self
            .db
            .delete(("session", session_id.to_string()))
            .await?;

        // Delete all context updates for this session
        self.db
            .query("DELETE context_update WHERE session_id = $session_id")
            .bind(("session_id", session_id.to_string()))
            .await?;

        // Delete all entities for this session
        self.db
            .query("DELETE entity WHERE session_id = $session_id")
            .bind(("session_id", session_id.to_string()))
            .await?;

        // Delete all embeddings for this session
        self.db
            .query("DELETE embedding WHERE session_id = $session_id")
            .bind(("session_id", session_id.to_string()))
            .await?;

        // Delete all relations for this session
        for table in [
            "required_by",
            "leads_to",
            "related_to",
            "conflicts_with",
            "depends_on",
            "implements",
            "caused_by",
            "solves",
        ] {
            self.db
                .query(format!("DELETE {} WHERE session_id = $session_id", table))
                .bind(("session_id", session_id.to_string()))
                .await?;
        }

        debug!("SurrealDBStorage: Session deleted successfully");

        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<Uuid>> {
        debug!("SurrealDBStorage: Listing sessions");

        let records: Vec<SessionRecord> = self.select_all("session").await?;

        let sessions: Vec<Uuid> = records
            .into_iter()
            .filter_map(|r| Uuid::parse_str(&r.session_id).ok())
            .collect();

        debug!("SurrealDBStorage: Found {} sessions", sessions.len());

        Ok(sessions)
    }

    async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        let record: Option<SessionRecord> = self
            .db
            .select(("session", session_id.to_string()))
            .await?;

        Ok(record.is_some())
    }

    async fn batch_save_updates(
        &self,
        session_id: Uuid,
        updates: Vec<ContextUpdate>,
    ) -> Result<()> {
        debug!(
            "SurrealDBStorage: Batch saving {} updates for session {}",
            updates.len(),
            session_id
        );

        for update in updates {
            let record = ContextUpdateRecord {
                update_id: update.id.to_string(),
                session_id: session_id.to_string(),
                timestamp: update.timestamp.to_rfc3339(),
                update_type: format!("{:?}", update.update_type),
                update_data: serde_json::to_value(&update)?,
            };

            let _: Option<ContextUpdateRecord> = self
                .db
                .upsert(("context_update", record.update_id.clone()))
                .content(record)
                .await?;
        }

        debug!("SurrealDBStorage: Batch save completed");

        Ok(())
    }

    async fn load_session_updates(&self, session_id: Uuid) -> Result<Vec<ContextUpdate>> {
        debug!(
            "SurrealDBStorage: Loading updates for session {}",
            session_id
        );

        let mut response = self
            .db
            .query("SELECT * FROM context_update WHERE session_id = $session_id ORDER BY timestamp")
            .bind(("session_id", session_id.to_string()))
            .await?;

        let records: Vec<ContextUpdateRecord> = response.take(0)?;

        // Parse update_data JSON back to ContextUpdate
        let updates: Vec<ContextUpdate> = records
            .into_iter()
            .filter_map(|r| serde_json::from_value(r.update_data).ok())
            .collect();

        debug!("SurrealDBStorage: Loaded {} updates", updates.len());

        Ok(updates)
    }

    async fn save_checkpoint(&self, checkpoint: &SessionCheckpoint) -> Result<()> {
        debug!(
            "SurrealDBStorage: Saving checkpoint with ID: {}",
            checkpoint.id
        );

        let record = CheckpointRecord {
            checkpoint_id: checkpoint.id.to_string(),
            session_id: checkpoint.session_id.to_string(),
            created_at: checkpoint.created_at.to_rfc3339(),
            structured_context: serde_json::to_value(&checkpoint.structured_context)?,
            recent_updates: serde_json::to_value(&checkpoint.recent_updates)?,
            code_references: serde_json::to_value(&checkpoint.code_references)?,
            change_history: serde_json::to_value(&checkpoint.change_history)?,
            total_updates: checkpoint.total_updates as u32,
            context_quality_score: checkpoint.context_quality_score,
            compression_ratio: checkpoint.compression_ratio,
        };

        let _: Option<CheckpointRecord> = self
            .db
            .upsert(("checkpoint", checkpoint.id.to_string()))
            .content(record)
            .await?;

        debug!("SurrealDBStorage: Checkpoint saved successfully");

        Ok(())
    }

    async fn load_checkpoint(&self, checkpoint_id: Uuid) -> Result<SessionCheckpoint> {
        debug!(
            "SurrealDBStorage: Loading checkpoint with ID: {}",
            checkpoint_id
        );

        let record: Option<CheckpointRecord> = self
            .db
            .select(("checkpoint", checkpoint_id.to_string()))
            .await?;

        match record {
            Some(r) => {
                let checkpoint = SessionCheckpoint {
                    id: Uuid::parse_str(&r.checkpoint_id)?,
                    session_id: Uuid::parse_str(&r.session_id)?,
                    created_at: DateTime::parse_from_rfc3339(&r.created_at)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    structured_context: serde_json::from_value(r.structured_context)?,
                    recent_updates: serde_json::from_value(r.recent_updates)?,
                    code_references: serde_json::from_value(r.code_references)?,
                    change_history: serde_json::from_value(r.change_history)?,
                    total_updates: r.total_updates as usize,
                    context_quality_score: r.context_quality_score,
                    compression_ratio: r.compression_ratio,
                };
                Ok(checkpoint)
            }
            None => Err(anyhow::anyhow!("Checkpoint not found")),
        }
    }

    async fn list_checkpoints(&self) -> Result<Vec<SessionCheckpoint>> {
        debug!("SurrealDBStorage: Listing checkpoints");

        let records: Vec<CheckpointRecord> = self.select_all("checkpoint").await?;

        let checkpoints: Vec<SessionCheckpoint> = records
            .into_iter()
            .filter_map(|r| {
                Some(SessionCheckpoint {
                    id: Uuid::parse_str(&r.checkpoint_id).ok()?,
                    session_id: Uuid::parse_str(&r.session_id).ok()?,
                    created_at: DateTime::parse_from_rfc3339(&r.created_at)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    structured_context: serde_json::from_value(r.structured_context).ok()?,
                    recent_updates: serde_json::from_value(r.recent_updates).ok()?,
                    code_references: serde_json::from_value(r.code_references).ok()?,
                    change_history: serde_json::from_value(r.change_history).ok()?,
                    total_updates: r.total_updates as usize,
                    context_quality_score: r.context_quality_score,
                    compression_ratio: r.compression_ratio,
                })
            })
            .collect();

        debug!("SurrealDBStorage: Listed {} checkpoints", checkpoints.len());

        Ok(checkpoints)
    }

    async fn save_workspace_metadata(
        &self,
        workspace_id: Uuid,
        name: &str,
        description: &str,
        session_ids: &[Uuid],
    ) -> Result<()> {
        debug!(
            "SurrealDBStorage: Saving workspace {} ({})",
            name, workspace_id
        );

        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let record = WorkspaceRecord {
            workspace_id: workspace_id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            created_at,
        };

        let _: Option<WorkspaceRecord> = self
            .db
            .upsert(("workspace", workspace_id.to_string()))
            .content(record)
            .await?;

        // Add session associations
        for session_id in session_ids {
            self.add_session_to_workspace(workspace_id, *session_id, SessionRole::Primary)
                .await?;
        }

        debug!("SurrealDBStorage: Workspace saved successfully");

        Ok(())
    }

    async fn delete_workspace(&self, workspace_id: Uuid) -> Result<()> {
        debug!(
            "SurrealDBStorage: Deleting workspace {}",
            workspace_id
        );

        // Delete workspace
        let _: Option<WorkspaceRecord> = self
            .db
            .delete(("workspace", workspace_id.to_string()))
            .await?;

        // Delete all workspace-session associations
        self.db
            .query("DELETE workspace_session WHERE workspace_id = $workspace_id")
            .bind(("workspace_id", workspace_id.to_string()))
            .await?;

        debug!("SurrealDBStorage: Workspace deleted successfully");

        Ok(())
    }

    async fn list_workspaces(&self) -> Result<Vec<StoredWorkspace>> {
        debug!("SurrealDBStorage: Listing workspaces");

        let records: Vec<WorkspaceRecord> = self.select_all("workspace").await?;

        let mut workspaces = Vec::new();
        for record in records {
            if let Ok(workspace_id) = Uuid::parse_str(&record.workspace_id) {
                // Get sessions for this workspace
                let mut response = self
                    .db
                    .query(
                        "SELECT * FROM workspace_session WHERE workspace_id = $workspace_id",
                    )
                    .bind(("workspace_id", record.workspace_id.clone()))
                    .await?;

                let session_records: Vec<WorkspaceSessionRecord> = response.take(0)?;

                let sessions: Vec<(Uuid, SessionRole)> = session_records
                    .into_iter()
                    .filter_map(|s| {
                        Uuid::parse_str(&s.session_id)
                            .ok()
                            .map(|id| {
                                let role = match s.role.as_str() {
                                    "Primary" => SessionRole::Primary,
                                    "Related" => SessionRole::Related,
                                    "Dependency" => SessionRole::Dependency,
                                    "Shared" => SessionRole::Shared,
                                    _ => SessionRole::Primary,
                                };
                                (id, role)
                            })
                    })
                    .collect();

                workspaces.push(StoredWorkspace {
                    id: workspace_id,
                    name: record.name,
                    description: record.description,
                    sessions,
                    created_at: record.created_at,
                });
            }
        }

        debug!("SurrealDBStorage: Listed {} workspaces", workspaces.len());

        Ok(workspaces)
    }

    async fn add_session_to_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
        role: SessionRole,
    ) -> Result<()> {
        debug!(
            "SurrealDBStorage: Adding session {} to workspace {} with role {:?}",
            session_id, workspace_id, role
        );

        let added_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let record = WorkspaceSessionRecord {
            workspace_id: workspace_id.to_string(),
            session_id: session_id.to_string(),
            role: format!("{:?}", role),
            added_at,
        };

        let key = format!("{}_{}", workspace_id, session_id);
        let _: Option<WorkspaceSessionRecord> = self
            .db
            .upsert(("workspace_session", key))
            .content(record)
            .await?;

        Ok(())
    }

    async fn remove_session_from_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
    ) -> Result<()> {
        debug!(
            "SurrealDBStorage: Removing session {} from workspace {}",
            session_id, workspace_id
        );

        let key = format!("{}_{}", workspace_id, session_id);
        let _: Option<WorkspaceSessionRecord> =
            self.delete("workspace_session", &key).await?;

        Ok(())
    }

    async fn compact(&self) -> Result<()> {
        debug!("SurrealDBStorage: Compact requested (no-op for SurrealDB)");
        // SurrealDB handles compaction internally
        Ok(())
    }

    async fn get_key_count(&self) -> Result<usize> {
        let mut response = self
            .db
            .query(
                r#"
                RETURN (SELECT count() FROM session GROUP ALL).count +
                       (SELECT count() FROM context_update GROUP ALL).count +
                       (SELECT count() FROM entity GROUP ALL).count +
                       (SELECT count() FROM embedding GROUP ALL).count +
                       (SELECT count() FROM workspace GROUP ALL).count +
                       (SELECT count() FROM checkpoint GROUP ALL).count
            "#,
            )
            .await?;

        let count: Option<i64> = response.take(0)?;

        Ok(count.unwrap_or(0) as usize)
    }

    async fn get_stats(&self) -> Result<String> {
        let mut response = self
            .db
            .query(
                r#"
                RETURN {
                    sessions: (SELECT count() FROM session GROUP ALL).count,
                    updates: (SELECT count() FROM context_update GROUP ALL).count,
                    entities: (SELECT count() FROM entity GROUP ALL).count,
                    embeddings: (SELECT count() FROM embedding GROUP ALL).count,
                    workspaces: (SELECT count() FROM workspace GROUP ALL).count,
                    checkpoints: (SELECT count() FROM checkpoint GROUP ALL).count
                }
            "#,
            )
            .await?;

        let stats: Option<serde_json::Value> = response.take(0)?;

        Ok(serde_json::to_string_pretty(&stats.unwrap_or_default())?)
    }
}

// ============================================================================
// GraphStorage Trait Implementation
// ============================================================================

#[async_trait]
impl GraphStorage for SurrealDBStorage {
    async fn upsert_entity(&self, session_id: Uuid, entity: &EntityData) -> Result<()> {
        debug!(
            "SurrealDBStorage: Upserting entity '{}' for session {}",
            entity.name, session_id
        );

        let entity_id = Self::entity_id(session_id, &entity.name);

        let record = EntityRecord {
            session_id: session_id.to_string(),
            name: entity.name.clone(),
            entity_type: format!("{:?}", entity.entity_type),
            first_mentioned: entity.first_mentioned.to_rfc3339(),
            last_mentioned: entity.last_mentioned.to_rfc3339(),
            mention_count: entity.mention_count,
            importance_score: entity.importance_score,
            description: entity.description.clone(),
        };

        // Use query with backticks for IDs containing special chars (UUIDs have dashes)
        let query = format!("UPSERT entity:`{}` CONTENT $content", entity_id);
        self.db
            .query(query)
            .bind(("content", record))
            .await?;

        Ok(())
    }

    async fn get_entity(&self, session_id: Uuid, name: &str) -> Result<Option<EntityData>> {
        let entity_id = Self::entity_id(session_id, name);

        let record: Option<EntityRecord> = self.select_one("entity", &entity_id).await?;

        Ok(record.map(|r| EntityData {
            name: r.name,
            entity_type: Self::parse_entity_type(&r.entity_type),
            first_mentioned: Self::parse_datetime(&r.first_mentioned),
            last_mentioned: Self::parse_datetime(&r.last_mentioned),
            mention_count: r.mention_count,
            importance_score: r.importance_score,
            description: r.description,
        }))
    }

    async fn list_entities(&self, session_id: Uuid) -> Result<Vec<EntityData>> {
        let mut response = self
            .db
            .query("SELECT * FROM entity WHERE session_id = $session_id ORDER BY importance_score DESC")
            .bind(("session_id", session_id.to_string()))
            .await?;

        let records: Vec<EntityRecord> = response.take(0)?;

        Ok(records
            .into_iter()
            .map(|r| EntityData {
                name: r.name,
                entity_type: Self::parse_entity_type(&r.entity_type),
                first_mentioned: Self::parse_datetime(&r.first_mentioned),
                last_mentioned: Self::parse_datetime(&r.last_mentioned),
                mention_count: r.mention_count,
                importance_score: r.importance_score,
                description: r.description,
            })
            .collect())
    }

    async fn delete_entity(&self, session_id: Uuid, name: &str) -> Result<()> {
        let entity_id = Self::entity_id(session_id, name);
        let _: Option<EntityRecord> = self.delete("entity", &entity_id).await?;
        Ok(())
    }

    async fn create_relationship(
        &self,
        session_id: Uuid,
        relationship: &EntityRelationship,
    ) -> Result<()> {
        debug!(
            "SurrealDBStorage: Creating relationship {} -> {} ({:?})",
            relationship.from_entity, relationship.to_entity, relationship.relation_type
        );

        let from_id = Self::entity_id(session_id, &relationship.from_entity);
        let to_id = Self::entity_id(session_id, &relationship.to_entity);
        let table = Self::relation_table_name(&relationship.relation_type);

        // Create relationship using RELATE (backticks needed for IDs with dashes/special chars)
        let query = format!(
            "RELATE entity:`{}`->{}->entity:`{}` SET context = $context, session_id = $session_id",
            from_id, table, to_id
        );

        self.db
            .query(query)
            .bind(("context", relationship.context.clone()))
            .bind(("session_id", session_id.to_string()))
            .await?;

        Ok(())
    }

    async fn find_related_entities(
        &self,
        session_id: Uuid,
        entity_name: &str,
    ) -> Result<Vec<String>> {
        let entity_id = Self::entity_id(session_id, entity_name);

        // Query all outgoing and incoming relations (backticks for IDs with special chars)
        let query = format!(
            r#"
            SELECT array::distinct(array::flatten([
                (SELECT VALUE out.name FROM entity:`{}`->*->entity WHERE session_id = $session_id),
                (SELECT VALUE in.name FROM entity:`{}`<-*<-entity WHERE session_id = $session_id)
            ])) AS related
            "#,
            entity_id, entity_id
        );

        let mut response = self
            .db
            .query(query)
            .bind(("session_id", session_id.to_string()))
            .await?;

        #[derive(Deserialize, SurrealValue)]
        struct RelatedResult {
            related: Vec<String>,
        }

        let results: Option<RelatedResult> = response.take(0)?;

        Ok(results.map(|r| r.related).unwrap_or_default())
    }

    async fn find_related_by_type(
        &self,
        session_id: Uuid,
        entity_name: &str,
        relation_type: &RelationType,
    ) -> Result<Vec<String>> {
        let entity_id = Self::entity_id(session_id, entity_name);
        let table = Self::relation_table_name(relation_type);

        let query = format!(
            r#"
            SELECT array::distinct(array::flatten([
                (SELECT VALUE out.name FROM entity:`{}`->{table}->entity WHERE session_id = $session_id),
                (SELECT VALUE in.name FROM entity:`{}`<-{table}<-entity WHERE session_id = $session_id)
            ])) AS related
            "#,
            entity_id, entity_id
        );

        let mut response = self
            .db
            .query(query)
            .bind(("session_id", session_id.to_string()))
            .await?;

        #[derive(Deserialize, SurrealValue)]
        struct RelatedResult {
            related: Vec<String>,
        }

        let results: Option<RelatedResult> = response.take(0)?;

        Ok(results.map(|r| r.related).unwrap_or_default())
    }

    async fn find_shortest_path(
        &self,
        session_id: Uuid,
        from: &str,
        to: &str,
    ) -> Result<Option<Vec<String>>> {
        let from_id = Self::entity_id(session_id, from);
        let to_id = Self::entity_id(session_id, to);

        // Use SurrealDB's graph traversal (backticks for IDs with special chars)
        let query = format!(
            r#"
            SELECT VALUE array::flatten([
                entity:`{}`.name,
                (SELECT VALUE name FROM entity:`{}`->*..5->entity:`{}` WHERE session_id = $session_id),
                entity:`{}`.name
            ])
            "#,
            from_id, from_id, to_id, to_id
        );

        let mut response = self
            .db
            .query(query)
            .bind(("session_id", session_id.to_string()))
            .await?;

        let path: Option<Vec<String>> = response.take(0)?;

        Ok(path.filter(|p| !p.is_empty()))
    }

    async fn get_entity_network(
        &self,
        session_id: Uuid,
        center: &str,
        max_depth: usize,
    ) -> Result<EntityNetwork> {
        let entity_id = Self::entity_id(session_id, center);

        // Get entities within max_depth hops (backticks for IDs with special chars)
        let entity_query = format!(
            r#"
            SELECT name, entity_type, first_mentioned, last_mentioned,
                   mention_count, importance_score, description
            FROM entity:`{}`<->*..{}->entity WHERE session_id = $session_id
            "#,
            entity_id, max_depth
        );

        let mut response = self
            .db
            .query(entity_query)
            .bind(("session_id", session_id.to_string()))
            .await?;

        let entity_records: Vec<EntityRecord> = response.take(0)?;

        let mut entities: BTreeMap<String, EntityData> = entity_records
            .into_iter()
            .map(|r| {
                (
                    r.name.clone(),
                    EntityData {
                        name: r.name,
                        entity_type: Self::parse_entity_type(&r.entity_type),
                        first_mentioned: Self::parse_datetime(&r.first_mentioned),
                        last_mentioned: Self::parse_datetime(&r.last_mentioned),
                        mention_count: r.mention_count,
                        importance_score: r.importance_score,
                        description: r.description,
                    },
                )
            })
            .collect();

        // Add center entity if not already included
        if let Some(center_entity) = self.get_entity(session_id, center).await? {
            entities.insert(center.to_string(), center_entity);
        }

        // Get relationships (simplified - just get direct connections from center)
        let mut relationships = Vec::new();

        // Query each relation type
        for (table, rel_type) in [
            ("required_by", RelationType::RequiredBy),
            ("leads_to", RelationType::LeadsTo),
            ("related_to", RelationType::RelatedTo),
            ("conflicts_with", RelationType::ConflictsWith),
            ("depends_on", RelationType::DependsOn),
            ("implements", RelationType::Implements),
            ("caused_by", RelationType::CausedBy),
            ("solves", RelationType::Solves),
        ] {
            let rel_query = format!(
                r#"
                SELECT in.name AS from_entity, out.name AS to_entity, context
                FROM {}
                WHERE session_id = $session_id
                "#,
                table
            );

            let mut rel_response = self
                .db
                .query(rel_query)
                .bind(("session_id", session_id.to_string()))
                .await?;

            #[derive(Deserialize, SurrealValue)]
            struct RelRecord {
                from_entity: String,
                to_entity: String,
                context: String,
            }

            let rel_records: Vec<RelRecord> = rel_response.take(0).unwrap_or_default();

            for record in rel_records {
                if entities.contains_key(&record.from_entity)
                    && entities.contains_key(&record.to_entity)
                {
                    relationships.push(EntityRelationship {
                        from_entity: record.from_entity,
                        to_entity: record.to_entity,
                        relation_type: rel_type.clone(),
                        context: record.context,
                    });
                }
            }
        }

        Ok(EntityNetwork {
            center: center.to_string(),
            entities,
            relationships,
        })
    }
}

// ============================================================================
// VectorStorage Trait Implementation
// ============================================================================

#[async_trait]
impl VectorStorage for SurrealDBStorage {
    async fn add_vector(&self, vector: Vec<f32>, metadata: VectorMetadata) -> Result<String> {
        if vector.len() != EMBEDDING_DIMENSION {
            return Err(anyhow::anyhow!(
                "Vector dimension mismatch: expected {}, got {}",
                EMBEDDING_DIMENSION,
                vector.len()
            ));
        }

        debug!(
            "SurrealDBStorage: Adding vector for content {}",
            metadata.id
        );

        let record = EmbeddingRecord {
            content_id: metadata.id.clone(),
            session_id: metadata.source.clone(),
            vector,
            text: metadata.text,
            content_type: metadata.content_type,
            timestamp: metadata.timestamp.to_rfc3339(),
            metadata: metadata.metadata,
        };

        let _: Option<EmbeddingRecord> = self
            .db
            .upsert(("embedding", metadata.id.clone()))
            .content(record)
            .await?;

        Ok(metadata.id)
    }

    async fn add_vectors_batch(
        &self,
        vectors: Vec<(Vec<f32>, VectorMetadata)>,
    ) -> Result<Vec<String>> {
        let mut ids = Vec::new();

        for (vector, metadata) in vectors {
            let id = self.add_vector(vector, metadata).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchMatch>> {
        if query.len() != EMBEDDING_DIMENSION {
            return Err(anyhow::anyhow!(
                "Query vector dimension mismatch: expected {}, got {}",
                EMBEDDING_DIMENSION,
                query.len()
            ));
        }

        debug!("SurrealDBStorage: Searching for {} nearest vectors", k);

        // Compute cosine similarity manually (SurrealDB 2.x doesn't have built-in vector ops)
        // In production, you might want to use an extension or compute similarity differently
        let mut response = self.db.query("SELECT * FROM embedding").await?;

        let records: Vec<EmbeddingRecord> = response.take(0)?;

        let mut matches: Vec<SearchMatch> = records
            .into_iter()
            .map(|r| {
                let similarity = cosine_similarity(query, &r.vector);
                SearchMatch {
                    vector_id: 0, // Not used with SurrealDB
                    similarity,
                    metadata: VectorMetadata {
                        id: r.content_id,
                        text: r.text,
                        source: r.session_id,
                        content_type: r.content_type,
                        timestamp: Self::parse_datetime(&r.timestamp),
                        metadata: r.metadata,
                    },
                }
            })
            .collect();

        // Sort by similarity descending
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

        // Take top k
        matches.truncate(k);

        debug!("SurrealDBStorage: Found {} matches", matches.len());

        Ok(matches)
    }

    async fn search_in_session(
        &self,
        query: &[f32],
        k: usize,
        session_id: &str,
    ) -> Result<Vec<SearchMatch>> {
        if query.len() != EMBEDDING_DIMENSION {
            return Err(anyhow::anyhow!(
                "Query vector dimension mismatch: expected {}, got {}",
                EMBEDDING_DIMENSION,
                query.len()
            ));
        }

        debug!(
            "SurrealDBStorage: Searching for {} nearest vectors in session {}",
            k, session_id
        );

        let mut response = self
            .db
            .query("SELECT * FROM embedding WHERE session_id = $session_id")
            .bind(("session_id", session_id.to_string()))
            .await?;

        let records: Vec<EmbeddingRecord> = response.take(0)?;

        let mut matches: Vec<SearchMatch> = records
            .into_iter()
            .map(|r| {
                let similarity = cosine_similarity(query, &r.vector);
                SearchMatch {
                    vector_id: 0,
                    similarity,
                    metadata: VectorMetadata {
                        id: r.content_id,
                        text: r.text,
                        source: r.session_id,
                        content_type: r.content_type,
                        timestamp: Self::parse_datetime(&r.timestamp),
                        metadata: r.metadata,
                    },
                }
            })
            .collect();

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        matches.truncate(k);

        Ok(matches)
    }

    async fn search_by_content_type(
        &self,
        query: &[f32],
        k: usize,
        content_type: &str,
    ) -> Result<Vec<SearchMatch>> {
        if query.len() != EMBEDDING_DIMENSION {
            return Err(anyhow::anyhow!(
                "Query vector dimension mismatch: expected {}, got {}",
                EMBEDDING_DIMENSION,
                query.len()
            ));
        }

        let mut response = self
            .db
            .query("SELECT * FROM embedding WHERE content_type = $content_type")
            .bind(("content_type", content_type.to_string()))
            .await?;

        let records: Vec<EmbeddingRecord> = response.take(0)?;

        let mut matches: Vec<SearchMatch> = records
            .into_iter()
            .map(|r| {
                let similarity = cosine_similarity(query, &r.vector);
                SearchMatch {
                    vector_id: 0,
                    similarity,
                    metadata: VectorMetadata {
                        id: r.content_id,
                        text: r.text,
                        source: r.session_id,
                        content_type: r.content_type,
                        timestamp: Self::parse_datetime(&r.timestamp),
                        metadata: r.metadata,
                    },
                }
            })
            .collect();

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        matches.truncate(k);

        Ok(matches)
    }

    async fn remove_vector(&self, id: &str) -> Result<bool> {
        let result: Option<EmbeddingRecord> = self.delete("embedding", id).await?;
        Ok(result.is_some())
    }

    async fn has_session_embeddings(&self, session_id: &str) -> bool {
        let count = self.count_session_embeddings(session_id).await;
        count > 0
    }

    async fn count_session_embeddings(&self, session_id: &str) -> usize {
        let result = self
            .db
            .query("SELECT count() FROM embedding WHERE session_id = $session_id GROUP ALL")
            .bind(("session_id", session_id.to_string()))
            .await;

        if let Ok(mut response) = result {
            #[derive(Deserialize, SurrealValue)]
            struct CountResult {
                count: i64,
            }
            if let Ok(Some(count)) = response.take::<Option<CountResult>>(0) {
                return count.count as usize;
            }
        }

        0
    }

    async fn total_count(&self) -> usize {
        let result = self
            .db
            .query("SELECT count() FROM embedding GROUP ALL")
            .await;

        if let Ok(mut response) = result {
            #[derive(Deserialize, SurrealValue)]
            struct CountResult {
                count: i64,
            }
            if let Ok(Some(count)) = response.take::<Option<CountResult>>(0) {
                return count.count as usize;
            }
        }

        0
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_surrealdb_session_operations() {
        let storage = SurrealDBStorage::new("mem://", None, None)
            .await
            .expect("Failed to create SurrealDB storage");

        // Create test session
        let session_id = Uuid::new_v4();
        let session = ActiveSession::new(
            session_id,
            Some("Test Session".to_string()),
            Some("A test session".to_string()),
        );

        // Save session
        storage
            .save_session(&session)
            .await
            .expect("Failed to save session");

        // Load session
        let loaded = storage
            .load_session(session_id)
            .await
            .expect("Failed to load session");
        assert_eq!(session.id(), loaded.id());

        // Check exists
        assert!(storage.session_exists(session_id).await.unwrap());

        // Delete
        storage
            .delete_session(session_id)
            .await
            .expect("Failed to delete session");
        assert!(!storage.session_exists(session_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_entity_operations() {
        let storage = SurrealDBStorage::new("mem://", None, None)
            .await
            .expect("Failed to create SurrealDB storage");

        let session_id = Uuid::new_v4();
        let entity = EntityData {
            name: "rust".to_string(),
            entity_type: EntityType::Technology,
            first_mentioned: Utc::now(),
            last_mentioned: Utc::now(),
            mention_count: 1,
            importance_score: 1.0,
            description: Some("Rust programming language".to_string()),
        };

        // Upsert entity
        storage
            .upsert_entity(session_id, &entity)
            .await
            .expect("Failed to upsert entity");

        // Get entity
        let loaded = storage
            .get_entity(session_id, "rust")
            .await
            .expect("Failed to get entity");
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().name, "rust");

        // List entities
        let entities = storage
            .list_entities(session_id)
            .await
            .expect("Failed to list entities");
        assert_eq!(entities.len(), 1);

        // Delete entity
        storage
            .delete_entity(session_id, "rust")
            .await
            .expect("Failed to delete entity");
        let deleted = storage
            .get_entity(session_id, "rust")
            .await
            .expect("Failed to check deleted");
        assert!(deleted.is_none());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.0001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.0001);
    }

    /// Test remote connection to Docker SurrealDB (requires running Docker container)
    /// Run with: cargo test --features surrealdb-storage test_remote_surrealdb -- --ignored
    #[tokio::test]
    #[ignore = "Requires running Docker SurrealDB at localhost:8000"]
    async fn test_remote_surrealdb_connection() {
        // Connect to Docker SurrealDB (default: root/root)
        let storage = SurrealDBStorage::new("localhost:8000", Some("root"), Some("root"))
            .await
            .expect("Failed to connect to remote SurrealDB");

        // Create test session
        let session_id = Uuid::new_v4();
        let session = ActiveSession::new(
            session_id,
            Some("Remote Test Session".to_string()),
            Some("Testing remote connection".to_string()),
        );

        // Save session
        storage
            .save_session(&session)
            .await
            .expect("Failed to save session to remote");

        // Load session
        let loaded = storage
            .load_session(session_id)
            .await
            .expect("Failed to load session from remote");
        assert_eq!(session.id(), loaded.id());
        assert_eq!(loaded.name().clone(), Some("Remote Test Session".to_string()));

        // Clean up
        storage
            .delete_session(session_id)
            .await
            .expect("Failed to delete session from remote");

        println!("Remote SurrealDB connection test passed!");
    }
}
