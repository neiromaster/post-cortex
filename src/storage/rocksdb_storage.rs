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
use crate::core::context_update::ContextUpdate;
use crate::core::lockfree_vector_db::{SearchMatch, VectorMetadata};
use crate::core::structured_context::StructuredContext;
use crate::session::active_session::{ActiveSession, ChangeRecord, CodeReference};
use crate::storage::traits::{Storage, VectorStorage};
use crate::workspace::SessionRole;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StoredWorkspace {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub sessions: Vec<(Uuid, crate::workspace::SessionRole)>,
    pub created_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StoredWorkspaceSession {
    pub workspace_id: Uuid,
    pub session_id: Uuid,
    pub role: crate::workspace::SessionRole,
    pub added_at: u64,
}

/// Real RocksDB storage implementation for high performance
#[derive(Clone)]
pub struct RealRocksDBStorage {
    db: Arc<DB>,
    #[allow(dead_code)]
    data_dir: PathBuf,
}

impl RealRocksDBStorage {
    /// Create new RocksDB storage instance
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data_dir = path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !data_dir.exists() {
            std::fs::create_dir_all(&data_dir)?;
        }

        let db_path = data_dir.join("rocksdb");

        // Configure RocksDB options for balanced performance and safety
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Memory settings - balanced for daemon use (max ~512MB for write buffers)
        opts.set_max_open_files(1000); // Reduced from 10000 - most systems have lower limits
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB per buffer (was 512MB)
        opts.set_max_write_buffer_number(6); // Max 384MB for write buffers (was 32 = 16GB!)
        opts.set_min_write_buffer_number_to_merge(2); // Merge sooner (was 4)
        opts.set_target_file_size_base(64 * 1024 * 1024); // 64MB SST files (was 1GB)

        // Write throttling - prevent memory exhaustion
        opts.set_level_zero_slowdown_writes_trigger(20); // Enable throttling (was 0 = disabled)
        opts.set_level_zero_stop_writes_trigger(36); // Reduced from 2000

        // Durability settings - balanced for daemon use
        opts.set_use_fsync(false); // fdatasync is sufficient for most cases
        opts.set_bytes_per_sync(1024 * 1024); // Sync every 1MB (was 0 = never)

        // Compaction settings
        opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);

        // Enable prefix bloom filter (16 bytes)
        // NOTE: prefix_iterator() only works correctly when the search prefix is >= 16 bytes.
        // For shorter prefixes like "session:" (8 bytes), use iterator() with IteratorMode::From instead.
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));

        let db = DB::open(&opts, &db_path)?;

        info!(
            "RealRocksDBStorage: Initialized RocksDB at {}",
            db_path.display()
        );

        Ok(Self {
            db: Arc::new(db),
            data_dir,
        })
    }

    /// Save session to RocksDB
    pub async fn save_session(&self, session: &ActiveSession) -> Result<()> {
        info!(
            "RealRocksDBStorage: Saving session with ID: {}",
            session.id()
        );

        let db = self.db.clone();
        let session = session.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            // Serialize session data
            let session_data = bincode::serde::encode_to_vec(&session, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Failed to serialize session: {}", e))?;

            info!(
                "RealRocksDBStorage: Session data serialized, size: {} bytes",
                session_data.len()
            );

            // Use binary key for better performance
            let key = format!("session:{}", session.id());

            // Write to RocksDB
            db.put(key.as_bytes(), &session_data)?;

            info!("RealRocksDBStorage: Session saved to RocksDB");
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    /// Load session from RocksDB
    pub async fn load_session(&self, session_id: Uuid) -> Result<ActiveSession> {
        info!(
            "RealRocksDBStorage: Loading session with ID: {}",
            session_id
        );

        let db = self.db.clone();
        let key = format!("session:{}", session_id);

        tokio::task::spawn_blocking(move || -> Result<ActiveSession> {
            match db.get(key.as_bytes())? {
                Some(data) => {
                    info!(
                        "RealRocksDBStorage: Session data found, size: {} bytes",
                        data.len()
                    );

                    let (session, _): (ActiveSession, usize) =
                        bincode::serde::decode_from_slice(&data, bincode::config::standard())
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize session: {}", e))?;

                    info!("RealRocksDBStorage: Session deserialized successfully");
                    Ok(session)
                }
                None => {
                    info!("RealRocksDBStorage: Session not found");
                    Err(anyhow::anyhow!("Session not found"))
                }
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Save checkpoint to RocksDB
    pub async fn save_checkpoint(&self, checkpoint: &SessionCheckpoint) -> Result<()> {
        let db = self.db.clone();
        let checkpoint = checkpoint.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let checkpoint_data =
                bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard())
                    .map_err(|e| anyhow::anyhow!("Failed to serialize checkpoint: {}", e))?;

            let key = format!("checkpoint:{}", checkpoint.id);
            db.put(key.as_bytes(), &checkpoint_data)?;

            info!(
                "RealRocksDBStorage: Checkpoint saved with ID: {}",
                checkpoint.id
            );
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    /// Load checkpoint from RocksDB
    pub async fn load_checkpoint(&self, checkpoint_id: Uuid) -> Result<SessionCheckpoint> {
        let db = self.db.clone();
        let key = format!("checkpoint:{}", checkpoint_id);

        tokio::task::spawn_blocking(move || -> Result<SessionCheckpoint> {
            match db.get(key.as_bytes())? {
                Some(data) => {
                    let (checkpoint, _): (SessionCheckpoint, usize) =
                        bincode::serde::decode_from_slice(&data, bincode::config::standard())
                            .map_err(|e| {
                                anyhow::anyhow!("Failed to deserialize checkpoint: {}", e)
                            })?;

                    info!(
                        "RealRocksDBStorage: Checkpoint loaded with ID: {}",
                        checkpoint_id
                    );
                    Ok(checkpoint)
                }
                None => Err(anyhow::anyhow!("Checkpoint not found")),
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Batch save multiple updates efficiently
    pub async fn batch_save_updates(
        &self,
        session_id: Uuid,
        updates: Vec<ContextUpdate>,
    ) -> Result<()> {
        let db = self.db.clone();
        let updates_len = updates.len();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut batch = WriteBatch::default();

            for update in &updates {
                let key = format!("session:{}:update:{}", session_id, update.id);
                let update_data =
                    bincode::serde::encode_to_vec(update, bincode::config::standard())
                        .map_err(|e| anyhow::anyhow!("Failed to serialize update: {}", e))?;

                batch.put(key.as_bytes(), &update_data);
            }

            db.write(batch)?;

            info!(
                "RealRocksDBStorage: Batch saved {} updates for session {}",
                updates_len, session_id
            );

            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    /// List all session IDs
    pub async fn list_sessions(&self) -> Result<Vec<Uuid>> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<Uuid>> {
            let mut sessions = Vec::new();

            // Use iterator with seek - prefix_iterator doesn't work correctly
            // when prefix_extractor size (16 bytes) doesn't match our prefix (8 bytes "session:")
            let iter = db.iterator(rocksdb::IteratorMode::From(b"session:", rocksdb::Direction::Forward));
            for item in iter {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);

                // Stop when we've passed the "session:" prefix
                if !key_str.starts_with("session:") {
                    break;
                }

                // Skip update keys (format: "session:{id}:update:{update_id}")
                if key_str.contains(":update:") {
                    continue;
                }

                // Extract UUID from "session:{uuid}" format
                if let Some(uuid_str) = key_str.strip_prefix("session:")
                    && let Ok(uuid) = Uuid::parse_str(uuid_str)
                {
                    sessions.push(uuid);
                }
            }

            info!("RealRocksDBStorage: Found {} sessions", sessions.len());
            Ok(sessions)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> Result<String> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<String> {
            let stats = db
                .property_value(rocksdb::properties::STATS)?
                .unwrap_or_else(|| "No stats available".to_string());

            Ok(stats)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Compact database
    /// Force database compaction
    pub async fn compact(&self) -> Result<()> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            db.compact_range(None::<&[u8]>, None::<&[u8]>);
            info!("RealRocksDBStorage: Database compacted");
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Delete session and all related data
    pub async fn delete_session(&self, session_id: Uuid) -> Result<()> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut batch = WriteBatch::default();

            // Delete main session
            let session_key = format!("session:{}", session_id);
            batch.delete(session_key.as_bytes());

            // Delete all updates for this session
            // Key format: "session:{session_id}:update:{update_id}"
            let update_prefix = format!("session:{}:update:", session_id);
            let iter = db.iterator(rocksdb::IteratorMode::From(update_prefix.as_bytes(), rocksdb::Direction::Forward));
            let mut keys_to_delete = Vec::new();

            for item in iter {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);

                // Stop when we've passed our prefix (prefix_iterator may continue)
                if !key_str.starts_with(&update_prefix) {
                    break;
                }
                keys_to_delete.push(key.to_vec());
            }

            // Batch delete all related keys
            for key in keys_to_delete {
                batch.delete(&key);
            }
            db.write(batch)?;

            info!(
                "RealRocksDBStorage: Deleted session {} and related data",
                session_id
            );
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Get estimated number of keys in database (O(1) using RocksDB property)
    pub async fn get_key_count(&self) -> Result<usize> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<usize> {
            // Use RocksDB's estimate-num-keys property for O(1) performance
            // This is an approximation but avoids full table scan
            if let Some(count_str) = db.property_value(rocksdb::properties::ESTIMATE_NUM_KEYS)? {
                if let Ok(count) = count_str.parse::<usize>() {
                    return Ok(count);
                }
            }

            // Fallback to counting if property not available (shouldn't happen)
            let mut count = 0;
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                let _ = item?;
                count += 1;
            }

            Ok(count)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Check if session exists
    pub async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        let db = self.db.clone();
        let key = format!("session:{}", session_id);

        tokio::task::spawn_blocking(move || -> Result<bool> {
            Ok(db.get(key.as_bytes())?.is_some())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    // ========================================================================
    // Workspace Persistence Methods
    // ========================================================================

    /// List all workspaces with their sessions (hydrated from ws_session records)
    pub async fn list_workspaces(&self) -> Result<Vec<StoredWorkspace>> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<StoredWorkspace>> {
            let mut workspaces: HashMap<Uuid, StoredWorkspace> = HashMap::new();

            // First pass: Load all workspaces using iterator with seek
            // (prefix_iterator doesn't work with our 16-byte prefix extractor)
            let workspace_iter = db.iterator(rocksdb::IteratorMode::From(b"workspace:", rocksdb::Direction::Forward));
            for item in workspace_iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                // Stop when we've passed the "workspace:" prefix
                if !key_str.starts_with("workspace:") {
                    break;
                }

                if let Ok((mut workspace, _)) =
                    bincode::serde::decode_from_slice::<StoredWorkspace, _>(
                        &value,
                        bincode::config::standard(),
                    )
                {
                    // Clear stored sessions as they might be stale (source of truth is ws_session records)
                    workspace.sessions.clear();
                    workspaces.insert(workspace.id, workspace);
                }
            }

            // Second pass: Load all workspace-session associations using iterator with seek
            let ws_session_iter = db.iterator(rocksdb::IteratorMode::From(b"ws_session:", rocksdb::Direction::Forward));
            for item in ws_session_iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                // Stop when we've passed the "ws_session:" prefix
                if !key_str.starts_with("ws_session:") {
                    break;
                }

                if let Ok((ws_session, _)) = bincode::serde::decode_from_slice::<
                    StoredWorkspaceSession,
                    _,
                >(&value, bincode::config::standard())
                {
                    if let Some(ws) = workspaces.get_mut(&ws_session.workspace_id) {
                        ws.sessions.push((ws_session.session_id, ws_session.role));
                    }
                }
            }

            Ok(workspaces.into_values().collect())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Save workspace metadata to RocksDB
    pub async fn save_workspace_metadata(
        &self,
        workspace_id: Uuid,
        name: &str,
        description: &str,
        session_ids: &[Uuid],
    ) -> Result<()> {
        use crate::workspace::SessionRole;

        info!(
            "RealRocksDBStorage: Saving workspace {} ({})",
            name, workspace_id
        );

        let db = self.db.clone();
        let name = name.to_string();
        let description = description.to_string();
        let session_ids = session_ids.to_vec();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let workspace_data = StoredWorkspace {
                id: workspace_id,
                name,
                description,
                sessions: session_ids
                    .iter()
                    .map(|id| (*id, SessionRole::Primary))
                    .collect(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let data = bincode::serde::encode_to_vec(&workspace_data, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Failed to serialize workspace: {}", e))?;

            let key = format!("workspace:{}", workspace_id);
            db.put(key.as_bytes(), data)?;

            info!(
                "RealRocksDBStorage: Workspace {} saved successfully",
                workspace_id
            );
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    /// Delete workspace from RocksDB
    pub async fn delete_workspace(&self, workspace_id: Uuid) -> Result<()> {
        info!("RealRocksDBStorage: Deleting workspace {}", workspace_id);

        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let key = format!("workspace:{}", workspace_id);
            db.delete(key.as_bytes())?;

            // Also delete all workspace-session associations
            let ws_session_prefix = format!("ws_session:{}:", workspace_id);
            let iter = db.iterator(rocksdb::IteratorMode::From(ws_session_prefix.as_bytes(), rocksdb::Direction::Forward));
            let mut keys_to_delete = Vec::new();

            for item in iter {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);

                // Stop when we've passed our prefix
                if !key_str.starts_with(&ws_session_prefix) {
                    break;
                }
                keys_to_delete.push(key.to_vec());
            }

            let mut batch = WriteBatch::default();
            for key in keys_to_delete {
                batch.delete(&key);
            }
            db.write(batch)?;

            info!(
                "RealRocksDBStorage: Workspace {} deleted successfully",
                workspace_id
            );
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Add session to workspace association
    pub async fn add_session_to_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
        role: crate::workspace::SessionRole,
    ) -> Result<()> {
        info!(
            "RealRocksDBStorage: Adding session {} to workspace {} with role {:?}",
            session_id, workspace_id, role
        );

        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let ws_session = StoredWorkspaceSession {
                workspace_id,
                session_id,
                role,
                added_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let data = bincode::serde::encode_to_vec(&ws_session, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Failed to serialize workspace-session: {}", e))?;

            let key = format!("ws_session:{}:{}", workspace_id, session_id);
            db.put(key.as_bytes(), data)?;

            info!(
                "RealRocksDBStorage: Session {} added to workspace {} successfully",
                session_id, workspace_id
            );
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    /// Remove session from workspace association
    pub async fn remove_session_from_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
    ) -> Result<()> {
        info!(
            "RealRocksDBStorage: Removing session {} from workspace {}",
            session_id, workspace_id
        );

        let db = self.db.clone();
        let key = format!("ws_session:{}:{}", workspace_id, session_id);

        tokio::task::spawn_blocking(move || -> Result<()> {
            db.delete(key.as_bytes())?;

            info!(
                "RealRocksDBStorage: Session {} removed from workspace {} successfully",
                session_id, workspace_id
            );
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    // ========================================================================
    // Export/Import Support Methods
    // ========================================================================

    /// Load all updates for a session
    pub async fn load_session_updates(&self, session_id: Uuid) -> Result<Vec<ContextUpdate>> {
        let db = self.db.clone();
        let update_prefix = format!("session:{}:update:", session_id);

        tokio::task::spawn_blocking(move || -> Result<Vec<ContextUpdate>> {
            let mut updates = Vec::new();
            let iter = db.iterator(rocksdb::IteratorMode::From(
                update_prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            for item in iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if !key_str.starts_with(&update_prefix) {
                    break;
                }

                if let Ok((update, _)) = bincode::serde::decode_from_slice::<ContextUpdate, _>(
                    &value,
                    bincode::config::standard(),
                ) {
                    updates.push(update);
                }
            }

            info!(
                "RealRocksDBStorage: Loaded {} updates for session {}",
                updates.len(),
                session_id
            );
            Ok(updates)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// List all checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<SessionCheckpoint>> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<SessionCheckpoint>> {
            let mut checkpoints = Vec::new();
            let iter = db.iterator(rocksdb::IteratorMode::From(
                b"checkpoint:",
                rocksdb::Direction::Forward,
            ));

            for item in iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if !key_str.starts_with("checkpoint:") {
                    break;
                }

                if let Ok((checkpoint, _)) =
                    bincode::serde::decode_from_slice::<SessionCheckpoint, _>(
                        &value,
                        bincode::config::standard(),
                    )
                {
                    checkpoints.push(checkpoint);
                }
            }

            info!("RealRocksDBStorage: Listed {} checkpoints", checkpoints.len());
            Ok(checkpoints)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }
}

// ============================================================================
// Storage Trait Implementation
// ============================================================================

#[async_trait]
impl Storage for RealRocksDBStorage {
    async fn save_session(&self, session: &ActiveSession) -> Result<()> {
        RealRocksDBStorage::save_session(self, session).await
    }

    async fn load_session(&self, session_id: Uuid) -> Result<ActiveSession> {
        RealRocksDBStorage::load_session(self, session_id).await
    }

    async fn delete_session(&self, session_id: Uuid) -> Result<()> {
        RealRocksDBStorage::delete_session(self, session_id).await
    }

    async fn list_sessions(&self) -> Result<Vec<Uuid>> {
        RealRocksDBStorage::list_sessions(self).await
    }

    async fn session_exists(&self, session_id: Uuid) -> Result<bool> {
        RealRocksDBStorage::session_exists(self, session_id).await
    }

    async fn batch_save_updates(
        &self,
        session_id: Uuid,
        updates: Vec<ContextUpdate>,
    ) -> Result<()> {
        RealRocksDBStorage::batch_save_updates(self, session_id, updates).await
    }

    async fn load_session_updates(&self, session_id: Uuid) -> Result<Vec<ContextUpdate>> {
        RealRocksDBStorage::load_session_updates(self, session_id).await
    }

    async fn save_checkpoint(&self, checkpoint: &SessionCheckpoint) -> Result<()> {
        RealRocksDBStorage::save_checkpoint(self, checkpoint).await
    }

    async fn load_checkpoint(&self, checkpoint_id: Uuid) -> Result<SessionCheckpoint> {
        RealRocksDBStorage::load_checkpoint(self, checkpoint_id).await
    }

    async fn list_checkpoints(&self) -> Result<Vec<SessionCheckpoint>> {
        RealRocksDBStorage::list_checkpoints(self).await
    }

    async fn save_workspace_metadata(
        &self,
        workspace_id: Uuid,
        name: &str,
        description: &str,
        session_ids: &[Uuid],
    ) -> Result<()> {
        RealRocksDBStorage::save_workspace_metadata(self, workspace_id, name, description, session_ids).await
    }

    async fn delete_workspace(&self, workspace_id: Uuid) -> Result<()> {
        RealRocksDBStorage::delete_workspace(self, workspace_id).await
    }

    async fn list_workspaces(&self) -> Result<Vec<StoredWorkspace>> {
        RealRocksDBStorage::list_workspaces(self).await
    }

    async fn add_session_to_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
        role: SessionRole,
    ) -> Result<()> {
        RealRocksDBStorage::add_session_to_workspace(self, workspace_id, session_id, role).await
    }

    async fn remove_session_from_workspace(
        &self,
        workspace_id: Uuid,
        session_id: Uuid,
    ) -> Result<()> {
        RealRocksDBStorage::remove_session_from_workspace(self, workspace_id, session_id).await
    }

    async fn compact(&self) -> Result<()> {
        RealRocksDBStorage::compact(self).await
    }

    async fn get_key_count(&self) -> Result<usize> {
        RealRocksDBStorage::get_key_count(self).await
    }

    async fn get_stats(&self) -> Result<String> {
        RealRocksDBStorage::get_stats(self).await
    }
}

// ============================================================================
// VectorStorage Trait Implementation - Embedding Persistence
// ============================================================================

/// Stored embedding record for RocksDB persistence
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StoredEmbedding {
    pub content_id: String,
    pub session_id: String,
    pub vector: Vec<f32>,
    pub text: String,
    pub content_type: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl StoredEmbedding {
    /// Create from vector and metadata
    pub fn new(vector: Vec<f32>, metadata: VectorMetadata) -> Self {
        Self {
            content_id: metadata.id,
            session_id: metadata.source,
            vector,
            text: metadata.text,
            content_type: metadata.content_type,
            timestamp: metadata.timestamp,
            metadata: metadata.metadata,
        }
    }

    /// Convert to VectorMetadata
    pub fn to_metadata(&self) -> VectorMetadata {
        VectorMetadata {
            id: self.content_id.clone(),
            text: self.text.clone(),
            source: self.session_id.clone(),
            content_type: self.content_type.clone(),
            timestamp: self.timestamp,
            metadata: self.metadata.clone(),
        }
    }
}

impl RealRocksDBStorage {
    /// Save an embedding to RocksDB
    pub async fn save_embedding(&self, embedding: &StoredEmbedding) -> Result<()> {
        let db = self.db.clone();
        let embedding = embedding.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let key = format!("embedding:{}:{}", embedding.session_id, embedding.content_id);
            let data = bincode::serde::encode_to_vec(&embedding, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Failed to serialize embedding: {}", e))?;
            db.put(key.as_bytes(), &data)?;
            Ok(())
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))??;

        Ok(())
    }

    /// Load all embeddings for a session
    pub async fn load_session_embeddings(&self, session_id: &str) -> Result<Vec<StoredEmbedding>> {
        let db = self.db.clone();
        let prefix = format!("embedding:{}:", session_id);

        tokio::task::spawn_blocking(move || -> Result<Vec<StoredEmbedding>> {
            let mut embeddings = Vec::new();
            let iter = db.iterator(rocksdb::IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            for item in iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if !key_str.starts_with(&prefix) {
                    break;
                }

                if let Ok((embedding, _)) =
                    bincode::serde::decode_from_slice::<StoredEmbedding, _>(
                        &value,
                        bincode::config::standard(),
                    )
                {
                    embeddings.push(embedding);
                }
            }

            Ok(embeddings)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Load all embeddings from storage
    pub async fn load_all_embeddings(&self) -> Result<Vec<StoredEmbedding>> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<StoredEmbedding>> {
            let mut embeddings = Vec::new();
            let iter = db.iterator(rocksdb::IteratorMode::From(
                b"embedding:",
                rocksdb::Direction::Forward,
            ));

            for item in iter {
                let (key, value) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if !key_str.starts_with("embedding:") {
                    break;
                }

                if let Ok((embedding, _)) =
                    bincode::serde::decode_from_slice::<StoredEmbedding, _>(
                        &value,
                        bincode::config::standard(),
                    )
                {
                    embeddings.push(embedding);
                }
            }

            Ok(embeddings)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Delete an embedding
    pub async fn delete_embedding(&self, session_id: &str, content_id: &str) -> Result<bool> {
        let db = self.db.clone();
        let key = format!("embedding:{}:{}", session_id, content_id);

        tokio::task::spawn_blocking(move || -> Result<bool> {
            let existed = db.get(key.as_bytes())?.is_some();
            db.delete(key.as_bytes())?;
            Ok(existed)
        })
        .await
        .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
    }

    /// Count embeddings for a session
    pub async fn count_embeddings(&self, session_id: &str) -> usize {
        self.load_session_embeddings(session_id)
            .await
            .map(|e| e.len())
            .unwrap_or(0)
    }
}

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

#[async_trait]
impl VectorStorage for RealRocksDBStorage {
    async fn add_vector(&self, vector: Vec<f32>, metadata: VectorMetadata) -> Result<String> {
        let id = metadata.id.clone();
        let embedding = StoredEmbedding::new(vector, metadata);
        self.save_embedding(&embedding).await?;
        Ok(id)
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
        let embeddings = self.load_all_embeddings().await?;

        let mut matches: Vec<SearchMatch> = embeddings
            .into_iter()
            .map(|e| {
                let similarity = cosine_similarity(query, &e.vector);
                SearchMatch {
                    vector_id: 0,
                    similarity,
                    metadata: e.to_metadata(),
                }
            })
            .collect();

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(k);

        Ok(matches)
    }

    async fn search_in_session(
        &self,
        query: &[f32],
        k: usize,
        session_id: &str,
    ) -> Result<Vec<SearchMatch>> {
        let embeddings = self.load_session_embeddings(session_id).await?;

        let mut matches: Vec<SearchMatch> = embeddings
            .into_iter()
            .map(|e| {
                let similarity = cosine_similarity(query, &e.vector);
                SearchMatch {
                    vector_id: 0,
                    similarity,
                    metadata: e.to_metadata(),
                }
            })
            .collect();

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(k);

        Ok(matches)
    }

    async fn search_by_content_type(
        &self,
        query: &[f32],
        k: usize,
        content_type: &str,
    ) -> Result<Vec<SearchMatch>> {
        let embeddings = self.load_all_embeddings().await?;

        let mut matches: Vec<SearchMatch> = embeddings
            .into_iter()
            .filter(|e| e.content_type == content_type)
            .map(|e| {
                let similarity = cosine_similarity(query, &e.vector);
                SearchMatch {
                    vector_id: 0,
                    similarity,
                    metadata: e.to_metadata(),
                }
            })
            .collect();

        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(k);

        Ok(matches)
    }

    async fn remove_vector(&self, id: &str) -> Result<bool> {
        // Parse id to find session_id:content_id
        // The id format is typically the content_id, we need to search for it
        let embeddings = self.load_all_embeddings().await?;
        for e in embeddings {
            if e.content_id == id {
                return self.delete_embedding(&e.session_id, &e.content_id).await;
            }
        }
        Ok(false)
    }

    async fn has_session_embeddings(&self, session_id: &str) -> bool {
        self.count_embeddings(session_id).await > 0
    }

    async fn count_session_embeddings(&self, session_id: &str) -> usize {
        self.count_embeddings(session_id).await
    }

    async fn total_count(&self) -> usize {
        self.load_all_embeddings()
            .await
            .map(|e| e.len())
            .unwrap_or(0)
    }

    async fn get_session_vectors(&self, session_id: &str) -> Result<Vec<(Vec<f32>, VectorMetadata)>> {
        let embeddings = self.load_session_embeddings(session_id).await?;
        Ok(embeddings
            .into_iter()
            .map(|e| {
                let metadata = e.to_metadata();
                (e.vector, metadata)
            })
            .collect())
    }

    async fn get_all_vectors(&self) -> Result<Vec<(Vec<f32>, VectorMetadata)>> {
        let embeddings = self.load_all_embeddings().await?;
        Ok(embeddings
            .into_iter()
            .map(|e| {
                let metadata = e.to_metadata();
                (e.vector, metadata)
            })
            .collect())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SessionCheckpoint {
    pub id: Uuid,
    pub session_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,

    // Complete context snapshot
    pub structured_context: StructuredContext,
    pub recent_updates: Vec<ContextUpdate>,
    pub code_references: HashMap<String, Vec<CodeReference>>,
    pub change_history: Vec<ChangeRecord>,

    // Metadata
    pub total_updates: usize,
    pub context_quality_score: f32,
    pub compression_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    // use tokio_test;

    #[tokio::test]
    async fn test_rocksdb_session_operations() {
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let storage = RealRocksDBStorage::new(temp_dir.path())
            .await
            .expect("Failed to create RocksDB storage");

        // Create test session
        let session_id = Uuid::new_v4();
        let session = ActiveSession::new(
            session_id,
            Some("Test Session".to_string()),
            Some("A test session for RocksDB storage operations".to_string()),
        );

        // Save session
        storage
            .save_session(&session)
            .await
            .expect("Failed to save session");

        // Load session
        let loaded_session = storage
            .load_session(session.id())
            .await
            .expect("Failed to load session");
        assert_eq!(session.id(), loaded_session.id());

        // Check if session exists
        assert!(
            storage
                .session_exists(session.id())
                .await
                .expect("Failed to check session existence")
        );

        // Delete session
        storage
            .delete_session(session.id())
            .await
            .expect("Failed to delete session");
        assert!(
            !storage
                .session_exists(session.id())
                .await
                .expect("Failed to check session existence after deletion")
        );
    }
}
