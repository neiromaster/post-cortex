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
use crate::core::structured_context::StructuredContext;
use crate::session::active_session::{ActiveSession, ChangeRecord, CodeReference};
use anyhow::Result;

use rocksdb::{DB, Options, WriteBatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

use uuid::Uuid;

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

        // Configure RocksDB options for optimal performance
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Performance optimizations
        opts.set_max_open_files(10000);
        opts.set_use_fsync(false);
        opts.set_bytes_per_sync(0);
        opts.set_max_write_buffer_number(32);
        opts.set_write_buffer_size(536_870_912);
        opts.set_target_file_size_base(1_073_741_824);
        opts.set_min_write_buffer_number_to_merge(4);
        opts.set_level_zero_stop_writes_trigger(2000);
        opts.set_level_zero_slowdown_writes_trigger(0);
        opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);

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
        info!("RealRocksDBStorage: Saving session with ID: {}", session.id());

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
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize checkpoint: {}", e))?;

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
                let update_data = bincode::serde::encode_to_vec(update, bincode::config::standard())
                    .map_err(|e| anyhow::anyhow!("Failed to serialize update: {}", e))?;

                batch.put(key.as_bytes(), &update_data);
            }

            db.write(batch)?;

            info!(
                "RealRocksDBStorage: Batch saved {} updates for session {}",
                updates_len,
                session_id
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

            let iter = db.iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if key_str.starts_with("session:")
                    && !key_str.contains(":update:")
                    && let Some(uuid_str) = key_str.strip_prefix("session:")
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
            let update_prefix = format!("update:{}:", session_id);
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            let mut keys_to_delete = Vec::new();

            for item in iter {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if key_str.starts_with(&update_prefix) {
                    keys_to_delete.push(key.to_vec());
                }
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

    /// Get total number of keys in database
    pub async fn get_key_count(&self) -> Result<usize> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<usize> {
            let mut count = 0;
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            for _ in iter {
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
            #[derive(Serialize, Deserialize)]
            struct WorkspaceData {
                id: Uuid,
                name: String,
                description: String,
                sessions: Vec<(Uuid, SessionRole)>,
                created_at: u64,
            }

            let workspace_data = WorkspaceData {
                id: workspace_id,
                name,
                description,
                sessions: session_ids.iter().map(|id| (*id, SessionRole::Primary)).collect(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let data = bincode::serde::encode_to_vec(&workspace_data, bincode::config::standard())
                .map_err(|e| anyhow::anyhow!("Failed to serialize workspace: {}", e))?;

            let key = format!("workspace:{}", workspace_id);
            db.put(key.as_bytes(), data)?;

            info!("RealRocksDBStorage: Workspace {} saved successfully", workspace_id);
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
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            let mut keys_to_delete = Vec::new();

            for item in iter {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);

                if key_str.starts_with(&ws_session_prefix) {
                    keys_to_delete.push(key.to_vec());
                }
            }

            let mut batch = WriteBatch::default();
            for key in keys_to_delete {
                batch.delete(&key);
            }
            db.write(batch)?;

            info!("RealRocksDBStorage: Workspace {} deleted successfully", workspace_id);
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
            #[derive(Serialize, Deserialize)]
            struct WorkspaceSession {
                workspace_id: Uuid,
                session_id: Uuid,
                role: crate::workspace::SessionRole,
                added_at: u64,
            }

            let ws_session = WorkspaceSession {
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
