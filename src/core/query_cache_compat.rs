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
//! Compatibility wrapper for lock-free query cache
//!
//! This module provides a compatibility layer that implements the same interface
//! as the original QueryCache but uses the lock-free implementation internally.

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::content_vectorizer::SemanticSearchResult;
use crate::core::lockfree_query_cache::{
    LockFreeQueryCache, LockFreeQueryCacheConfig
};

/// Configuration for query caching system (original format for compatibility)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryCacheConfig {
    /// Maximum number of cached queries
    pub max_cache_size: usize,
    /// Time-to-live for cached results in minutes
    pub ttl_minutes: i64,
    /// Similarity threshold for query matching (0.0-1.0)
    pub similarity_threshold: f32,
    /// Enable intelligent prefetching
    pub enable_prefetching: bool,
    /// Maximum number of query variations to prefetch
    pub max_prefetch_variations: usize,
    /// Enable cache statistics collection
    pub enable_stats: bool,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1000,
            ttl_minutes: 30,
            similarity_threshold: 0.85,
            enable_prefetching: true,
            max_prefetch_variations: 5,
            enable_stats: true,
        }
    }
}

/// Compatibility wrapper that provides the same interface as QueryCache but uses lock-free implementation
pub struct QueryCache {
    inner: Arc<LockFreeQueryCache>,
}

impl QueryCache {
    /// Create a new query cache with the original config
    pub fn new(config: QueryCacheConfig) -> Self {
        // Convert original config to lock-free config
        let lockfree_config = LockFreeQueryCacheConfig {
            max_cache_size: config.max_cache_size,
            ttl_minutes: config.ttl_minutes,
            similarity_threshold: config.similarity_threshold,
            enable_prefetching: config.enable_prefetching,
            max_prefetch_variations: config.max_prefetch_variations,
            enable_stats: config.enable_stats,
        };

        Self {
            inner: Arc::new(LockFreeQueryCache::new(lockfree_config)),
        }
    }

    /// Search for cached results for a query (async for compatibility)
    pub async fn search(
        &self,
        query_text: &str,
        query_vector: &[f32],
        params_hash: u64,
    ) -> Option<Vec<SemanticSearchResult>> {
        // The lock-free implementation is synchronous, but we maintain async interface for compatibility
        self.inner.search(query_text, query_vector, params_hash)
    }

    /// Cache query results (async for compatibility)
    pub async fn cache_results(
        &self,
        query_text: String,
        query_vector: Vec<f32>,
        results: Vec<SemanticSearchResult>,
        params_hash: u64,
        session_id: Option<Uuid>,
    ) -> Result<()> {
        // The lock-free implementation is synchronous, but we maintain async interface for compatibility
        self.inner.cache_results(query_text, query_vector, results, params_hash, session_id)
    }

    /// Get cache statistics (async for compatibility)
    pub async fn get_stats(&self) -> QueryCacheStats {
        let snapshot = self.inner.get_stats();
        
        // Convert snapshot to original format for compatibility
        QueryCacheStats {
            total_queries: snapshot.total_queries,
            cache_hits: snapshot.cache_hits,
            cache_misses: snapshot.cache_misses,
            expired_removed: snapshot.expired_removed,
            evicted_entries: snapshot.evicted_entries,
            avg_hit_similarity: snapshot.avg_hit_similarity,
            current_cache_size: snapshot.current_cache_size,
            estimated_memory_bytes: snapshot.estimated_memory_bytes,
            hit_rate: snapshot.hit_rate,
            avg_time_saved_ms: snapshot.avg_time_saved_ms,
        }
    }

    /// Clear all cached entries (async for compatibility)
    pub async fn clear(&self) -> Result<()> {
        // The lock-free implementation is synchronous, but we maintain async interface for compatibility
        self.inner.clear()
    }

    /// Invalidate all cache entries for a specific session
    /// More efficient than clearing the entire cache when only one session changed
    pub async fn invalidate_session(&self, session_id: Uuid) -> Result<()> {
        // The lock-free implementation is synchronous, but we maintain async interface for compatibility
        self.inner.invalidate_session(session_id)
    }

    /// Get cache efficiency metrics (async for compatibility)
    pub async fn get_efficiency_metrics(&self) -> std::collections::HashMap<String, f32> {
        // The lock-free implementation is synchronous, but we maintain async interface for compatibility
        self.inner.get_efficiency_metrics()
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new(QueryCacheConfig::default())
    }
}

/// Statistics for query cache performance (original format for compatibility)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryCacheStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub expired_removed: u64,
    pub evicted_entries: u64,
    pub avg_hit_similarity: f32,
    pub current_cache_size: usize,
    pub estimated_memory_bytes: usize,
    pub hit_rate: f32,
    pub avg_time_saved_ms: f32,
}

/// A cached query (original format for compatibility)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CachedQuery {
    pub id: Uuid,
    pub query_text: String,
    pub query_vector: Vec<f32>,
    pub results: Vec<SemanticSearchResult>,
    pub cached_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u64,
    pub params_hash: u64,
    pub session_id: Option<Uuid>,
    pub efficiency_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compatibility_wrapper() {
        let config = QueryCacheConfig::default();
        let cache = QueryCache::new(config);

        let stats = cache.get_stats().await;
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[tokio::test]
    async fn test_async_compatibility() {
        let cache = QueryCache::default();

        let query_text = "test query".to_string();
        let query_vector = vec![0.1, 0.2, 0.3];
        let results = vec![];
        let params_hash = 12345u64;

        // Cache the results
        cache
            .cache_results(
                query_text.clone(),
                query_vector.clone(),
                results,
                params_hash,
                None,
            )
            .await
            .unwrap();

        // Try to retrieve
        let cached_results = cache.search(&query_text, &query_vector, params_hash).await;
        assert!(cached_results.is_some());
    }
}