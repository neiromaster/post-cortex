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
//! Content Vectorization Module
//!
//! This module handles the conversion of text content into vector representations
//! for semantic search and similarity analysis.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::core::context_update::ContextUpdate;
use crate::core::embeddings::{EmbeddingConfig, LocalEmbeddingEngine};
use crate::core::query_cache::{QueryCache, QueryCacheConfig};
use crate::core::vector_db::{FastVectorDB, VectorDbConfig, VectorMetadata};
use crate::session::active_session::ActiveSession;

/// Content types for vectorization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentType {
    /// Update content from context updates
    UpdateContent,
    /// Entity descriptions
    EntityDescription,
    /// User questions and messages
    UserMessage,
    /// Decision points and rationale
    DecisionPoint,
    /// Code snippets and technical content
    CodeSnippet,
    /// Problem-solution pairs
    ProblemSolution,
    /// General session metadata
    SessionMetadata,
}

impl ContentType {
    /// Get the importance weight for this content type
    #[must_use]
    pub const fn importance_weight(&self) -> f32 {
        match self {
            Self::DecisionPoint => 1.0,
            Self::ProblemSolution => 0.9,
            Self::UserMessage => 0.8,
            Self::UpdateContent => 0.7,
            Self::CodeSnippet => 0.6,
            Self::EntityDescription => 0.5,
            Self::SessionMetadata => 0.3,
        }
    }
}

/// Threshold for switching to parallel processing
/// Below this threshold, sequential processing is faster due to reduced overhead
const PARALLEL_PROCESSING_THRESHOLD: usize = 50;

impl SemanticSearchResult {
    /// Interpret the similarity score with a human-readable quality level
    #[must_use]
    pub const fn similarity_quality(&self) -> &'static str {
        if self.similarity_score >= 0.85 {
            "Excellent"
        } else if self.similarity_score >= 0.70 {
            "Very Good"
        } else if self.similarity_score >= 0.55 {
            "Good"
        } else if self.similarity_score >= 0.40 {
            "Moderate"
        } else if self.similarity_score >= 0.30 {
            "Fair"
        } else {
            "Weak"
        }
    }

    /// Get a detailed explanation of how the combined score was calculated
    #[must_use]
    pub fn score_explanation(&self) -> String {
        format!(
            "Combined Score: {:.4} = (Similarity: {:.4} × 0.7) + (Importance: {:.2} × 0.3) | Quality: {}",
            self.combined_score,
            self.similarity_score,
            self.importance_score,
            self.similarity_quality()
        )
    }

    /// Check if this result meets a given quality threshold
    #[must_use]
    pub fn meets_quality(&self, threshold: &str) -> bool {
        let min_score = match threshold {
            "excellent" => 0.85,
            "very_good" => 0.70,
            "good" => 0.55,
            "moderate" => 0.40,
            "fair" => 0.30,
            _ => 0.0,
        };
        self.similarity_score >= min_score
    }
}

/// Search result with enhanced metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchResult {
    pub content_id: String,
    pub session_id: Uuid,
    pub content_type: ContentType,
    pub text_content: String,
    pub similarity_score: f32,
    pub importance_score: f32,
    pub timestamp: DateTime<Utc>,
    pub combined_score: f32,
}

/// Configuration for content vectorization
#[derive(Debug, Clone)]
pub struct ContentVectorizerConfig {
    pub embedding_config: EmbeddingConfig,
    pub vector_db_config: VectorDbConfig,
    pub min_text_length: usize,
    pub max_text_length: usize,
    pub batch_size: usize,
    pub enable_entity_vectorization: bool,
    pub enable_cross_session_search: bool,
    pub query_cache_config: QueryCacheConfig,
    pub enable_query_caching: bool,
}

impl Default for ContentVectorizerConfig {
    fn default() -> Self {
        Self {
            embedding_config: EmbeddingConfig::default(),
            vector_db_config: VectorDbConfig::default(),
            min_text_length: 10,
            max_text_length: 2000,
            batch_size: 32,
            enable_entity_vectorization: true,
            enable_cross_session_search: true,
            query_cache_config: QueryCacheConfig::default(),
            enable_query_caching: true,
        }
    }
}

/// Main content vectorization pipeline
pub struct ContentVectorizer {
    embedding_engine: LocalEmbeddingEngine,
    vector_db: FastVectorDB,
    config: ContentVectorizerConfig,
    query_cache: Option<QueryCache>,
}

impl ContentVectorizer {
    /// Create a new `ContentVectorizer` with the given configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the vector database or content vectorizer fails to initialize
    pub async fn new(config: ContentVectorizerConfig) -> Result<Self> {
        info!(
            "Initializing ContentVectorizer with caching: {}",
            config.enable_query_caching
        );

        let embedding_engine = LocalEmbeddingEngine::new(config.embedding_config.clone())
            .await
            .context("Failed to initialize embedding engine")?;

        let vector_db = FastVectorDB::new(config.vector_db_config.clone())
            .context("Failed to initialize vector database")?;

        let query_cache = if config.enable_query_caching {
            Some(QueryCache::new(config.query_cache_config.clone()))
        } else {
            None
        };

        debug!(
            "ContentVectorizer initialized successfully with query caching: {}",
            config.enable_query_caching
        );

        Ok(Self {
            embedding_engine,
            vector_db,
            config,
            query_cache,
        })
    }

    /// Vectorize all content in a session
    /// # Errors
    ///
    /// Returns an error if vectorization fails or if there are issues with the vector database
    pub async fn vectorize_session(&self, session: &ActiveSession) -> Result<usize> {
        info!("Vectorizing session: {}", session.id());
        let mut vectorized_count = 0;

        // Vectorize context updates
        vectorized_count += self.vectorize_context_updates(session).await?;

        // Vectorize entity descriptions if enabled
        if self.config.enable_entity_vectorization {
            vectorized_count += self.vectorize_entities(session).await?;
        }

        info!(
            "Vectorized {} items for session {}",
            vectorized_count, session.id()
        );
        Ok(vectorized_count)
    }

    /// Vectorize only the most recent update (incremental vectorization)
    /// This is much more efficient than re-vectorizing the entire session
    pub async fn vectorize_latest_update(&self, session: &ActiveSession) -> Result<usize> {
        debug!("Vectorizing latest update for session: {}", session.id());

        // Get the most recent update from hot context (VecDeque uses back() not last())
        let latest_update = session.hot_context.back();

        // Extract update or return early if none
        let update = match latest_update {
            Some(u) => u,
            None => {
                debug!("No updates to vectorize for session {}", session.id());
                return Ok(0);
            }
        };

        let raw_text = Self::extract_text_from_update(&update);

        if !self.should_vectorize_text(&raw_text) {
            debug!("Skipping vectorization - text too short or empty");
            return Ok(0);
        }

        // Apply smart summarization if text is too long
        let prepared_text = self.prepare_text_for_vectorization(&raw_text);

        debug!(
            "Prepared text for vectorization: {} chars (original: {} chars)",
            prepared_text.len(),
            raw_text.len()
        );

        let content_type = Self::determine_content_type(&update);

        // Generate embedding for single update
        let embeddings = self.embedding_engine.encode_batch(vec![prepared_text.clone()]).await?;

        if let Some(embedding) = embeddings.into_iter().next() {
            let metadata = VectorMetadata::new(
                update.id.to_string(),
                prepared_text,
                session.id().to_string(),
                format!("{content_type:?}"),
            );

            match self.vector_db.add_vector(embedding, metadata) {
                Ok(_) => {
                    debug!("Successfully vectorized latest update {}", update.id);
                    Ok(1)
                }
                Err(e) => {
                    warn!("Failed to vectorize latest update: {}", e);
                    Ok(0)
                }
            }
        } else {
            warn!("No embedding generated for latest update");
            Ok(0)
        }
    }

    /// Vectorize context updates from a session
    async fn vectorize_context_updates(&self, session: &ActiveSession) -> Result<usize> {
        let total_updates = session.hot_context.len() + session.warm_context.len();

        // Use parallel processing for large sessions, sequential for small ones
        let (texts_to_embed, metadata_list) = if total_updates >= PARALLEL_PROCESSING_THRESHOLD {
            debug!(
                "Using parallel processing for {} updates (threshold: {})",
                total_updates, PARALLEL_PROCESSING_THRESHOLD
            );
            self.collect_updates_parallel(session)?
        } else {
            debug!("Using sequential processing for {} updates", total_updates);
            self.collect_updates_sequential(session)?
        };

        if texts_to_embed.is_empty() {
            debug!("No texts to vectorize for session {}", session.id());
            return Ok(0);
        }

        // Generate embeddings in batches
        let embeddings = self.embedding_engine.encode_batch(texts_to_embed.clone()).await?;

        // Add to vector database
        let mut added_count = 0;
        for (embedding, metadata) in embeddings.into_iter().zip(metadata_list) {
            match self.vector_db.add_vector(embedding, metadata) {
                Ok(_) => added_count += 1,
                Err(e) => warn!("Failed to add vector to database: {}", e),
            }
        }

        debug!("Added {} context update vectors", added_count);
        Ok(added_count)
    }

    /// Vectorize entity descriptions
    async fn vectorize_entities(&self, session: &ActiveSession) -> Result<usize> {
        let entity_count = session.entity_graph.entities.len();

        // Use parallel processing for large entity graphs, sequential for small ones
        let (texts_to_embed, metadata_list) = if entity_count >= PARALLEL_PROCESSING_THRESHOLD {
            debug!(
                "Using parallel processing for {} entities (threshold: {})",
                entity_count, PARALLEL_PROCESSING_THRESHOLD
            );

            // Parallel entity processing
            let results: Vec<(String, VectorMetadata)> = session
                .entity_graph
                .entities
                .par_iter()
                .filter_map(|(entity_name, entity_data)| {
                    // Create a comprehensive description of the entity
                    let mut description_parts = Vec::new();

                    // Add entity name
                    description_parts.push(entity_name.clone());

                    // Add entity type if available
                    description_parts.push(format!("Type: {:?}", entity_data.entity_type));

                    // Add related entities context
                    let related: Vec<String> = vec![]; // TODO: Get relationships from entity graph
                    if !related.is_empty() {
                        description_parts.push(format!("Related to: {}", related.join(", ")));
                    }

                    // Add mention context
                    if entity_data.mention_count > 0 {
                        description_parts
                            .push(format!("Mentioned {} times", entity_data.mention_count));
                    }

                    let entity_text = description_parts.join(". ");

                    if self.should_vectorize_text(&entity_text) {
                        let metadata = VectorMetadata::new(
                            format!("entity:{entity_name}"),
                            entity_text.clone(),
                            session.id().to_string(),
                            "EntityDescription".to_string(),
                        );
                        Some((entity_text, metadata))
                    } else {
                        None
                    }
                })
                .collect();

            results.into_iter().unzip()
        } else {
            debug!("Using sequential processing for {} entities", entity_count);

            // Sequential entity processing
            let mut texts_to_embed = Vec::new();
            let mut metadata_list = Vec::new();

            for (entity_name, entity_data) in &session.entity_graph.entities {
                // Create a comprehensive description of the entity
                let mut description_parts = Vec::new();

                // Add entity name
                description_parts.push(entity_name.clone());

                // Add entity type if available
                description_parts.push(format!("Type: {:?}", entity_data.entity_type));

                // Add related entities context
                let related: Vec<String> = vec![]; // TODO: Get relationships from entity graph
                if !related.is_empty() {
                    description_parts.push(format!("Related to: {}", related.join(", ")));
                }

                // Add mention context
                if entity_data.mention_count > 0 {
                    description_parts
                        .push(format!("Mentioned {} times", entity_data.mention_count));
                }

                let entity_text = description_parts.join(". ");

                if self.should_vectorize_text(&entity_text) {
                    texts_to_embed.push(entity_text.clone());
                    metadata_list.push(VectorMetadata::new(
                        format!("entity:{entity_name}"),
                        entity_text,
                        session.id().to_string(),
                        "EntityDescription".to_string(),
                    ));
                }
            }

            (texts_to_embed, metadata_list)
        };

        if texts_to_embed.is_empty() {
            debug!("No entity texts to vectorize for session {}", session.id());
            return Ok(0);
        }

        // Generate embeddings
        let embeddings = self.embedding_engine.encode_batch(texts_to_embed.clone()).await?;

        // Add to vector database
        let mut added_count = 0;
        for (embedding, metadata) in embeddings.into_iter().zip(metadata_list) {
            match self.vector_db.add_vector(embedding, metadata) {
                Ok(_) => added_count += 1,
                Err(e) => warn!("Failed to add entity vector to database: {}", e),
            }
        }

        debug!("Added {} entity vectors", added_count);
        Ok(added_count)
    }

    /// Perform semantic search across all vectorized content
    /// # Errors
    ///
    /// Returns an error if the query cannot be embedded or if the vector database search fails
    pub async fn semantic_search(
        &self,
        query: &str,
        limit: usize,
        session_filter: Option<Uuid>,
        date_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
    ) -> Result<Vec<SemanticSearchResult>> {
        debug!("Performing semantic search for: '{}'", query);

        // Create a hash of search parameters for cache key
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        query.hash(&mut hasher);
        limit.hash(&mut hasher);
        if let Some(ref session_id) = session_filter {
            session_id.hash(&mut hasher);
        }
        if let Some(ref range) = date_range {
            range.0.timestamp().hash(&mut hasher);
            range.1.timestamp().hash(&mut hasher);
        }
        let params_hash = hasher.finish();

        // Check query cache first
        let query_embedding = if let Some(ref cache) = self.query_cache {
            let query_embedding = self.embedding_engine.encode_text(query).await?;

            if let Some(cached_results) = cache.search(query, &query_embedding, params_hash).await {
                debug!("Cache hit for semantic search query: '{}'", query);
                return Ok(cached_results);
            }
            query_embedding
        } else {
            self.embedding_engine.encode_text(query).await?
        };

        // Perform actual search with optional filters
        let search_results = match (session_filter, date_range) {
            // Both session and date filters
            (Some(session_id), Some((start, end))) => {
                self.vector_db
                    .search_with_filter(&query_embedding, limit, |metadata| {
                        metadata.source == session_id.to_string()
                            && metadata.timestamp >= start
                            && metadata.timestamp <= end
                    })?
            }
            // Only session filter
            (Some(session_id), None) => {
                self.vector_db
                    .search_in_source(&query_embedding, limit, &session_id.to_string())?
            }
            // Only date filter
            (None, Some((start, end))) if self.config.enable_cross_session_search => self
                .vector_db
                .search_with_filter(&query_embedding, limit, |metadata| {
                    metadata.timestamp >= start && metadata.timestamp <= end
                })?,
            // No filters, cross-session enabled
            (None, None) if self.config.enable_cross_session_search => {
                self.vector_db.search(&query_embedding, limit)?
            }
            // No cross-session search enabled
            _ => return Ok(Vec::new()),
        };

        // Convert to semantic search results
        let mut results = Vec::new();
        for result in search_results {
            let content_type = Self::parse_content_type(&result.metadata.content_type);
            let session_id = Uuid::parse_str(&result.metadata.source)
                .context("Invalid session ID in metadata")?;

            // Calculate combined score (similarity + importance + content type weight)
            let importance_weight = content_type.importance_weight();
            let combined_score = result.similarity.mul_add(0.7, importance_weight * 0.3);

            results.push(SemanticSearchResult {
                content_id: result.metadata.id,
                session_id,
                content_type,
                text_content: result.metadata.text,
                similarity_score: result.similarity,
                importance_score: importance_weight,
                timestamp: result.metadata.timestamp,
                combined_score,
            });
        }

        // Deduplicate results by content_id (keeps highest score or newest timestamp)
        let original_count = results.len();
        let mut dedup_map: HashMap<String, SemanticSearchResult> =
            HashMap::with_capacity(results.len());

        for result in results {
            let content_id = result.content_id.clone();

            match dedup_map.entry(content_id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get();
                    // Replace if new result has higher score or newer timestamp
                    if result.combined_score > existing.combined_score
                        || (result.combined_score == existing.combined_score
                            && result.timestamp > existing.timestamp)
                    {
                        entry.insert(result);
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(result);
                }
            }
        }

        let results: Vec<SemanticSearchResult> = dedup_map.into_values().collect();
        debug!(
            "Deduplicated {} results to {} unique",
            original_count,
            results.len()
        );

        // Sort by combined score (highest first)
        let mut results = results;
        results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cache the results
        if let Some(ref cache) = self.query_cache
            && let Err(e) = cache
                .cache_results(
                    query.to_string(),
                    query_embedding,
                    results.clone(),
                    params_hash,
                    session_filter,
                )
                .await
        {
            warn!("Failed to cache search results: {}", e);
        }

        debug!("Found {} semantic search results", results.len());
        Ok(results)
    }

    /// Get query cache statistics
    pub async fn get_query_cache_stats(&self) -> Option<crate::core::query_cache::QueryCacheStats> {
        match &self.query_cache {
            Some(cache) => Some(cache.get_stats().await),
            None => None,
        }
    }

    /// Get query cache efficiency metrics
    pub async fn get_cache_efficiency_metrics(&self) -> Option<HashMap<String, f32>> {
        match &self.query_cache {
            Some(cache) => Some(cache.get_efficiency_metrics().await),
            None => None,
        }
    }

    /// Clear query cache
    ///
    /// # Errors
    ///
    /// Returns an error if the cache clearing operation fails
    pub async fn clear_query_cache(&self) -> Result<()> {
        if let Some(ref cache) = self.query_cache {
            cache.clear().await?;
            info!("Query cache cleared successfully");
        }
        Ok(())
    }

    /// Check if query caching is enabled
    pub const fn is_query_caching_enabled(&self) -> bool {
        self.query_cache.is_some()
    }

    /// Find related content across sessions
    /// Find related content within a session
    ///
    /// # Errors
    ///
    /// Returns an error if the semantic search fails or if query processing encounters issues
    pub async fn find_related_content(
        &self,
        session: &ActiveSession,
        topic: &str,
        limit: usize,
    ) -> Result<Vec<SemanticSearchResult>> {
        let results = self.semantic_search(topic, limit * 2, None, None).await?;

        // Filter out current session and return top results
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| r.session_id != session.id())
            .take(limit)
            .collect();

        debug!(
            "Found {} related content items across sessions for topic: '{}'",
            filtered.len(),
            topic
        );

        Ok(filtered)
    }

    /// Get statistics about vectorized content
    pub fn get_vectorization_stats(&self) -> HashMap<String, usize> {
        let db_stats = self.vector_db.get_stats();

        let mut stats = HashMap::new();
        stats.insert("total_vectors".to_string(), db_stats.total_vectors);
        stats.insert(
            "memory_usage_mb".to_string(),
            db_stats.memory_usage_bytes / 1024 / 1024,
        );
        stats.insert(
            "embedding_dimension".to_string(),
            self.embedding_engine.embedding_dimension(),
        );

        stats
    }

    /// Check if a session has been vectorized
    pub fn is_session_vectorized(&self, session_id: uuid::Uuid) -> bool {
        // Check if session has UPDATE embeddings (not just entities)
        // This ensures auto-vectorization runs even if only entities are vectorized
        self.vector_db
            .has_session_update_embeddings(&session_id.to_string())
    }

    /// Count embeddings for a specific session
    pub fn count_session_embeddings(&self, session_id: uuid::Uuid) -> usize {
        self.vector_db
            .count_session_embeddings(&session_id.to_string())
    }

    /// Collect updates sequentially (for small sessions)
    fn collect_updates_sequential(
        &self,
        session: &ActiveSession,
    ) -> Result<(Vec<String>, Vec<VectorMetadata>)> {
        let mut texts_to_embed = Vec::new();
        let mut metadata_list = Vec::new();

        // Process hot context
        for update in &session.hot_context.iter() {
            let raw_text = Self::extract_text_from_update(update);
            if self.should_vectorize_text(&raw_text) {
                let content_type = Self::determine_content_type(update);
                let prepared_text = self.prepare_text_for_vectorization(&raw_text);
                texts_to_embed.push(prepared_text.clone());
                metadata_list.push(VectorMetadata::new(
                    update.id.to_string(),
                    prepared_text,
                    session.id().to_string(),
                    format!("{content_type:?}"),
                ));
            }
        }

        // Process warm context
        for update in &session.warm_context {
            let raw_text = Self::extract_text_from_compressed_update(update);
            if self.should_vectorize_text(&raw_text) {
                let content_type = Self::determine_content_type(&update.update);
                let prepared_text = self.prepare_text_for_vectorization(&raw_text);
                texts_to_embed.push(prepared_text.clone());
                metadata_list.push(VectorMetadata::new(
                    update.update.id.to_string(),
                    prepared_text,
                    session.id().to_string(),
                    format!("{content_type:?}"),
                ));
            }
        }

        Ok((texts_to_embed, metadata_list))
    }

    /// Collect updates in parallel (for large sessions)
    fn collect_updates_parallel(
        &self,
        session: &ActiveSession,
    ) -> Result<(Vec<String>, Vec<VectorMetadata>)> {
        // Process hot context in parallel
        let hot_results: Vec<(String, VectorMetadata)> = session
            .hot_context
            .iter()
            .par_iter()
            .filter_map(|update| {
                let raw_text = Self::extract_text_from_update(update);
                if self.should_vectorize_text(&raw_text) {
                    let content_type = Self::determine_content_type(update);
                    let prepared_text = self.prepare_text_for_vectorization(&raw_text);
                    let metadata = VectorMetadata::new(
                        update.id.to_string(),
                        prepared_text.clone(),
                        session.id().to_string(),
                        format!("{content_type:?}"),
                    );
                    Some((prepared_text, metadata))
                } else {
                    None
                }
            })
            .collect();

        // Process warm context in parallel
        let warm_results: Vec<(String, VectorMetadata)> = session
            .warm_context
            .par_iter()
            .filter_map(|update| {
                let raw_text = Self::extract_text_from_compressed_update(update);
                if self.should_vectorize_text(&raw_text) {
                    let content_type = Self::determine_content_type(&update.update);
                    let prepared_text = self.prepare_text_for_vectorization(&raw_text);
                    let metadata = VectorMetadata::new(
                        update.update.id.to_string(),
                        prepared_text.clone(),
                        session.id().to_string(),
                        format!("{content_type:?}"),
                    );
                    Some((prepared_text, metadata))
                } else {
                    None
                }
            })
            .collect();

        // Combine results
        let mut texts_to_embed = Vec::with_capacity(hot_results.len() + warm_results.len());
        let mut metadata_list = Vec::with_capacity(hot_results.len() + warm_results.len());

        for (text, metadata) in hot_results.into_iter().chain(warm_results.into_iter()) {
            texts_to_embed.push(text);
            metadata_list.push(metadata);
        }

        Ok((texts_to_embed, metadata_list))
    }

    /// Extract text content from a context update
    fn extract_text_from_update(update: &ContextUpdate) -> String {
        let mut text_parts = Vec::new();

        // ALWAYS include title (question) for all update types
        // For QA updates: Including the question is CRITICAL for semantic search
        // Users typically search with questions, not answers, so we need question text
        // in the embedding to match query embeddings effectively
        let title = update.content.title.clone();
        let description = update.content.description.clone();

        tracing::debug!("extract_text_from_update: title='{}', description='{}'",
            &title[..title.len().min(50)],
            &description[..description.len().min(50)]);

        text_parts.push(title);
        text_parts.push(description);

        // Add details
        for detail in &update.content.details {
            text_parts.push(detail.clone());
        }

        // Add examples
        for example in &update.content.examples {
            text_parts.push(example.clone());
        }

        // Add implications
        for implication in &update.content.implications {
            text_parts.push(implication.clone());
        }

        // Add code reference if available
        if let Some(code_ref) = &update.related_code {
            text_parts.push(code_ref.code_snippet.clone());
            text_parts.push(code_ref.file_path.clone());
        }

        text_parts.join(" ")
    }

    /// Extract text content from a compressed update
    fn extract_text_from_compressed_update(
        update: &crate::session::active_session::CompressedUpdate,
    ) -> String {
        Self::extract_text_from_update(&update.update)
    }

    /// Determine content type from context update
    const fn determine_content_type(update: &ContextUpdate) -> ContentType {
        match &update.update_type {
            crate::core::context_update::UpdateType::QuestionAnswered => ContentType::UserMessage,
            crate::core::context_update::UpdateType::ProblemSolved => ContentType::ProblemSolution,
            crate::core::context_update::UpdateType::CodeChanged => ContentType::CodeSnippet,
            crate::core::context_update::UpdateType::DecisionMade => ContentType::DecisionPoint,
            crate::core::context_update::UpdateType::ConceptDefined
            | crate::core::context_update::UpdateType::RequirementAdded => {
                ContentType::UpdateContent
            }
        }
    }

    /// Check if text should be vectorized (with smart summarization for long texts)
    fn should_vectorize_text(&self, text: &str) -> bool {
        let len = text.trim().len();
        len >= self.config.min_text_length
        // Note: No upper limit check - long texts will be summarized
    }

    /// Prepare text for vectorization, applying smart summarization if needed
    fn prepare_text_for_vectorization(&self, text: &str) -> String {
        let len = text.trim().len();

        if len <= self.config.max_text_length {
            // Text is within limits, use as-is
            return text.to_string();
        }

        // Text is too long, apply smart summarization
        debug!(
            "Text length {} exceeds max {}, applying smart summarization",
            len, self.config.max_text_length
        );

        Self::extract_key_points(text, self.config.max_text_length)
    }

    /// Extract key points from long text using smart heuristics
    ///
    /// Strategy:
    /// 1. Preserve Title and Description (most important context)
    /// 2. Extract code snippets (technical details)
    /// 3. Extract bullet points and numbered lists (structured info)
    /// 4. Include beginning (context) and end (conclusion)
    /// 5. Preserve key phrases and technical terms
    fn extract_key_points(text: &str, max_length: usize) -> String {
        let mut parts = Vec::new();
        let mut current_length = 0;

        // Helper function to add part if it fits
        let add_part = |parts: &mut Vec<String>, current_len: &mut usize, part: &str| -> bool {
            let part_len = part.len();
            if *current_len + part_len + 3 <= max_length {  // +3 for " | "
                parts.push(part.to_string());
                *current_len += part_len + 3;
                true
            } else {
                false
            }
        };

        // 1. Always preserve Title and Description (highest priority)
        for line in text.lines() {
            if (line.starts_with("Title:") || line.starts_with("Description:"))
                && !add_part(&mut parts, &mut current_length, line) {
                    // Title/Description alone exceeds max_length - truncate it
                    let truncated = &line.chars().take(max_length - 20).collect::<String>();
                    parts.push(format!("{}...", truncated));
                    return parts.join(" | ");
                }
        }

        // 2. Extract code snippets (technical information is valuable)
        let code_blocks: Vec<&str> = text
            .split("Code:")
            .skip(1)  // Skip first part (before "Code:")
            .filter_map(|segment| {
                // Extract first line after "Code:"
                segment.split('|').next().map(|s| s.trim())
            })
            .collect();

        for code in code_blocks.iter().take(2) {  // Max 2 code snippets
            if !add_part(&mut parts, &mut current_length, &format!("Code: {}", code)) {
                break;
            }
        }

        // 3. Extract bullet points and key technical terms
        // Look for patterns like: "- item", "* item", "1. item", "Performance:", "Algorithm:"
        let mut key_lines = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('-')
                || trimmed.starts_with('*')
                || trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
                || trimmed.contains("Performance:")
                || trimmed.contains("Algorithm:")
                || trimmed.contains("O(")  // Big-O notation
                || trimmed.contains("speedup")
                || trimmed.contains("optimization")
            {
                key_lines.push(trimmed);
            }
        }

        for line in key_lines.iter().take(5) {  // Max 5 key lines
            if !add_part(&mut parts, &mut current_length, line) {
                break;
            }
        }

        // 4. If we still have space, add beginning context
        if current_length < max_length * 3 / 4 {
            let intro_budget = (max_length - current_length).min(300);
            if intro_budget > 50 {
                let intro: String = text.chars().take(intro_budget).collect();
                if let Some(last_space) = intro.rfind(' ') {
                    let _ = add_part(&mut parts, &mut current_length, &format!("Context: {}...", &intro[..last_space]));
                }
            }
        }

        // 5. Add metadata about summarization
        if current_length < max_length - 50 {
            let _ = add_part(&mut parts, &mut current_length, &format!("[Summarized from {} chars]", text.len()));
        }

        if parts.is_empty() {
            // Fallback: just truncate at word boundary
            let truncated: String = text.chars().take(max_length - 20).collect();
            if let Some(last_space) = truncated.rfind(' ') {
                format!("{}... [Truncated]", &truncated[..last_space])
            } else {
                format!("{}... [Truncated]", truncated)
            }
        } else {
            parts.join(" | ")
        }
    }

    /// Parse content type from string
    fn parse_content_type(content_type_str: &str) -> ContentType {
        match content_type_str {
            "EntityDescription" => ContentType::EntityDescription,
            "UserMessage" => ContentType::UserMessage,
            "DecisionPoint" => ContentType::DecisionPoint,
            "CodeSnippet" => ContentType::CodeSnippet,
            "ProblemSolution" => ContentType::ProblemSolution,
            "SessionMetadata" => ContentType::SessionMetadata,
            _ => ContentType::UpdateContent,
        }
    }
}
