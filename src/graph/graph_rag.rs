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

//! Graph-RAG: Graph-enhanced Retrieval Augmented Generation
//!
//! This module provides graph-based enrichment for semantic search results.
//! It uses the entity graph to:
//! - Expand query context with related entities
//! - Find structural connections between search results
//! - Add causal chains and relationship paths
//!
//! # Example
//! ```ignore
//! let config = GraphRagConfig::default();
//! let enricher = GraphRagEnricher::new(config);
//! let enriched = enricher.enrich_results(&graph, query, results).await;
//! ```

use crate::core::context_update::RelationType;
use crate::graph::entity_graph::SimpleEntityGraph;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for Graph-RAG enrichment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagConfig {
    /// Enable/disable graph enrichment
    pub enabled: bool,

    /// Maximum traversal depth (default: 2)
    pub max_depth: usize,

    /// Maximum relations to include per entity (default: 10)
    pub max_relations_per_entity: usize,

    /// Minimum entities in graph to enable enrichment (default: 5)
    pub min_graph_size: usize,

    /// Timeout for graph operations in milliseconds (default: 100)
    pub timeout_ms: u64,

    /// Cache TTL in seconds (default: 300 = 5 minutes)
    pub cache_ttl_secs: u64,

    /// Relation types priority (higher index = lower priority)
    pub relation_priority: Vec<RelationType>,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 2,
            max_relations_per_entity: 10,
            min_graph_size: 5,
            timeout_ms: 100,
            cache_ttl_secs: 300,
            relation_priority: vec![
                RelationType::CausedBy,     // Most useful for debugging
                RelationType::DependsOn,    // Architectural dependency
                RelationType::Implements,   // Implementation relationship
                RelationType::LeadsTo,      // Flow relationship
                RelationType::Solves,       // Problem-solution relationship
                RelationType::RelatedTo,    // General (lowest priority)
            ],
        }
    }
}

/// Cached relation data with expiry
#[derive(Debug, Clone)]
struct CachedRelations {
    relations: Vec<RelationInfo>,
    cached_at: Instant,
}

/// Information about a single relation
#[derive(Debug, Clone)]
pub struct RelationInfo {
    pub target_entity: String,
    pub relation_type: RelationType,
    pub context: String,
    pub depth: usize,
}

/// Result of graph enrichment for a single search result
#[derive(Debug, Clone)]
pub struct GraphEnrichment {
    /// Related entities found via graph traversal
    pub related_entities: Vec<RelationInfo>,

    /// Paths found to query entities
    pub paths_to_query: Vec<Vec<String>>,

    /// Formatted context string to append
    pub context_string: String,
}

/// Global insights from graph analysis
#[derive(Debug, Clone, Default)]
pub struct GlobalGraphInsights {
    /// Central entities related to query
    pub query_entity_map: Vec<(String, Vec<String>)>,

    /// Cross-result connections
    pub structural_insights: Vec<String>,

    /// Formatted string for prepending to results
    pub formatted: String,
}

/// Graph-RAG enricher that adds graph context to search results
pub struct GraphRagEnricher {
    config: GraphRagConfig,
    /// Cache: entity_name -> (relations, cached_at)
    relation_cache: Arc<DashMap<String, CachedRelations>>,
}

impl GraphRagEnricher {
    /// Create a new enricher with the given configuration
    pub fn new(config: GraphRagConfig) -> Self {
        Self {
            config,
            relation_cache: Arc::new(DashMap::new()),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(GraphRagConfig::default())
    }

    /// Check if graph is large enough for enrichment
    pub fn should_enrich(&self, graph: &SimpleEntityGraph) -> bool {
        if !self.config.enabled {
            return false;
        }
        graph.entity_count() >= self.config.min_graph_size
    }

    /// Extract entities from text that exist in the graph
    pub fn extract_graph_entities(&self, graph: &SimpleEntityGraph, text: &str) -> Vec<String> {
        let mut found = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for word in words {
            // Clean punctuation
            let clean = word
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');

            if clean.len() < 2 {
                continue;
            }

            // Check exact match (case-insensitive)
            let lower = clean.to_lowercase();
            if graph.has_entity(&lower) {
                found.push(lower);
                continue;
            }

            // Check original case
            if graph.has_entity(clean) {
                found.push(clean.to_string());
            }
        }

        // Deduplicate while preserving order
        let mut seen = HashSet::new();
        found.retain(|e| seen.insert(e.clone()));
        found
    }

    /// Find related entities with depth limit and cycle detection
    pub fn find_related_with_depth(
        &self,
        graph: &SimpleEntityGraph,
        start_entity: &str,
        max_depth: usize,
    ) -> Vec<RelationInfo> {
        // Check cache first
        if let Some(cached) = self.get_cached_relations(start_entity) {
            return cached;
        }

        let mut results = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start_entity.to_string());

        self.traverse_relations(
            graph,
            start_entity,
            1,
            max_depth,
            &mut visited,
            &mut results,
        );

        // Sort by priority (relation type) then by depth
        results.sort_by(|a, b| {
            let priority_a = self.get_relation_priority(&a.relation_type);
            let priority_b = self.get_relation_priority(&b.relation_type);
            priority_a.cmp(&priority_b).then(a.depth.cmp(&b.depth))
        });

        // Limit results
        results.truncate(self.config.max_relations_per_entity);

        // Cache the results
        self.cache_relations(start_entity, results.clone());

        results
    }

    /// Recursive traversal with cycle detection
    fn traverse_relations(
        &self,
        graph: &SimpleEntityGraph,
        entity: &str,
        current_depth: usize,
        max_depth: usize,
        visited: &mut HashSet<String>,
        results: &mut Vec<RelationInfo>,
    ) {
        if current_depth > max_depth {
            return;
        }

        // Get direct relations from graph
        let relations = graph.get_entity_relationships(entity);

        for (target, rel_type, context) in relations {
            if visited.contains(&target) {
                continue; // Cycle detected, skip
            }

            results.push(RelationInfo {
                target_entity: target.clone(),
                relation_type: rel_type.clone(),
                context: context.clone(),
                depth: current_depth,
            });

            // Mark as visited and recurse
            visited.insert(target.clone());
            self.traverse_relations(
                graph,
                &target,
                current_depth + 1,
                max_depth,
                visited,
                results,
            );
        }
    }

    /// Get cached relations if not expired
    fn get_cached_relations(&self, entity: &str) -> Option<Vec<RelationInfo>> {
        let cached = self.relation_cache.get(entity)?;
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);

        if cached.cached_at.elapsed() < ttl {
            Some(cached.relations.clone())
        } else {
            // Expired, remove from cache
            drop(cached);
            self.relation_cache.remove(entity);
            None
        }
    }

    /// Cache relations for an entity
    fn cache_relations(&self, entity: &str, relations: Vec<RelationInfo>) {
        self.relation_cache.insert(
            entity.to_string(),
            CachedRelations {
                relations,
                cached_at: Instant::now(),
            },
        );
    }

    /// Get priority for a relation type (lower = higher priority)
    fn get_relation_priority(&self, rel_type: &RelationType) -> usize {
        self.config
            .relation_priority
            .iter()
            .position(|r| r == rel_type)
            .unwrap_or(usize::MAX)
    }

    /// Analyze query and build global insights
    pub fn analyze_query(
        &self,
        graph: &SimpleEntityGraph,
        query_entities: &[String],
    ) -> GlobalGraphInsights {
        let mut insights = GlobalGraphInsights::default();

        for entity in query_entities {
            let relations = self.find_related_with_depth(graph, entity, self.config.max_depth);
            if !relations.is_empty() {
                let related: Vec<String> = relations
                    .iter()
                    .take(5)
                    .map(|r| r.target_entity.clone())
                    .collect();
                insights.query_entity_map.push((entity.clone(), related));
            }
        }

        // Format the insights
        if !insights.query_entity_map.is_empty() {
            let mut formatted = String::from("\n[System Knowledge Map]:\n");
            for (entity, related) in &insights.query_entity_map {
                formatted.push_str(&format!("• {} → {}\n", entity, related.join(", ")));
            }
            insights.formatted = formatted;
        }

        insights
    }

    /// Enrich a single search result with graph context
    pub fn enrich_result(
        &self,
        graph: &SimpleEntityGraph,
        content: &str,
        query_entities: &[String],
    ) -> GraphEnrichment {
        let result_entities = self.extract_graph_entities(graph, content);
        let mut related = Vec::new();
        let mut paths = Vec::new();

        // Find relations for entities in this result
        for entity in result_entities.iter().take(3) {
            let entity_relations = self.find_related_with_depth(graph, entity, 1);
            related.extend(entity_relations);
        }

        // Find paths to query entities
        for result_entity in result_entities.iter().take(2) {
            for query_entity in query_entities.iter().take(2) {
                if result_entity != query_entity {
                    if let Some(path) = graph.find_shortest_path(result_entity, query_entity) {
                        if path.len() > 2 && path.len() <= 5 {
                            paths.push(path);
                        }
                    }
                }
            }
        }

        // Deduplicate related entities
        let mut seen = HashSet::new();
        related.retain(|r| seen.insert(r.target_entity.clone()));
        related.truncate(self.config.max_relations_per_entity);

        // Format context string
        let context_string = self.format_enrichment(&related, &paths);

        GraphEnrichment {
            related_entities: related,
            paths_to_query: paths,
            context_string,
        }
    }

    /// Find structural insights between top results
    pub fn find_cross_result_insights(
        &self,
        graph: &SimpleEntityGraph,
        result_entities: &[Vec<String>],
    ) -> Vec<String> {
        let mut insights = Vec::new();

        if result_entities.len() < 2 {
            return insights;
        }

        // Check for paths between entities from different results
        for i in 0..result_entities.len().min(3) {
            for j in (i + 1)..result_entities.len().min(3) {
                if let (Some(e1), Some(e2)) = (
                    result_entities[i].first(),
                    result_entities[j].first(),
                ) {
                    if e1 != e2 {
                        if let Some(path) = graph.find_shortest_path(e1, e2) {
                            if path.len() > 2 && path.len() <= 5 {
                                insights.push(format!(
                                    "Connection: {} → {}",
                                    path.join(" → "),
                                    ""
                                ).trim_end_matches(" → ").to_string());
                            }
                        }
                    }
                }
            }
        }

        insights
    }

    /// Format enrichment data as a readable string
    fn format_enrichment(&self, relations: &[RelationInfo], paths: &[Vec<String>]) -> String {
        if relations.is_empty() && paths.is_empty() {
            return String::new();
        }

        let mut output = String::from("\n───────────────────────────\n📊 Graph Context:\n");

        // Add relations grouped by type
        if !relations.is_empty() {
            let mut by_type: std::collections::HashMap<&RelationType, Vec<&str>> =
                std::collections::HashMap::new();

            for rel in relations.iter().take(8) {
                by_type
                    .entry(&rel.relation_type)
                    .or_default()
                    .push(&rel.target_entity);
            }

            for (rel_type, entities) in by_type {
                output.push_str(&format!(
                    "• {:?}: {}\n",
                    rel_type,
                    entities.join(", ")
                ));
            }
        }

        // Add paths
        for path in paths.iter().take(2) {
            output.push_str(&format!("• Path: {}\n", path.join(" → ")));
        }

        output.push_str("───────────────────────────");
        output
    }

    /// Clear expired cache entries
    pub fn cleanup_cache(&self) {
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);
        self.relation_cache.retain(|_, cached| {
            cached.cached_at.elapsed() < ttl
        });
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let total = self.relation_cache.len();
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);
        let valid = self.relation_cache
            .iter()
            .filter(|e| e.cached_at.elapsed() < ttl)
            .count();
        (valid, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context_update::EntityType;

    fn create_test_graph() -> SimpleEntityGraph {
        let mut graph = SimpleEntityGraph::new();
        let now = chrono::Utc::now();

        // Add entities using add_or_update_entity
        graph.add_or_update_entity(
            "payment".to_string(),
            EntityType::Concept,
            now,
            "Payment processing",
        );

        graph.add_or_update_entity(
            "database".to_string(),
            EntityType::Technology,
            now,
            "Database system",
        );

        graph.add_or_update_entity(
            "timeout".to_string(),
            EntityType::Concept,
            now,
            "Connection timeout",
        );

        graph.add_or_update_entity(
            "auth".to_string(),
            EntityType::Concept,
            now,
            "Authentication",
        );

        graph.add_or_update_entity(
            "user".to_string(),
            EntityType::Concept,
            now,
            "User entity",
        );

        // Add relationships
        use crate::core::context_update::EntityRelationship;

        graph.add_relationship(EntityRelationship {
            from_entity: "payment".to_string(),
            to_entity: "database".to_string(),
            relation_type: RelationType::DependsOn,
            context: "Payment depends on database".to_string(),
        });

        graph.add_relationship(EntityRelationship {
            from_entity: "database".to_string(),
            to_entity: "timeout".to_string(),
            relation_type: RelationType::CausedBy,
            context: "Database can cause timeout".to_string(),
        });

        graph.add_relationship(EntityRelationship {
            from_entity: "auth".to_string(),
            to_entity: "user".to_string(),
            relation_type: RelationType::LeadsTo,
            context: "Auth leads to user".to_string(),
        });

        graph.add_relationship(EntityRelationship {
            from_entity: "payment".to_string(),
            to_entity: "auth".to_string(),
            relation_type: RelationType::DependsOn,
            context: "Payment requires auth".to_string(),
        });

        graph
    }

    #[test]
    fn test_should_enrich() {
        let enricher = GraphRagEnricher::with_defaults();
        let graph = create_test_graph();

        assert!(enricher.should_enrich(&graph));

        // Empty graph should not enrich
        let empty_graph = SimpleEntityGraph::new();
        assert!(!enricher.should_enrich(&empty_graph));
    }

    #[test]
    fn test_extract_graph_entities() {
        let enricher = GraphRagEnricher::with_defaults();
        let graph = create_test_graph();

        let text = "The payment system had a database timeout";
        let entities = enricher.extract_graph_entities(&graph, text);

        assert!(entities.contains(&"payment".to_string()));
        assert!(entities.contains(&"database".to_string()));
        assert!(entities.contains(&"timeout".to_string()));
    }

    #[test]
    fn test_find_related_with_depth() {
        let enricher = GraphRagEnricher::with_defaults();
        let graph = create_test_graph();

        let related = enricher.find_related_with_depth(&graph, "payment", 2);

        // Should find database (depth 1) and timeout (depth 2)
        let targets: Vec<_> = related.iter().map(|r| &r.target_entity).collect();
        assert!(targets.contains(&&"database".to_string()));
        assert!(targets.contains(&&"auth".to_string()));
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = SimpleEntityGraph::new();
        let now = chrono::Utc::now();

        // Create a cycle: A -> B -> C -> A
        for name in ["a", "b", "c"] {
            graph.add_or_update_entity(
                name.to_string(),
                EntityType::Concept,
                now,
                "Test entity",
            );
        }

        use crate::core::context_update::EntityRelationship;
        graph.add_relationship(EntityRelationship {
            from_entity: "a".to_string(),
            to_entity: "b".to_string(),
            relation_type: RelationType::RelatedTo,
            context: String::new(),
        });
        graph.add_relationship(EntityRelationship {
            from_entity: "b".to_string(),
            to_entity: "c".to_string(),
            relation_type: RelationType::RelatedTo,
            context: String::new(),
        });
        graph.add_relationship(EntityRelationship {
            from_entity: "c".to_string(),
            to_entity: "a".to_string(),
            relation_type: RelationType::RelatedTo,
            context: String::new(),
        });

        let enricher = GraphRagEnricher::with_defaults();
        let related = enricher.find_related_with_depth(&graph, "a", 5);

        // Should not infinite loop, should find b and c exactly once
        assert_eq!(related.len(), 2);
    }

    #[test]
    fn test_cache_works() {
        let enricher = GraphRagEnricher::with_defaults();
        let graph = create_test_graph();

        // First call populates cache
        let related1 = enricher.find_related_with_depth(&graph, "payment", 2);

        // Second call should use cache
        let related2 = enricher.find_related_with_depth(&graph, "payment", 2);

        assert_eq!(related1.len(), related2.len());

        let (valid, total) = enricher.cache_stats();
        assert!(valid > 0);
        assert_eq!(valid, total);
    }

    #[test]
    fn test_enrich_result() {
        let enricher = GraphRagEnricher::with_defaults();
        let graph = create_test_graph();

        let content = "Payment processing failed due to database issues";
        let query_entities = vec!["timeout".to_string()];

        let enrichment = enricher.enrich_result(&graph, content, &query_entities);

        assert!(!enrichment.related_entities.is_empty());
        assert!(!enrichment.context_string.is_empty());
        assert!(enrichment.context_string.contains("Graph Context"));
    }
}
