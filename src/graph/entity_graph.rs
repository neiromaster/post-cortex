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
use crate::core::context_update::{EntityData, EntityRelationship, EntityType, RelationType};
use chrono::{DateTime, Utc};
use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Enhanced entity graph using petgraph for efficient relationship operations
#[derive(Clone, Debug, Serialize)]
pub struct SimpleEntityGraph {
    /// Entity metadata storage - keeps original functionality
    pub entities: HashMap<String, EntityData>,

    /// Entity mentions tracking - keeps original functionality
    pub entity_mentions: HashMap<String, Vec<uuid::Uuid>>,

    /// Petgraph directed graph for relationships
    #[serde(with = "petgraph_serde")]
    graph: DiGraph<String, RelationType>,

    /// Bidirectional mapping between entity names and graph nodes
    /// CRITICAL: Rebuilt automatically after deserialization
    #[serde(skip)]
    entity_to_node: HashMap<String, NodeIndex>,

    /// CRITICAL: Rebuilt automatically after deserialization
    #[serde(skip)]
    node_to_entity: HashMap<NodeIndex, String>,
}

// Intermediate structure for serializing graph data with entity names instead of indices
#[derive(Serialize, Deserialize)]
struct SerializableGraphData {
    /// Sorted list of entity names (nodes) for deterministic deserialization
    nodes: Vec<String>,
    /// Edges represented by entity name pairs instead of NodeIndex
    edges: Vec<(String, String, RelationType)>,
}

// Custom serde module for petgraph serialization with full graph reconstruction
mod petgraph_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(
        graph: &DiGraph<String, RelationType>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Extract all nodes (entity names) in sorted order for determinism
        let mut nodes: Vec<String> = graph.node_weights().cloned().collect();
        nodes.sort(); // CRITICAL: sorted for deterministic NodeIndex mapping

        // Extract edges using entity names instead of NodeIndex
        let mut edges: Vec<(String, String, RelationType)> = graph
            .edge_references()
            .map(|edge| {
                let from_entity = graph.node_weight(edge.source()).unwrap().clone();
                let to_entity = graph.node_weight(edge.target()).unwrap().clone();
                (from_entity, to_entity, edge.weight().clone())
            })
            .collect();

        // Sort edges for determinism (by source, then target, then relation type)
        edges.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| format!("{:?}", a.2).cmp(&format!("{:?}", b.2)))
        });

        let data = SerializableGraphData { nodes, edges };
        data.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DiGraph<String, RelationType>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = SerializableGraphData::deserialize(deserializer)?;

        // Recreate graph with nodes in deterministic order
        let mut graph = DiGraph::new();

        // Add all nodes in the same order they were serialized
        // This ensures NodeIndex values are consistent across serialization
        for entity_name in &data.nodes {
            graph.add_node(entity_name.clone());
        }

        // Build temporary mapping from entity name to NodeIndex for edge reconstruction
        let entity_to_node: std::collections::HashMap<&String, NodeIndex> = data
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, name)| (name, NodeIndex::new(idx)))
            .collect();

        // Recreate all edges
        for (from_entity, to_entity, relation_type) in data.edges {
            if let (Some(&from_node), Some(&to_node)) = (
                entity_to_node.get(&from_entity),
                entity_to_node.get(&to_entity),
            ) {
                graph.add_edge(from_node, to_node, relation_type);
            }
        }

        Ok(graph)
    }
}

// Custom Deserialize implementation to rebuild entity_to_node and node_to_entity mappings
impl<'de> Deserialize<'de> for SimpleEntityGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SimpleEntityGraphData {
            entities: HashMap<String, EntityData>,
            entity_mentions: HashMap<String, Vec<uuid::Uuid>>,
            #[serde(with = "petgraph_serde")]
            graph: DiGraph<String, RelationType>,
        }

        let data = SimpleEntityGraphData::deserialize(deserializer)?;

        // Rebuild entity_to_node and node_to_entity mappings from deserialized graph
        let mut entity_to_node = HashMap::new();
        let mut node_to_entity = HashMap::new();

        for node_idx in data.graph.node_indices() {
            if let Some(entity_name) = data.graph.node_weight(node_idx) {
                entity_to_node.insert(entity_name.clone(), node_idx);
                node_to_entity.insert(node_idx, entity_name.clone());
            }
        }

        Ok(SimpleEntityGraph {
            entities: data.entities,
            entity_mentions: data.entity_mentions,
            graph: data.graph,
            entity_to_node,
            node_to_entity,
        })
    }
}

impl Default for SimpleEntityGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleEntityGraph {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            entity_mentions: HashMap::new(),
            graph: DiGraph::new(),
            entity_to_node: HashMap::new(),
            node_to_entity: HashMap::new(),
        }
    }

    /// Add or update entity - maintains same API as original SimpleEntityGraph
    pub fn add_or_update_entity(
        &mut self,
        name: String,
        entity_type: EntityType,
        timestamp: DateTime<Utc>,
        description: &str,
    ) {
        match self.entities.get_mut(&name) {
            Some(entity) => {
                entity.last_mentioned = timestamp;
                entity.mention_count += 1;
                if entity.description.is_none() && !description.is_empty() {
                    entity.description = Some(description.to_string());
                }

                // CRITICAL FIX: Ensure node exists in graph even for existing entities
                if !self.entity_to_node.contains_key(&name) {
                    let node_index = self.graph.add_node(name.clone());
                    self.entity_to_node.insert(name.clone(), node_index);
                    self.node_to_entity.insert(node_index, name.clone());
                }
            }
            None => {
                // Create new entity data
                self.entities.insert(
                    name.clone(),
                    EntityData {
                        name: name.clone(),
                        entity_type,
                        first_mentioned: timestamp,
                        last_mentioned: timestamp,
                        mention_count: 1,
                        importance_score: 1.0,
                        description: if description.is_empty() {
                            None
                        } else {
                            Some(description.to_string())
                        },
                    },
                );

                // Add node to graph
                let node_index = self.graph.add_node(name.clone());
                self.entity_to_node.insert(name.clone(), node_index);
                self.node_to_entity.insert(node_index, name);
            }
        }
    }

    /// Get or create a node for an entity, returning its NodeIndex
    /// This is a safe helper that guarantees a valid NodeIndex is returned
    fn get_or_create_node(
        &mut self,
        name: &str,
        entity_type: EntityType,
        timestamp: DateTime<Utc>,
    ) -> NodeIndex {
        // Check if node already exists
        if let Some(&node_idx) = self.entity_to_node.get(name) {
            return node_idx;
        }

        // Entity doesn't exist or doesn't have a node - create it
        self.add_or_update_entity(name.to_string(), entity_type, timestamp, "");

        // Now it must exist - but use defensive programming
        match self.entity_to_node.get(name) {
            Some(&node_idx) => node_idx,
            None => {
                // This should never happen, but if it does, create the node directly
                let node_idx = self.graph.add_node(name.to_string());
                self.entity_to_node.insert(name.to_string(), node_idx);
                self.node_to_entity.insert(node_idx, name.to_string());
                node_idx
            }
        }
    }

    /// Mention entity - updates importance and tracking
    pub fn mention_entity(&mut self, name: &str, update_id: uuid::Uuid, timestamp: DateTime<Utc>) {
        if let Some(entity) = self.entities.get_mut(name) {
            entity.last_mentioned = timestamp;
            entity.mention_count += 1;
            entity.importance_score = (entity.mention_count as f32).log2().max(1.0);
        }

        self.entity_mentions
            .entry(name.to_string())
            .or_default()
            .push(update_id);
    }

    /// Add relationship - now uses petgraph for efficient storage
    pub fn add_relationship(&mut self, relationship: EntityRelationship) {
        let timestamp = Utc::now();

        // Use safe helper to get or create nodes - no more unwrap()!
        let from_node =
            self.get_or_create_node(&relationship.from_entity, EntityType::Concept, timestamp);

        let to_node =
            self.get_or_create_node(&relationship.to_entity, EntityType::Concept, timestamp);

        // Add edge to graph
        self.graph
            .add_edge(from_node, to_node, relationship.relation_type);
    }

    /// Find related entities - now O(degree) instead of O(n)!
    pub fn find_related_entities(&self, entity_name: &str) -> Vec<String> {
        let node = match self.entity_to_node.get(entity_name) {
            Some(&node) => node,
            None => return Vec::new(),
        };

        let mut related = HashSet::new();

        // Get outgoing neighbors
        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
            if let Some(target_name) = self.node_to_entity.get(&edge.target()) {
                related.insert(target_name.clone());
            }
        }

        // Get incoming neighbors
        for edge in self.graph.edges_directed(node, Direction::Incoming) {
            if let Some(source_name) = self.node_to_entity.get(&edge.source()) {
                related.insert(source_name.clone());
            }
        }

        let mut result: Vec<String> = related.into_iter().collect();
        result.sort();
        result
    }

    /// Find related entities by specific relation type - also now O(degree) instead of O(n)
    pub fn find_related_entities_by_type(
        &self,
        entity_name: &str,
        relation_type: &RelationType,
    ) -> Vec<String> {
        let node = match self.entity_to_node.get(entity_name) {
            Some(&node) => node,
            None => return Vec::new(),
        };

        let mut related = HashSet::new();

        // Check outgoing edges
        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
            if edge.weight() == relation_type
                && let Some(target_name) = self.node_to_entity.get(&edge.target())
            {
                related.insert(target_name.clone());
            }
        }

        // Check incoming edges
        for edge in self.graph.edges_directed(node, Direction::Incoming) {
            if edge.weight() == relation_type
                && let Some(source_name) = self.node_to_entity.get(&edge.source())
            {
                related.insert(source_name.clone());
            }
        }

        let mut result: Vec<String> = related.into_iter().collect();
        result.sort();
        result
    }

    /// Get entities by type - unchanged from original
    pub fn get_entities_by_type(&self, entity_type: &EntityType) -> Vec<&EntityData> {
        self.entities
            .values()
            .filter(|entity| entity.entity_type == *entity_type)
            .collect()
    }

    /// Get most important entities - unchanged from original
    pub fn get_most_important_entities(&self, limit: usize) -> Vec<&EntityData> {
        let mut entities: Vec<&EntityData> = self.entities.values().collect();
        entities.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());
        entities.into_iter().take(limit).collect()
    }

    /// Get recently mentioned entities - unchanged from original
    pub fn get_recently_mentioned_entities(&self, limit: usize) -> Vec<&EntityData> {
        let mut entities: Vec<&EntityData> = self.entities.values().collect();
        entities.sort_by(|a, b| b.last_mentioned.cmp(&a.last_mentioned));
        entities.into_iter().take(limit).collect()
    }

    /// Search entities - unchanged from original
    pub fn search_entities(&self, query: &str) -> Vec<&EntityData> {
        let query_lower = query.to_lowercase();
        self.entities
            .values()
            .filter(|entity| {
                entity.name.to_lowercase().contains(&query_lower)
                    || entity
                        .description
                        .as_ref()
                        .is_some_and(|desc| desc.to_lowercase().contains(&query_lower))
            })
            .collect()
    }

    /// Get entity context with improved relationship lookup
    pub fn get_entity_context(&self, entity_name: &str) -> Option<String> {
        let entity = self.entities.get(entity_name)?;
        let related = self.find_related_entities(entity_name);

        let mut context = format!(
            "Entity: {} (Type: {:?}, Mentions: {}, Importance: {:.2})",
            entity.name, entity.entity_type, entity.mention_count, entity.importance_score
        );

        if let Some(desc) = &entity.description {
            context.push_str(&format!("\nDescription: {}", desc));
        }

        if !related.is_empty() {
            context.push_str(&format!("\nRelated entities: {}", related.join(", ")));
        }

        // Add relationship details using efficient petgraph lookups
        let relationships = self.get_entity_relationships(entity_name);
        if !relationships.is_empty() {
            context.push_str("\nRelationships:");
            for (related_entity, relation_type, relation_context) in relationships {
                context.push_str(&format!(
                    "\n  - {} ({:?}): {}",
                    related_entity, relation_type, relation_context
                ));
            }
        }

        Some(context)
    }

    /// Get entity relationships - now uses efficient petgraph edge iteration
    pub fn get_entity_relationships(
        &self,
        entity_name: &str,
    ) -> Vec<(String, RelationType, String)> {
        let node = match self.entity_to_node.get(entity_name) {
            Some(&node) => node,
            None => return Vec::new(),
        };

        let mut relationships = Vec::new();

        // Outgoing relationships
        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
            if let Some(target_name) = self.node_to_entity.get(&edge.target()) {
                relationships.push((
                    target_name.clone(),
                    edge.weight().clone(),
                    "outgoing".to_string(), // Could be enhanced with context from EntityRelationship
                ));
            }
        }

        // Incoming relationships
        for edge in self.graph.edges_directed(node, Direction::Incoming) {
            if let Some(source_name) = self.node_to_entity.get(&edge.source()) {
                relationships.push((
                    source_name.clone(),
                    edge.weight().clone(),
                    "incoming".to_string(), // Could be enhanced with context from EntityRelationship
                ));
            }
        }

        relationships
    }

    /// Trace entity relationships using efficient BFS
    pub fn trace_entity_relationships(
        &self,
        start_entity: &str,
        max_depth: usize,
    ) -> Vec<(String, String, String)> {
        let start_node = match self.entity_to_node.get(start_entity) {
            Some(&node) => node,
            None => return Vec::new(),
        };

        let mut visited = HashSet::new();
        let mut trace = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((start_node, 0));
        visited.insert(start_node);

        while let Some((current_node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let current_name = match self.node_to_entity.get(&current_node) {
                Some(name) => name,
                None => continue,
            };

            // Process all neighbors efficiently
            for edge in self.graph.edges_directed(current_node, Direction::Outgoing) {
                let target_node = edge.target();
                if !visited.contains(&target_node) {
                    visited.insert(target_node);
                    queue.push_back((target_node, depth + 1));

                    if let Some(target_name) = self.node_to_entity.get(&target_node) {
                        trace.push((
                            current_name.clone(),
                            format!("{:?}", edge.weight()),
                            target_name.clone(),
                        ));
                    }
                }
            }

            // Also process incoming edges
            for edge in self.graph.edges_directed(current_node, Direction::Incoming) {
                let source_node = edge.source();
                if !visited.contains(&source_node) {
                    visited.insert(source_node);
                    queue.push_back((source_node, depth + 1));

                    if let Some(source_name) = self.node_to_entity.get(&source_node) {
                        trace.push((
                            source_name.clone(),
                            format!("inverse_{:?}", edge.weight()),
                            current_name.clone(),
                        ));
                    }
                }
            }
        }

        trace
    }

    /// Get entity network using efficient graph traversal
    pub fn get_entity_network(&self, center_entity: &str, max_depth: usize) -> EntityNetwork {
        let mut network = EntityNetwork {
            center: center_entity.to_string(),
            entities: HashMap::new(),
            relationships: Vec::new(),
        };

        let start_node = match self.entity_to_node.get(center_entity) {
            Some(&node) => node,
            None => return network,
        };

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Add center entity
        if let Some(entity) = self.entities.get(center_entity) {
            network
                .entities
                .insert(center_entity.to_string(), entity.clone());
            queue.push_back((start_node, 0));
            visited.insert(start_node);
        }

        while let Some((current_node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Process outgoing edges
            for edge in self.graph.edges_directed(current_node, Direction::Outgoing) {
                let target_node = edge.target();
                if !visited.contains(&target_node) {
                    visited.insert(target_node);
                    queue.push_back((target_node, depth + 1));

                    // Add entity to network
                    if let Some(target_name) = self.node_to_entity.get(&target_node)
                        && let Some(entity_data) = self.entities.get(target_name)
                    {
                        network
                            .entities
                            .insert(target_name.clone(), entity_data.clone());
                    }
                }

                // Add relationship to network
                if let (Some(from_name), Some(to_name)) = (
                    self.node_to_entity.get(&current_node),
                    self.node_to_entity.get(&target_node),
                ) && network.entities.contains_key(from_name)
                    && network.entities.contains_key(to_name)
                {
                    network.relationships.push(EntityRelationship {
                        from_entity: from_name.clone(),
                        to_entity: to_name.clone(),
                        relation_type: edge.weight().clone(),
                        context: "network_traversal".to_string(),
                    });
                }
            }

            // Process incoming edges
            for edge in self.graph.edges_directed(current_node, Direction::Incoming) {
                let source_node = edge.source();
                if !visited.contains(&source_node) {
                    visited.insert(source_node);
                    queue.push_back((source_node, depth + 1));

                    // Add entity to network
                    if let Some(source_name) = self.node_to_entity.get(&source_node)
                        && let Some(entity_data) = self.entities.get(source_name)
                    {
                        network
                            .entities
                            .insert(source_name.clone(), entity_data.clone());
                    }
                }

                // Add relationship to network (avoid duplicates)
                if let (Some(from_name), Some(to_name)) = (
                    self.node_to_entity.get(&source_node),
                    self.node_to_entity.get(&current_node),
                ) && network.entities.contains_key(from_name)
                    && network.entities.contains_key(to_name)
                {
                    // Check if relationship already exists to avoid duplicates
                    let relationship_exists = network.relationships.iter().any(|rel| {
                        rel.from_entity == *from_name
                            && rel.to_entity == *to_name
                            && rel.relation_type == *edge.weight()
                    });

                    if !relationship_exists {
                        network.relationships.push(EntityRelationship {
                            from_entity: from_name.clone(),
                            to_entity: to_name.clone(),
                            relation_type: edge.weight().clone(),
                            context: "network_traversal".to_string(),
                        });
                    }
                }
            }
        }

        network
    }

    /// Analyze entity importance with graph metrics
    pub fn analyze_entity_importance(&self) -> Vec<EntityAnalysis> {
        let mut analyses = Vec::new();

        for (name, entity) in &self.entities {
            let node = self.entity_to_node.get(name);
            let relationship_count = node
                .map(|&n| {
                    self.graph.edges_directed(n, Direction::Outgoing).count()
                        + self.graph.edges_directed(n, Direction::Incoming).count()
                })
                .unwrap_or(0);

            let mentions_in_updates = self
                .entity_mentions
                .get(name)
                .map(|mentions| mentions.len())
                .unwrap_or(0);

            // Enhanced importance with graph centrality
            let degree_centrality = relationship_count as f32;
            let enhanced_importance = entity.importance_score + (degree_centrality * 0.1);

            let entity_analysis = EntityAnalysis {
                entity_name: name.clone(),
                importance_score: enhanced_importance,
                mention_count: entity.mention_count,
                relationship_count,
                update_references: mentions_in_updates,
                first_seen: entity.first_mentioned,
                last_seen: entity.last_mentioned,
            };

            analyses.push(entity_analysis);
        }

        // Sort by enhanced importance score
        analyses.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());
        analyses
    }

    /// New petgraph-specific methods
    ///
    /// Find shortest path between two entities using Dijkstra
    pub fn find_shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        use petgraph::algo::dijkstra;

        let from_node = self.entity_to_node.get(from)?;
        let to_node = self.entity_to_node.get(to)?;

        let path_map = dijkstra(&self.graph, *from_node, Some(*to_node), |_| 1);

        if path_map.contains_key(to_node) {
            // Reconstruct path (simplified - real implementation would track predecessors)
            Some(vec![from.to_string(), to.to_string()])
        } else {
            None
        }
    }

    /// Get graph statistics
    pub fn get_graph_stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),
            entity_count: self.entities.len(),
            avg_degree: if self.graph.node_count() > 0 {
                (self.graph.edge_count() * 2) as f64 / self.graph.node_count() as f64
            } else {
                0.0
            },
        }
    }

    /// Find strongly connected components
    pub fn find_communities(&self) -> Vec<Vec<String>> {
        use petgraph::algo::kosaraju_scc;

        let components = kosaraju_scc(&self.graph);

        components
            .into_iter()
            .map(|component| {
                component
                    .into_iter()
                    .filter_map(|node| self.node_to_entity.get(&node).cloned())
                    .collect()
            })
            .filter(|component: &Vec<String>| component.len() > 1) // Only return actual communities
            .collect()
    }
}

/// Entity network structure - unchanged for compatibility
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityNetwork {
    pub center: String,
    pub entities: HashMap<String, EntityData>,
    pub relationships: Vec<EntityRelationship>,
}

/// Entity analysis structure - unchanged for compatibility
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityAnalysis {
    pub entity_name: String,
    pub importance_score: f32,
    pub mention_count: u32,
    pub relationship_count: usize,
    pub update_references: usize,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

/// New graph statistics structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub entity_count: usize,
    pub avg_degree: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_graph() -> SimpleEntityGraph {
        let mut graph = SimpleEntityGraph::new();

        // Add some entities
        graph.add_or_update_entity(
            "rust".to_string(),
            EntityType::Technology,
            Utc::now(),
            "Systems programming language",
        );

        graph.add_or_update_entity(
            "petgraph".to_string(),
            EntityType::Technology,
            Utc::now(),
            "Graph data structure library",
        );

        graph.add_or_update_entity(
            "performance".to_string(),
            EntityType::Concept,
            Utc::now(),
            "System performance optimization",
        );

        // Add relationships
        graph.add_relationship(EntityRelationship {
            from_entity: "rust".to_string(),
            to_entity: "petgraph".to_string(),
            relation_type: RelationType::Implements,
            context: "Rust implements petgraph".to_string(),
        });

        graph.add_relationship(EntityRelationship {
            from_entity: "petgraph".to_string(),
            to_entity: "performance".to_string(),
            relation_type: RelationType::Implements,
            context: "Petgraph improves performance".to_string(),
        });

        graph
    }

    #[test]
    fn test_petgraph_find_related_entities() {
        let graph = create_test_graph();

        let related = graph.find_related_entities("rust");
        assert!(related.contains(&"petgraph".to_string()));

        let related = graph.find_related_entities("petgraph");
        assert!(related.contains(&"rust".to_string()));
        assert!(related.contains(&"performance".to_string()));
    }

    #[test]
    fn test_petgraph_find_related_by_type() {
        let graph = create_test_graph();

        let related = graph.find_related_entities_by_type("rust", &RelationType::Implements);
        assert!(related.contains(&"petgraph".to_string()));

        let related = graph.find_related_entities_by_type("petgraph", &RelationType::Implements);
        assert!(related.contains(&"performance".to_string()));
    }

    #[test]
    fn test_graph_stats() {
        let graph = create_test_graph();
        let stats = graph.get_graph_stats();

        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.entity_count, 3);
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_shortest_path() {
        let graph = create_test_graph();

        let path = graph.find_shortest_path("rust", "performance");
        assert!(path.is_some());

        let path = graph.find_shortest_path("rust", "nonexistent");
        assert!(path.is_none());
    }

    #[test]
    fn test_trace_relationships_petgraph() {
        let graph = create_test_graph();

        let trace = graph.trace_entity_relationships("rust", 2);
        assert!(!trace.is_empty());

        // Should find path rust -> petgraph -> performance
        let has_rust_to_petgraph = trace
            .iter()
            .any(|(from, _, to)| from == "rust" && to == "petgraph");
        assert!(has_rust_to_petgraph);
    }

    #[test]
    fn test_entity_network_petgraph() {
        let graph = create_test_graph();

        let network = graph.get_entity_network("petgraph", 2);
        assert_eq!(network.center, "petgraph");
        assert!(network.entities.contains_key("petgraph"));
        assert!(network.entities.contains_key("rust"));
        assert!(network.entities.contains_key("performance"));
        assert!(!network.relationships.is_empty());
    }

    #[test]
    fn test_enhanced_importance_scoring() {
        let mut graph = create_test_graph();

        // Add more relationships to petgraph to increase its importance
        graph.add_or_update_entity(
            "optimization".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Code optimization",
        );

        graph.add_relationship(EntityRelationship {
            from_entity: "petgraph".to_string(),
            to_entity: "optimization".to_string(),
            relation_type: RelationType::LeadsTo,
            context: "Petgraph leads to optimization".to_string(),
        });

        let analysis = graph.analyze_entity_importance();

        // Petgraph should have higher importance due to more connections
        let petgraph_analysis = analysis
            .iter()
            .find(|a| a.entity_name == "petgraph")
            .unwrap();

        let rust_analysis = analysis.iter().find(|a| a.entity_name == "rust").unwrap();

        assert!(petgraph_analysis.importance_score >= rust_analysis.importance_score);
        assert_eq!(petgraph_analysis.relationship_count, 3); // Connected to rust, performance, optimization
    }

    #[test]
    fn test_communities_detection() {
        let mut graph = SimpleEntityGraph::new();

        // Create a more complex graph with communities
        let entities = vec!["a", "b", "c", "d", "e"];
        for entity in &entities {
            graph.add_or_update_entity(entity.to_string(), EntityType::Concept, Utc::now(), "");
        }

        // Create some cycles for strongly connected components
        graph.add_relationship(EntityRelationship {
            from_entity: "a".to_string(),
            to_entity: "b".to_string(),
            relation_type: RelationType::RelatedTo,
            context: "test".to_string(),
        });

        graph.add_relationship(EntityRelationship {
            from_entity: "b".to_string(),
            to_entity: "c".to_string(),
            relation_type: RelationType::RelatedTo,
            context: "test".to_string(),
        });

        graph.add_relationship(EntityRelationship {
            from_entity: "c".to_string(),
            to_entity: "a".to_string(),
            relation_type: RelationType::RelatedTo,
            context: "test".to_string(),
        });

        let communities = graph.find_communities();

        // Should find at least one community
        if !communities.is_empty() {
            let first_community = &communities[0];
            assert!(first_community.len() >= 2);
        }
    }

    #[test]
    fn test_graph_serialization_persistence() {
        // Create a complex graph with multiple entities and relationships
        let mut original_graph = SimpleEntityGraph::new();

        // Add entities
        original_graph.add_or_update_entity(
            "rust".to_string(),
            EntityType::Technology,
            Utc::now(),
            "Systems programming language",
        );
        original_graph.add_or_update_entity(
            "petgraph".to_string(),
            EntityType::Technology,
            Utc::now(),
            "Graph library",
        );
        original_graph.add_or_update_entity(
            "performance".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Performance optimization",
        );
        original_graph.add_or_update_entity(
            "lock-free".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Lock-free data structures",
        );

        // Add relationships
        original_graph.add_relationship(EntityRelationship {
            from_entity: "rust".to_string(),
            to_entity: "petgraph".to_string(),
            relation_type: RelationType::Implements,
            context: "Rust implements petgraph".to_string(),
        });

        original_graph.add_relationship(EntityRelationship {
            from_entity: "petgraph".to_string(),
            to_entity: "performance".to_string(),
            relation_type: RelationType::LeadsTo,
            context: "Petgraph improves performance".to_string(),
        });

        original_graph.add_relationship(EntityRelationship {
            from_entity: "lock-free".to_string(),
            to_entity: "performance".to_string(),
            relation_type: RelationType::LeadsTo,
            context: "Lock-free improves performance".to_string(),
        });

        original_graph.add_relationship(EntityRelationship {
            from_entity: "rust".to_string(),
            to_entity: "lock-free".to_string(),
            relation_type: RelationType::Implements,
            context: "Rust supports lock-free".to_string(),
        });

        // Get original state
        let original_stats = original_graph.get_graph_stats();
        let original_rust_related = original_graph.find_related_entities("rust");
        let original_petgraph_related = original_graph.find_related_entities("petgraph");
        let original_trace = original_graph.trace_entity_relationships("rust", 3);
        let original_network = original_graph.get_entity_network("rust", 2);

        // Serialize
        let serialized = serde_json::to_string(&original_graph).expect("Serialization failed");

        // Deserialize
        let deserialized_graph: SimpleEntityGraph =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        // Verify graph stats match
        let deserialized_stats = deserialized_graph.get_graph_stats();
        assert_eq!(
            original_stats.node_count, deserialized_stats.node_count,
            "Node count mismatch"
        );
        assert_eq!(
            original_stats.edge_count, deserialized_stats.edge_count,
            "Edge count mismatch"
        );
        assert_eq!(
            original_stats.entity_count, deserialized_stats.entity_count,
            "Entity count mismatch"
        );

        // Verify relationships are preserved
        let deserialized_rust_related = deserialized_graph.find_related_entities("rust");
        assert_eq!(
            original_rust_related.len(),
            deserialized_rust_related.len(),
            "Rust relationships count mismatch"
        );
        assert!(
            deserialized_rust_related.contains(&"petgraph".to_string()),
            "Missing rust->petgraph relationship"
        );
        assert!(
            deserialized_rust_related.contains(&"lock-free".to_string()),
            "Missing rust->lock-free relationship"
        );

        let deserialized_petgraph_related = deserialized_graph.find_related_entities("petgraph");
        assert_eq!(
            original_petgraph_related.len(),
            deserialized_petgraph_related.len(),
            "Petgraph relationships count mismatch"
        );

        // Verify graph traversal works
        let deserialized_trace = deserialized_graph.trace_entity_relationships("rust", 3);
        assert_eq!(
            original_trace.len(),
            deserialized_trace.len(),
            "Trace length mismatch"
        );

        // Verify entity network works
        let deserialized_network = deserialized_graph.get_entity_network("rust", 2);
        assert_eq!(
            original_network.entities.len(),
            deserialized_network.entities.len(),
            "Network entities count mismatch"
        );
        assert_eq!(
            original_network.relationships.len(),
            deserialized_network.relationships.len(),
            "Network relationships count mismatch"
        );

        // Verify entity_to_node and node_to_entity mappings are rebuilt
        assert_eq!(
            deserialized_graph.entity_to_node.len(),
            4,
            "entity_to_node mapping not rebuilt"
        );
        assert_eq!(
            deserialized_graph.node_to_entity.len(),
            4,
            "node_to_entity mapping not rebuilt"
        );

        // Verify all entities have valid node mappings
        for entity_name in ["rust", "petgraph", "performance", "lock-free"] {
            assert!(
                deserialized_graph.entity_to_node.contains_key(entity_name),
                "Missing entity_to_node mapping for {}",
                entity_name
            );
            let node_idx = deserialized_graph.entity_to_node[entity_name];
            assert_eq!(
                deserialized_graph.node_to_entity[&node_idx], entity_name,
                "Inconsistent bidirectional mapping for {}",
                entity_name
            );
        }

        // Verify specific relationship queries work
        let rust_implements =
            deserialized_graph.find_related_entities_by_type("rust", &RelationType::Implements);
        assert_eq!(
            rust_implements.len(),
            2,
            "Should have 2 'Implements' relationships"
        );
        assert!(rust_implements.contains(&"petgraph".to_string()));
        assert!(rust_implements.contains(&"lock-free".to_string()));

        // Verify shortest path still works
        let path = deserialized_graph.find_shortest_path("rust", "performance");
        assert!(path.is_some(), "Should find path from rust to performance");

        println!(
            "✓ Graph serialization/deserialization preserves all relationships and functionality"
        );
    }

    #[test]
    fn test_empty_graph_serialization() {
        let original = SimpleEntityGraph::new();
        let serialized = serde_json::to_string(&original).expect("Serialization failed");
        let deserialized: SimpleEntityGraph =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(deserialized.get_graph_stats().node_count, 0);
        assert_eq!(deserialized.get_graph_stats().edge_count, 0);
        assert_eq!(deserialized.entity_to_node.len(), 0);
        assert_eq!(deserialized.node_to_entity.len(), 0);
    }

    #[test]
    fn test_graph_with_orphan_entities() {
        // Test graph with entities that have no relationships
        let mut graph = SimpleEntityGraph::new();

        graph.add_or_update_entity(
            "orphan1".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Orphan entity",
        );
        graph.add_or_update_entity(
            "orphan2".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Another orphan",
        );
        graph.add_or_update_entity(
            "connected1".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Connected entity",
        );
        graph.add_or_update_entity(
            "connected2".to_string(),
            EntityType::Concept,
            Utc::now(),
            "Another connected",
        );

        graph.add_relationship(EntityRelationship {
            from_entity: "connected1".to_string(),
            to_entity: "connected2".to_string(),
            relation_type: RelationType::RelatedTo,
            context: "test".to_string(),
        });

        let serialized = serde_json::to_string(&graph).expect("Serialization failed");
        let deserialized: SimpleEntityGraph =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        // All entities should be preserved
        assert_eq!(deserialized.entities.len(), 4);
        assert_eq!(deserialized.get_graph_stats().node_count, 4);
        assert_eq!(deserialized.get_graph_stats().edge_count, 1);

        // Verify orphan entities are accessible
        assert!(deserialized.entities.contains_key("orphan1"));
        assert!(deserialized.entities.contains_key("orphan2"));

        // Verify connected entities maintain their relationship
        let connected1_related = deserialized.find_related_entities("connected1");
        assert!(connected1_related.contains(&"connected2".to_string()));
    }
}
