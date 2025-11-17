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
use crate::core::context_update::{ContextUpdate, EntityType, RelationType, UpdateType};
use crate::core::structured_context::StructuredContext;
use crate::graph::entity_graph::SimpleEntityGraph;

use chrono::DateTime;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveSession {
    pub id: Uuid,
    pub name: Option<String>,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,

    // Tiered context storage
    pub hot_context: VecDeque<ContextUpdate>, // Last 20-50 updates (memory)
    pub warm_context: Vec<CompressedUpdate>,  // Compressed updates (storage)
    pub cold_context: Vec<StructuredSummary>, // Periodic summaries (storage)

    // Structured context
    pub current_state: StructuredContext, // Current queryable state
    pub incremental_updates: Vec<ContextUpdate>, // All incremental updates

    // Code integration
    pub code_references: HashMap<String, Vec<CodeReference>>, // By file path
    pub change_history: Vec<ChangeRecord>,                    // Change history

    pub user_preferences: UserPreferences,

    // Simple graph for entity relationships
    pub entity_graph: SimpleEntityGraph,

    // Entity extraction configuration
    #[serde(default = "default_max_entities")]
    pub max_extracted_entities: usize,
    #[serde(default = "default_max_entities")]
    pub max_referenced_entities: usize,
    #[serde(default = "default_true")]
    pub enable_smart_entity_ranking: bool,

    // Entity extraction metrics
    #[serde(default)]
    pub total_entity_truncations: usize,
    #[serde(default)]
    pub total_entities_truncated: usize,
}

fn default_max_entities() -> usize {
    15
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompressedUpdate {
    pub update: ContextUpdate,
    pub compression_ratio: f32,
    pub compressed_at: DateTime<Utc>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StructuredSummary {
    pub summary_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub context_snapshot: StructuredContext,
    pub referenced_updates: Vec<Uuid>,
    pub summary_quality: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CodeReference {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub code_snippet: String,
    pub commit_hash: Option<String>,
    pub branch: Option<String>,
    pub change_description: String,
}

// Remove duplicate CodeReference definition since we're using the one from core
// Removed duplicate field declarations - using CodeReference from core::context_update
// No extra closing brace needed here

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChangeRecord {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub change_type: String,
    pub description: String,
    pub related_update_id: Option<Uuid>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UserPreferences {
    pub auto_save_enabled: bool,
    pub context_retention_days: u32,
    pub max_hot_context_size: usize,
    pub auto_summary_threshold: usize,
    pub important_keywords: Vec<String>,
}

impl Default for ActiveSession {
    fn default() -> Self {
        Self::new(Uuid::new_v4(), None, None)
    }
}

impl ActiveSession {
    pub fn new(id: Uuid, name: Option<String>, description: Option<String>) -> Self {
        Self {
            id,
            name,
            description,
            created_at: Utc::now(),
            last_updated: Utc::now(),
            hot_context: VecDeque::new(),
            warm_context: Vec::new(),
            cold_context: Vec::new(),
            current_state: StructuredContext::new(),
            incremental_updates: Vec::new(),
            code_references: HashMap::new(),
            change_history: Vec::new(),
            user_preferences: UserPreferences {
                auto_save_enabled: true,
                context_retention_days: 30,
                max_hot_context_size: 50,
                auto_summary_threshold: 100,
                important_keywords: vec![],
            },
            entity_graph: SimpleEntityGraph::new(),
            max_extracted_entities: 15,
            max_referenced_entities: 15,
            enable_smart_entity_ranking: true,
            total_entity_truncations: 0,
            total_entities_truncated: 0,
        }
    }

    #[instrument(skip(self, update), fields(session_id = %self.id))]
    pub async fn add_incremental_update(&mut self, update: ContextUpdate) -> anyhow::Result<()> {
        info!(
            "ActiveSession: Starting add_incremental_update for update ID: {}",
            update.id
        );
        info!("Update type: {:?}", update.update_type);
        info!("Content title: '{}'", update.content.title);
        info!("Content description: '{}'", update.content.description);

        // Limit content size to prevent processing issues
        let mut limited_update = update.clone();
        if limited_update.content.description.len() > 2000 {
            limited_update.content.description.truncate(1800);
            limited_update
                .content
                .description
                .push_str("... (truncated)");
            warn!("ActiveSession: Content description truncated to prevent timeout");
        }
        if limited_update.content.title.len() > 200 {
            limited_update.content.title.truncate(190);
            limited_update.content.title.push_str("...");
        }

        // Add to hot context
        debug!("ActiveSession: Adding to hot context");
        self.hot_context.push_back(limited_update.clone());
        debug!(
            "ActiveSession: Hot context updated, size: {}",
            self.hot_context.len()
        );

        // Update structured state with timeout
        debug!("ActiveSession: Calling update_current_state");
        match timeout(
            Duration::from_secs(3),
            self.update_current_state(&limited_update),
        )
        .await
        {
            Ok(result) => result?,
            Err(_) => {
                warn!("ActiveSession: update_current_state timed out");
                return Err(anyhow::anyhow!("Current state update timeout"));
            }
        }
        debug!("ActiveSession: update_current_state completed");

        // Update entity graph with timeout
        debug!("ActiveSession: Calling update_entity_graph");
        match timeout(
            Duration::from_secs(5),
            self.update_entity_graph(&limited_update),
        )
        .await
        {
            Ok(result) => result?,
            Err(_) => {
                warn!("ActiveSession: update_entity_graph timed out");
                return Err(anyhow::anyhow!("Entity graph update timeout"));
            }
        }
        debug!("ActiveSession: update_entity_graph completed");

        // Add code reference if present with timeout
        debug!("DEBUG: About to check limited_update.related_code");
        if let Some(code_ref) = &limited_update.related_code {
            debug!("ActiveSession: Code reference found, processing...");
            let code_ref_clone = CodeReference {
                file_path: code_ref.file_path.clone(),
                start_line: code_ref.start_line,
                end_line: code_ref.end_line,
                code_snippet: code_ref.code_snippet.clone(),
                commit_hash: code_ref.commit_hash.clone(),
                branch: code_ref.branch.clone(),
                change_description: code_ref.change_description.clone(),
            };
            debug!("ActiveSession: Calling add_code_reference");
            match timeout(
                Duration::from_secs(2),
                self.add_code_reference(&code_ref_clone),
            )
            .await
            {
                Ok(result) => result?,
                Err(_) => {
                    warn!("ActiveSession: add_code_reference timed out");
                    // Continue without failing the entire operation
                }
            }
            debug!("ActiveSession: add_code_reference completed");
        } else {
            debug!("ActiveSession: No code reference in update");
        }

        // Record change (sync now)
        debug!("ActiveSession: Calling record_change");
        self.record_change(&limited_update)?;
        debug!("ActiveSession: record_change completed");

        // Maintain context size (sync now)
        debug!("ActiveSession: Calling maintain_context");
        self.maintain_context()?;
        debug!("ActiveSession: maintain_context completed");

        // Update last updated timestamp
        debug!("ActiveSession: Updating timestamp");
        self.last_updated = Utc::now();

        // Add to incremental updates (use original update for storage)
        self.incremental_updates.push(limited_update.clone());

        info!("ActiveSession: add_incremental_update completed successfully");

        Ok(())
    }

    async fn update_current_state(&mut self, update: &ContextUpdate) -> anyhow::Result<()> {
        // Update structured context based on update type
        match &update.update_type {
            UpdateType::QuestionAnswered => {
                // Add to conversation flow
                self.current_state.conversation_flow.push(
                    crate::core::structured_context::FlowItem {
                        step_description: format!("Q&A: {}", update.content.title),
                        timestamp: update.timestamp,
                        related_updates: vec![update.id],
                        outcome: Some(update.content.description.clone()),
                    },
                );
            }
            UpdateType::ProblemSolved => {
                // Add to conversation flow
                self.current_state.conversation_flow.push(
                    crate::core::structured_context::FlowItem {
                        step_description: format!("Problem Solved: {}", update.content.title),
                        timestamp: update.timestamp,
                        related_updates: vec![update.id],
                        outcome: Some(update.content.description.clone()),
                    },
                );
            }
            UpdateType::CodeChanged => {
                // Add to conversation flow
                self.current_state.conversation_flow.push(
                    crate::core::structured_context::FlowItem {
                        step_description: format!("Code Change: {}", update.content.title),
                        timestamp: update.timestamp,
                        related_updates: vec![update.id],
                        outcome: Some(update.content.description.clone()),
                    },
                );
            }
            UpdateType::DecisionMade => {
                // Add to key decisions
                self.current_state.key_decisions.push(
                    crate::core::structured_context::DecisionItem {
                        description: update.content.title.clone(),
                        context: update.content.description.clone(),
                        alternatives: update.content.details.clone(),
                        confidence: 1.0,
                        timestamp: update.timestamp,
                    },
                );

                // Add to conversation flow
                self.current_state.conversation_flow.push(
                    crate::core::structured_context::FlowItem {
                        step_description: format!("Decision Made: {}", update.content.title),
                        timestamp: update.timestamp,
                        related_updates: vec![update.id],
                        outcome: Some(update.content.description.clone()),
                    },
                );
            }
            UpdateType::ConceptDefined => {
                // Add to key concepts
                self.current_state.key_concepts.push(
                    crate::core::structured_context::ConceptItem {
                        name: update.content.title.clone(),
                        definition: update.content.description.clone(),
                        examples: update.content.examples.clone(),
                        related_concepts: update.content.details.clone(),
                        timestamp: update.timestamp,
                    },
                );

                // Add to conversation flow
                self.current_state.conversation_flow.push(
                    crate::core::structured_context::FlowItem {
                        step_description: format!("Concept Defined: {}", update.content.title),
                        timestamp: update.timestamp,
                        related_updates: vec![update.id],
                        outcome: Some(update.content.description.clone()),
                    },
                );
            }
            UpdateType::RequirementAdded => {
                // Add to technical specifications
                self.current_state.technical_specifications.push(
                    crate::core::structured_context::SpecItem {
                        title: update.content.title.clone(),
                        description: update.content.description.clone(),
                        requirements: update.content.details.clone(),
                        constraints: update.content.implications.clone(),
                        timestamp: update.timestamp,
                    },
                );

                // Add to conversation flow
                self.current_state.conversation_flow.push(
                    crate::core::structured_context::FlowItem {
                        step_description: format!("Requirement Added: {}", update.content.title),
                        timestamp: update.timestamp,
                        related_updates: vec![update.id],
                        outcome: Some(update.content.description.clone()),
                    },
                );
            }
        }

        Ok(())
    }

    async fn update_entity_graph(&mut self, update: &ContextUpdate) -> anyhow::Result<()> {
        info!(
            "update_entity_graph: Starting entity graph update for update {}",
            update.id
        );

        // Extract entities from update content if not explicitly provided
        info!("DEBUG: About to clone creates_entities");
        let mut extracted_entities = update.creates_entities.clone();
        info!(
            "DEBUG: Cloned creates_entities, length={}",
            extracted_entities.len()
        );

        info!("DEBUG: About to clone references_entities");
        let mut referenced_entities = update.references_entities.clone();
        info!(
            "DEBUG: Cloned references_entities, length={}",
            referenced_entities.len()
        );

        eprintln!("EPRINTLN TEST: About to log Explicit entities");
        info!(
            "Explicit entities: creates={}, references={}",
            extracted_entities.len(),
            referenced_entities.len()
        );
        eprintln!("EPRINTLN TEST: After Explicit entities log");

        // If no entities were explicitly provided, extract from content
        if extracted_entities.is_empty() && referenced_entities.is_empty() {
            let content_text = format!(
                "{} {} {}",
                update.content.title,
                update.content.description,
                update.content.details.join(" ")
            );
            info!("Extracting entities from text: '{}'", content_text);

            let auto_extracted = self.extract_entities_from_text(&content_text);
            info!(
                "Auto-extracted {} entities: {:?}",
                auto_extracted.len(),
                auto_extracted
            );

            extracted_entities.extend(auto_extracted.clone());
            referenced_entities.extend(auto_extracted);
        }

        // Prepare content text for entity scoring
        let content_text = format!(
            "{} {} {}",
            update.content.title,
            update.content.description,
            update.content.details.join(" ")
        );

        // Smart ranking and truncation of entities
        let (ranked_extracted, extracted_truncated) = self.rank_and_truncate_entities(
            extracted_entities,
            &content_text,
            self.max_extracted_entities,
            self.enable_smart_entity_ranking,
        );
        let (ranked_referenced, referenced_truncated) = self.rank_and_truncate_entities(
            referenced_entities,
            &content_text,
            self.max_referenced_entities,
            self.enable_smart_entity_ranking,
        );

        extracted_entities = ranked_extracted;
        referenced_entities = ranked_referenced;

        // Update truncation metrics
        let total_truncated = extracted_truncated + referenced_truncated;
        if total_truncated > 0 {
            self.total_entity_truncations += 1;
            self.total_entities_truncated += total_truncated;
        }

        info!(
            "Final entity counts - extracted: {} (truncated: {}), referenced: {} (truncated: {}). Total session truncations: {}, total entities lost: {}",
            extracted_entities.len(),
            extracted_truncated,
            referenced_entities.len(),
            referenced_truncated,
            self.total_entity_truncations,
            self.total_entities_truncated
        );

        // Create new entities (batch operation)
        info!("DEBUG: About to start entity loop");
        for entity_name in &extracted_entities {
            info!("Adding entity: '{}'", entity_name);
            self.entity_graph.add_or_update_entity(
                entity_name.clone(),
                self.infer_entity_type(&update.update_type, entity_name),
                update.timestamp,
                &update.content.description,
            );
        }

        info!(
            "Entity graph now has {} total entities",
            self.entity_graph.entities.len()
        );

        // Add entity references (batch operation)
        info!(
            "DEBUG: Adding entity references for {} entities",
            referenced_entities.len()
        );
        for entity_name in &referenced_entities {
            self.entity_graph
                .mention_entity(entity_name, update.id, update.timestamp);
        }
        info!("DEBUG: Entity references added successfully");

        // Create relationships (always process explicit relationships)
        info!(
            "DEBUG: Creating {} explicit relationships",
            update.creates_relationships.len()
        );
        for relationship in &update.creates_relationships {
            self.entity_graph.add_relationship(relationship.clone());
        }
        info!("DEBUG: Explicit relationships created successfully");

        // Auto-generate relationships between entities in the same update
        // Always generate relationships - internal limits (max 10 entities, max 15 relationships) protect performance
        let total_entities = extracted_entities.len() + referenced_entities.len();
        info!(
            "DEBUG: Auto-generating relationships for {} total entities",
            total_entities
        );
        self.auto_generate_relationships(&extracted_entities, &referenced_entities, update)?;
        info!("DEBUG: Auto-generate relationships completed");

        info!("DEBUG: add_incremental_update completed successfully, returning Ok");
        Ok(())
    }

    fn infer_entity_type(&self, update_type: &UpdateType, entity_name: &str) -> EntityType {
        match update_type {
            UpdateType::CodeChanged => EntityType::CodeComponent,
            UpdateType::ProblemSolved => EntityType::Solution,
            UpdateType::DecisionMade => EntityType::Decision,
            UpdateType::ConceptDefined => EntityType::Concept,
            _ => {
                // Enhanced heuristics based on entity name
                let name_lower = entity_name.to_lowercase();

                // Technologies
                if name_lower.contains("rust")
                    || name_lower.contains("cargo")
                    || name_lower.contains("postgresql")
                    || name_lower.contains("postgres")
                    || name_lower.contains("jwt")
                    || name_lower.contains("json")
                    || name_lower.contains("api")
                    || name_lower.contains("http")
                    || name_lower.contains("sql")
                    || name_lower.contains("database")
                    || name_lower.contains("redis")
                    || name_lower.contains("docker")
                    || name_lower.contains("kubernetes")
                    || name_lower.contains("git")
                    || name_lower.contains("server")
                    || name_lower.contains("client")
                    || name_lower.ends_with(".rs")
                    || name_lower.ends_with(".sql")
                    || name_lower.ends_with(".toml")
                    || name_lower.ends_with(".json")
                {
                    EntityType::Technology
                }
                // Problems
                else if name_lower.contains("bug")
                    || name_lower.contains("issue")
                    || name_lower.contains("problem")
                    || name_lower.contains("error")
                    || name_lower.contains("fail")
                    || name_lower.contains("crash")
                    || name_lower.contains("exception")
                    || name_lower.contains("panic")
                {
                    EntityType::Problem
                }
                // Code components
                else if name_lower.contains("function")
                    || name_lower.contains("method")
                    || name_lower.contains("struct")
                    || name_lower.contains("enum")
                    || name_lower.contains("trait")
                    || name_lower.contains("impl")
                    || name_lower.contains("module")
                    || name_lower.contains("lib")
                    || name_lower.contains("crate")
                    || name_lower.contains("package")
                    || name_lower.contains("/")
                    || name_lower.contains("::")
                {
                    EntityType::CodeComponent
                }
                // Solutions
                else if name_lower.contains("fix")
                    || name_lower.contains("solution")
                    || name_lower.contains("resolve")
                    || name_lower.contains("implement")
                    || name_lower.contains("patch")
                    || name_lower.contains("update")
                {
                    EntityType::Solution
                }
                // Default to concept
                else {
                    EntityType::Concept
                }
            }
        }
    }

    // ========== Entity Intelligence Helpers ==========

    /// Clean entity name by removing punctuation (safe, defensive)
    fn clean_entity_name(&self, name: &str) -> Option<String> {
        if name.is_empty() {
            return None;
        }

        let cleaned = name
            .trim()
            .trim_end_matches(|c: char| c.is_ascii_punctuation() && c != '(' && c != ')')
            .trim_start_matches(|c: char| c.is_ascii_punctuation() && c != '(' && c != ')')
            .trim();

        if cleaned.is_empty() {
            return None;
        }

        Some(cleaned.to_string())
    }

    /// Check if entity name is a stop word
    fn is_stop_word(&self, name: &str) -> bool {
        if name.is_empty() {
            return true;
        }

        const STOP_WORDS: &[&str] = &[
            "a", "an", "the", "this", "that", "these", "those", "in", "on", "at", "by", "for",
            "with", "from", "to", "of", "and", "or", "but", "nor", "so", "yet", "all", "some",
            "any", "each", "every", "both", "few", "many", "total", "using", "used", "uses", "use",
            "made", "make", "now", "then", "when", "where", "how", "what", "why", "one", "two",
            "three", "first", "second", "last",
        ];

        let normalized = name.to_lowercase();
        STOP_WORDS.contains(&normalized.as_str()) || normalized.parse::<f64>().is_ok()
    }

    /// Normalize entity name (safe, returns None on error)
    fn normalize_entity(&self, name: &str) -> Option<String> {
        let cleaned = self.clean_entity_name(name)?;

        // Remove function call parentheses
        let mut normalized = if cleaned.ends_with("()") {
            cleaned.trim_end_matches("()").to_string()
        } else {
            cleaned
        };

        // Safety check after removing ()
        if normalized.is_empty() {
            return None;
        }

        // Normalize common technical terms
        let lower = normalized.to_lowercase();
        normalized = match lower.as_str() {
            "rwlock" | "rw_lock" => "RwLock".to_string(),
            "mutex" => "Mutex".to_string(),
            "arc" => "Arc".to_string(),
            "hashmap" | "hash_map" => "HashMap".to_string(),
            "vec" => "Vec".to_string(),
            "string" => "String".to_string(),
            "option" => "Option".to_string(),
            "result" => "Result".to_string(),
            _ => normalized,
        };

        Some(normalized)
    }

    /// Validate entity (safe, defensive checks)
    fn is_valid_entity(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }

        let cleaned = match self.clean_entity_name(name) {
            Some(c) => c,
            None => return false,
        };

        if cleaned.len() < 2 || cleaned.len() > 30 {
            return false;
        }

        !self.is_stop_word(&cleaned)
    }

    // ========== End Entity Intelligence Helpers ==========

    fn extract_entities_from_text(&self, text: &str) -> Vec<String> {
        use std::collections::HashSet;
        let mut entities = HashSet::new();
        let text_lower = text.to_lowercase();

        info!("extract_entities_from_text: Processing text: '{}'", text);
        info!("Text lowercase: '{}'", text_lower);

        // Check for non-ASCII text (e.g., Cyrillic, Chinese, etc.)
        let is_ascii = text.is_ascii();
        if !is_ascii {
            info!("Non-ASCII text detected, using intelligent multilingual entity extraction");

            // Extract all words with frequency counting
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut term_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

            for word in &words {
                // Clean punctuation from word boundaries
                let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
                if !cleaned.is_empty() {
                    *term_freq.entry(cleaned.to_lowercase()).or_insert(0) += 1;
                }
            }

            // Score and filter terms
            for (term, count) in term_freq {
                let mut score = 0.0;

                // Filter stop words first (Bulgarian and English)
                if self.is_bulgarian_stop_word(&term) || self.is_common_word(&term) {
                    continue;
                }

                // Length scoring (prefer longer, more specific terms)
                if term.len() >= 8 {
                    score += 3.5; // Very long technical terms (ContentVectorizer, SemanticQueryEngine)
                } else if term.len() >= 5 {
                    score += 2.5; // Long technical terms (dashmap, atomic)
                } else if term.len() >= 3 {
                    score += 1.0; // Short but potentially meaningful (cpu, api)
                } else {
                    continue; // Skip very short terms
                }

                // Frequency bonus (repeated terms are important)
                score += (count as f64 * 0.8).min(2.5);

                // Technical indicators bonus (STRONG signals)
                if term.contains('_') || term.contains('-') {
                    score += 3.0; // snake_case or hyphenated terms → definitely technical
                }

                // Compound technical patterns (angle brackets, generics)
                if term.contains('<') || term.contains('>') {
                    score += 3.5; // RwLock<HashMap>, Vec<String>
                }

                // CamelCase detection (works for any script)
                let has_mixed_case = term.chars().any(|c| c.is_uppercase())
                    && term.chars().any(|c| c.is_lowercase());
                if has_mixed_case {
                    score += 2.5; // CamelCase → likely technical
                }

                // Numbers in term (version numbers, technical IDs)
                if term.chars().any(|c| c.is_numeric()) {
                    score += 2.0; // AtomicU64, http2
                }

                // All uppercase (acronyms, constants)
                if term.len() >= 2 && term.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()) {
                    score += 3.0; // HNSW, CPU, API
                }

                // Known technical suffixes (language-agnostic)
                if term.ends_with("engine") || term.ends_with("config") || term.ends_with("manager")
                    || term.ends_with("handler") || term.ends_with("vectorizer") || term.ends_with("controller")
                {
                    score += 2.5;
                }

                // Accept if score >= 3.0 (balanced threshold for quality)
                if score >= 3.0 && term.len() <= 35 {
                    entities.insert(term);
                }
            }

            // Also extract quoted terms and code patterns (language-agnostic)
            for word in &words {
                let word_str = *word;

                // Backticks, quotes (code/technical terms)
                if (word_str.starts_with('`') && word_str.ends_with('`'))
                    || (word_str.starts_with('"') && word_str.ends_with('"'))
                    || (word_str.starts_with('\'') && word_str.ends_with('\''))
                {
                    let term = word_str
                        .trim_matches(|c: char| c == '`' || c == '"' || c == '\'')
                        .to_string();
                    if term.len() >= 3 && term.len() <= 25 {
                        entities.insert(term);
                    }
                }

                // File extensions
                if word_str.contains('.')
                    && (word_str.ends_with(".rs")
                        || word_str.ends_with(".sql")
                        || word_str.ends_with(".json")
                        || word_str.ends_with(".toml")
                        || word_str.ends_with(".md"))
                {
                    entities.insert(word_str.to_string());
                }
            }

            let final_result = entities.into_iter().take(20).collect::<Vec<_>>();
            info!("Intelligent multilingual extracted entities: {:?}", final_result);
            return final_result;
        }

        // Extract entities using multiple intelligent patterns (only for ASCII text)
        self.extract_proper_nouns(text, &mut entities);
        self.extract_technical_terms(&text_lower, &mut entities);
        self.extract_quoted_terms(text, &mut entities);
        self.extract_compound_terms(text, &mut entities);
        self.extract_domain_specific_terms(&text_lower, &mut entities);

        info!("Found {} entities from pattern matching", entities.len());

        // Extract file paths (simple pattern)
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if word.contains('.')
                && (word.ends_with(".rs")
                    || word.ends_with(".sql")
                    || word.ends_with(".json")
                    || word.ends_with(".toml"))
                && word.len() > 5
            {
                entities.insert(word.to_string());
            }
        }

        // Clean, normalize, and filter entities (safe with Option handling)
        let mut cleaned_entities = HashSet::new();
        for entity in entities.iter() {
            // Validate first
            if !self.is_valid_entity(entity) {
                continue;
            }

            // Normalize (returns None on error)
            if let Some(normalized) = self.normalize_entity(entity) {
                // Re-validate after normalization
                if self.is_valid_entity(&normalized) && !normalized.is_empty() {
                    cleaned_entities.insert(normalized);
                }
            }
        }

        info!(
            "Entities after cleaning: {} -> {} (filtered {} invalid)",
            entities.len(),
            cleaned_entities.len(),
            entities.len().saturating_sub(cleaned_entities.len())
        );

        // Score and filter entities by relevance
        let scored_entities = self.score_entities(&cleaned_entities, text);
        let final_result = scored_entities.into_iter().take(20).collect::<Vec<_>>();
        info!("Final extracted entities: {:?}", final_result);
        final_result
    }

    /// Extract proper nouns and capitalized terms
    fn extract_proper_nouns(&self, text: &str, entities: &mut std::collections::HashSet<String>) {
        // Use expect() for compile-time constant regex - this should never fail
        // If it does, it indicates a programming error, not a runtime condition
        let capitalized_regex = regex::Regex::new(r"\b[A-Z][a-zA-Z-]{2,}\b")
            .expect("Built-in regex pattern should always compile");

        let common_words = [
            "The", "This", "That", "These", "Those", "When", "Where", "What", "Why", "How", "Who",
            "Which", "Will", "Would", "Could", "Should", "Must", "Can", "May", "Might", "And",
            "But", "Or", "Not", "So", "Yet", "For", "Nor", "Because", "Although", "Since", "While",
            "Until", "Unless", "Before", "After", "During", "Through",
        ];

        for cap in capitalized_regex.find_iter(text) {
            let original = cap.as_str();
            let term = original.to_lowercase();

            if term.len() >= 3 && term.len() <= 20 && !common_words.contains(&original) {
                entities.insert(term);
            }
        }
    }

    /// Extract technical and domain-specific terms using patterns
    fn extract_technical_terms(
        &self,
        text_lower: &str,
        entities: &mut std::collections::HashSet<String>,
    ) {
        // Skip regex patterns for non-ASCII text to prevent infinite loops
        if !text_lower.is_ascii() {
            return;
        }

        // Programming language patterns
        let prog_patterns = [
            r"\b\w+script\b",
            r"\b\w+lang\b",
            r"\b\w+\+\+\b",
            r"\b\w*sql\b",
            r"\b\w*db\b",
            r"\b\w*api\b",
            r"\b\w*json\b",
            r"\b\w*xml\b",
        ];

        // Extract lowercase technical terms (3-15 chars) with frequency-based scoring
        // This is language-agnostic and works for any domain
        let lowercase_term_regex = regex::Regex::new(r"\b[a-z][a-z0-9_]{2,14}\b")
            .expect("Built-in regex pattern should always compile");

        // Step 1: Collect all terms with frequency count
        let mut term_freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for cap in lowercase_term_regex.find_iter(text_lower) {
            let term = cap.as_str().to_string();
            *term_freq.entry(term).or_insert(0) += 1;
        }

        // Step 2: Score each term based on multiple factors
        for (term, count) in term_freq {
            let mut score = 0.0;

            // Length bonus (longer terms are more likely technical)
            if term.len() >= 5 {
                score += 2.0; // petgraph, dashmap, tokio
            } else {
                score += 0.5; // arc, vec, api (short but potentially important)
            }

            // Frequency bonus (repeated terms are important)
            // Cap at 2.0 to prevent common words from dominating
            score += (count as f64 * 0.5).min(2.0);

            // Pattern bonuses (technical indicators)
            if term.contains('_') {
                score += 1.5; // snake_case → definitely technical
            }
            if term.chars().any(|c| c.is_numeric()) {
                score += 1.5; // http2, version3 → technical
            }

            // Accept if score >= 2.0 (threshold for technical relevance)
            // Stop words filter will remove common words even if they score high
            if score >= 2.0 {
                entities.insert(term);
            }
        }

        // Process patterns
        let process_patterns = [
            r"\b\w*-free\b",
            r"\b\w*safe\b",
            r"\b\w*async\b",
            r"\b\w*sync\b",
            r"\b\w*lock\b",
            r"\b\w*thread\b",
            r"\b\w*process\b",
            r"\b\w*cache\b",
        ];

        // Architecture patterns
        let arch_patterns = [
            r"\bmicro\w*\b",
            r"\bmulti-\w+\b",
            r"\bdistributed-\w+\b",
            r"\bcloud-\w+\b",
            r"\bserver\w*\b",
            r"\bclient\w*\b",
            r"\bprotocol\w*\b",
        ];

        let all_patterns = [
            prog_patterns.as_slice(),
            process_patterns.as_slice(),
            arch_patterns.as_slice(),
        ]
        .concat();

        // Limit processing to prevent timeout
        const MAX_PATTERNS: usize = 10;

        for (processed_patterns, pattern) in all_patterns.into_iter().enumerate() {
            if processed_patterns >= MAX_PATTERNS {
                break;
            }
            if let Ok(regex) = regex::Regex::new(pattern) {
                let mut matches = 0;
                for mat in regex.find_iter(text_lower) {
                    let term = mat.as_str();
                    if term.len() >= 3 && term.len() <= 20 {
                        entities.insert(term.to_string());
                        matches += 1;
                        if matches > 20 {
                            // Limit matches per pattern
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Extract terms in quotes, backticks, or code blocks
    fn extract_quoted_terms(&self, text: &str, entities: &mut std::collections::HashSet<String>) {
        let quoted_patterns = [
            r#"["']([^"']{2,20})["']"#, // Single/double quotes
            r"`([^`]{2,20})`",          // Backticks
            r"\{([^}]{2,15})\}",        // Curly braces
            r"\[([^\]]{2,15})\]",       // Square brackets
        ];

        for pattern in quoted_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for cap in regex.captures_iter(text) {
                    if let Some(quoted_term) = cap.get(1) {
                        let term = quoted_term.as_str().to_lowercase();
                        if term.len() >= 2
                            && term.len() <= 15
                            && !term.chars().all(|c| c.is_numeric())
                        {
                            entities.insert(term);
                        }
                    }
                }
            }
        }
    }

    /// Extract compound terms with hyphens, underscores, dots
    fn extract_compound_terms(&self, text: &str, entities: &mut std::collections::HashSet<String>) {
        let compound_patterns = [
            r"\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b",   // Hyphenated
            r"\b[a-zA-Z]+_[a-zA-Z]+(?:_[a-zA-Z]+)*\b",   // Underscore
            r"\b[a-zA-Z]+\.[a-zA-Z]+(?:\.[a-zA-Z]+)*\b", // Dotted (like file.ext, domain.com)
            r"\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b",            // CamelCase
        ];

        for pattern in compound_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                for cap in regex.find_iter(text) {
                    let term = cap.as_str().to_lowercase();
                    if term.len() >= 4 && term.len() <= 25 {
                        entities.insert(term);
                    }
                }
            }
        }
    }

    /// Extract domain-specific terms based on context clues
    fn extract_domain_specific_terms(
        &self,
        text_lower: &str,
        entities: &mut std::collections::HashSet<String>,
    ) {
        // Look for terms near context indicators
        let context_indicators = [
            ("using", 2),
            ("with", 2),
            ("via", 2),
            ("through", 2),
            ("technology", 3),
            ("system", 3),
            ("framework", 3),
            ("library", 3),
            ("protocol", 3),
            ("method", 3),
            ("approach", 3),
            ("solution", 3),
            ("tool", 2),
            ("service", 3),
            ("platform", 3),
            ("engine", 3),
        ];

        let words: Vec<&str> = text_lower.split_whitespace().collect();

        for (indicator, range) in context_indicators {
            if let Some(pos) = words.iter().position(|&w| w == indicator) {
                // Look for entities around the indicator
                let start = pos.saturating_sub(range);
                let end = (pos + range + 1).min(words.len());

                for &word in &words[start..end] {
                    if word != indicator && word.len() >= 3 && word.len() <= 20 {
                        // Filter out common English words
                        if !self.is_common_word(word) {
                            entities.insert(word.to_string());
                        }
                    }
                }
            }
        }
    }

    /// Check if a word is a common English word that shouldn't be an entity
    fn is_common_word(&self, word: &str) -> bool {
        let common_words = [
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "who",
            "boy",
            "did",
            "man",
            "men",
            "put",
            "say",
            "she",
            "too",
            "use",
            "this",
            "that",
            "with",
            "have",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
            "come",
            "here",
            "just",
            "like",
            "long",
            "make",
            "many",
            "over",
            "such",
            "take",
            "than",
            "them",
            "well",
            "were",
            "will",
            "your",
            "about",
            "after",
            "again",
            "back",
            "could",
            "every",
            "first",
            "going",
            "house",
            "little",
            "might",
            "never",
            "only",
            "other",
            "right",
            "should",
            "through",
            "under",
            "where",
            "while",
            "would",
            "write",
            "years",
            "because",
            "before",
            "being",
            "between",
            "during",
            "through",
            "without",
            "within",
            "against",
            "across",
            "around",
            "behind",
            "beside",
            "beneath",
            "beyond",
            "inside",
            "outside",
            "toward",
            "towards",
            "underneath",
        ];

        common_words.contains(&word)
    }

    /// Check if a word is a common Bulgarian stop word
    fn is_bulgarian_stop_word(&self, word: &str) -> bool {
        let bulgarian_stop_words = [
            // Common Bulgarian stop words
            "и", "в", "на", "за", "с", "от", "до", "по", "при", "без",
            "е", "са", "си", "съм", "беше", "бяха", "ще", "би",
            "това", "тази", "тези", "този", "онзи", "която", "който", "които",
            "как", "какво", "къде", "кога", "защо", "кой",
            "не", "да", "ли", "че", "ако", "когато", "докато",
            "или", "но", "а", "още", "само", "вече", "тук", "там",
            "всички", "всяка", "всеки", "няколко", "много", "малко",
            "го", "му", "й", "ги", "им", "ме", "ти", "ви", "ни",
            "един", "една", "едно", "два", "две", "три",
            // Common verbs
            "прави", "прави", "работи", "работят", "има", "имат",
            "мога", "може", "могат", "трябва", "искам", "иска",
            "използва", "използваме", "използвам", "използват",
            "осигурява", "осигуряват", "позволява", "позволяват",
            "оркестрира", "оркестрират", "върна", "връща", "връщат",
            "търсения", "търсене", "достъп", "данни", "параметри",
            "сесии", "сесия", "система", "системи",
            // Common prepositions and conjunctions
            "през", "след", "преди", "около", "между", "над", "под",
            "към", "чрез", "според", "заради", "поради",
            // Common adjectives and descriptors
            "различни", "различен", "различна", "различно",
            "семантични", "семантичен", "семантична", "семантично", "семантичните",
            "приблизително", "приблизителен", "приблизителна",
            "по-добър", "по-добра", "по-добро", "по-добре",
            "висока", "висок", "високо", "високи",
            "вместо", "заради", "поради",
            "индексът", "индекса", "индекси",
            "използване", "използването",
            // Common nouns that are too generic
            "начин", "начина", "начини",
            "вид", "вида", "видове",
            "тип", "типа", "типове",
            "част", "частта", "части",
            "случай", "случая", "случаи",
        ];

        bulgarian_stop_words.contains(&word)
    }

    /// Score entities based on frequency, length, and context relevance
    fn score_entities(
        &self,
        entities: &std::collections::HashSet<String>,
        original_text: &str,
    ) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = entities
            .iter()
            .map(|entity| {
                let score = self.calculate_entity_score(entity, original_text);
                (entity.clone(), score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(entity, _)| entity).collect()
    }

    /// Calculate relevance score for an entity
    fn calculate_entity_score(&self, entity: &str, text: &str) -> f64 {
        let mut score = 0.0;

        // Base score from length (sweet spot is 4-12 characters)
        let length_score = match entity.len() {
            1..=2 => 0.1,
            3 => 0.3,
            4..=8 => 1.0,
            9..=12 => 0.8,
            13..=20 => 0.5,
            _ => 0.2,
        };
        score += length_score;

        // Frequency score (more mentions = more important) - case insensitive
        let freq_count = text.to_lowercase().matches(&entity.to_lowercase()).count() as f64;
        score += freq_count * 0.3;

        // Pattern bonuses
        if entity.contains('-') || entity.contains('_') {
            score += 0.4; // Compound terms often important
        }

        if entity.chars().any(|c| c.is_uppercase()) {
            score += 0.3; // Proper nouns often important
        }

        // Technical term indicators
        let tech_suffixes = ["api", "db", "sql", "json", "xml", "http", "tcp", "udp"];
        if tech_suffixes.iter().any(|&suffix| entity.ends_with(suffix)) {
            score += 0.5;
        }

        let tech_prefixes = ["micro", "multi", "auto", "async", "sync"];
        if tech_prefixes
            .iter()
            .any(|&prefix| entity.starts_with(prefix))
        {
            score += 0.4;
        }

        // Architecture/process terms
        let important_patterns = [
            "system",
            "service",
            "protocol",
            "framework",
            "library",
            "engine",
            "platform",
            "server",
            "client",
            "cache",
            "storage",
            "database",
            "memory",
            "thread",
            "process",
        ];
        if important_patterns
            .iter()
            .any(|&pattern| entity.contains(pattern))
        {
            score += 0.6;
        }

        // Penalize very common words that slipped through
        let somewhat_common = ["thing", "stuff", "something", "anything", "everything"];
        if somewhat_common.contains(&entity) {
            score *= 0.1;
        }

        score
    }

    /// Rank entities by importance and truncate to specified limit
    /// Uses frequency scoring to keep the most relevant entities
    fn rank_and_truncate_entities(
        &self,
        entities: Vec<String>,
        text: &str,
        max_count: usize,
        enable_ranking: bool,
    ) -> (Vec<String>, usize) {
        let original_count = entities.len();

        // If ranking is disabled or we're under the limit, just truncate
        if !enable_ranking || original_count <= max_count {
            let truncated = entities.into_iter().take(max_count).collect();
            return (truncated, 0);
        }

        // Score all entities
        let mut scored: Vec<(String, f64)> = entities
            .into_iter()
            .map(|entity| {
                let score = self.calculate_entity_score(&entity, text);
                (entity, score)
            })
            .collect();

        // Sort by score (highest first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        let ranked_entities: Vec<String> = scored
            .into_iter()
            .take(max_count)
            .map(|(entity, _score)| entity)
            .collect();

        let truncated_count = original_count.saturating_sub(max_count);

        if truncated_count > 0 {
            info!(
                "Smart ranking: kept top {} of {} entities (truncated {})",
                ranked_entities.len(),
                original_count,
                truncated_count
            );
        }

        (ranked_entities, truncated_count)
    }

    fn auto_generate_relationships(
        &mut self,
        created_entities: &[String],
        referenced_entities: &[String],
        update: &ContextUpdate,
    ) -> anyhow::Result<()> {
        use crate::core::context_update::EntityRelationship;

        // Limit entities to prevent excessive relationship generation
        let max_entities = 10;
        let limited_created: Vec<_> = created_entities.iter().take(max_entities).collect();
        let _limited_referenced: Vec<_> = referenced_entities.iter().take(max_entities).collect();

        // Create relationships between entities in the same update (limit pairs)
        let mut relationship_count = 0;
        let max_relationships = 15;

        for (i, entity1) in limited_created.iter().enumerate() {
            if relationship_count >= max_relationships {
                break;
            }
            for entity2 in limited_created.iter().skip(i + 1) {
                if relationship_count >= max_relationships {
                    break;
                }
                // Safe relationship creation with validation
                if entity1.is_empty() || entity2.is_empty() || entity1 == entity2 {
                    continue;
                }

                let relationship = EntityRelationship {
                    from_entity: entity1.to_string(),
                    to_entity: entity2.to_string(),
                    relation_type: self.infer_relationship_type(&update.update_type),
                    context: format!("Co-mentioned in: {}", update.content.title),
                };
                self.entity_graph.add_relationship(relationship);
                relationship_count += 1;
            }
        }

        Ok(())
    }

    fn infer_relationship_type(&self, update_type: &UpdateType) -> RelationType {
        match update_type {
            UpdateType::ProblemSolved => RelationType::Solves,
            UpdateType::CodeChanged => RelationType::Implements,
            UpdateType::DecisionMade => RelationType::LeadsTo,
            UpdateType::RequirementAdded => RelationType::RequiredBy,
            _ => RelationType::RelatedTo,
        }
    }

    async fn add_code_reference(&mut self, code_ref: &CodeReference) -> anyhow::Result<()> {
        let code_refs = self
            .code_references
            .entry(code_ref.file_path.clone())
            .or_default();
        code_refs.push(code_ref.clone());
        Ok(())
    }

    fn record_change(&mut self, update: &ContextUpdate) -> anyhow::Result<()> {
        self.change_history.push(ChangeRecord {
            id: Uuid::new_v4(),
            timestamp: update.timestamp,
            change_type: format!("{:?}", update.update_type),
            description: update.content.description.clone(),
            related_update_id: Some(update.id),
        });
        Ok(())
    }

    fn maintain_context(&mut self) -> anyhow::Result<()> {
        // Add a safeguard to prevent infinite loops
        let original_hot_len = self.hot_context.len();

        // Move from hot to warm if needed
        if self.hot_context.len() > self.user_preferences.max_hot_context_size {
            self.promote_to_warm()?;
        }

        // Create summary if needed
        if self.should_create_summary() {
            self.create_periodic_summary()?;
        }

        // Compress old data (placeholder for now)
        // self.compress_old_data().await?;

        // Verify we didn't get stuck in a loop
        if self.hot_context.len() > original_hot_len + 1 {
            return Err(anyhow::anyhow!(
                "Context maintenance created more items than expected"
            ));
        }

        Ok(())
    }

    fn promote_to_warm(&mut self) -> anyhow::Result<()> {
        if let Some(oldest_update) = self.hot_context.pop_front() {
            self.warm_context.push(CompressedUpdate {
                update: oldest_update,
                compression_ratio: 1.0, // Placeholder - would calculate actual compression
                compressed_at: Utc::now(),
            });
        }
        Ok(())
    }

    fn should_create_summary(&self) -> bool {
        self.incremental_updates.len() % self.user_preferences.auto_summary_threshold == 0
            && !self.incremental_updates.is_empty()
    }

    fn create_periodic_summary(&mut self) -> anyhow::Result<()> {
        // Create a summary from current state
        let summary = StructuredSummary {
            summary_id: Uuid::new_v4(),
            created_at: Utc::now(),
            context_snapshot: self.current_state.clone(),
            referenced_updates: self.incremental_updates.iter().map(|u| u.id).collect(),
            summary_quality: 1.0, // Placeholder - would calculate actual quality
        };

        self.cold_context.push(summary);
        Ok(())
    }

    /// Update the session name
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
        self.last_updated = Utc::now();
    }

    /// Update the session description
    pub fn set_description(&mut self, description: Option<String>) {
        self.description = description;
        self.last_updated = Utc::now();
    }

    /// Update both name and description (preserves existing values if None provided)
    pub fn update_metadata(&mut self, name: Option<String>, description: Option<String>) {
        if name.is_some() {
            self.name = name;
        }
        if description.is_some() {
            self.description = description;
        }
        self.last_updated = Utc::now();
    }

    /// Get the current name and description
    pub fn get_metadata(&self) -> (Option<String>, Option<String>) {
        (self.name.clone(), self.description.clone())
    }
}
