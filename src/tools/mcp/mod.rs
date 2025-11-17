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
use crate::core::context_update::{CodeReference, ContextUpdate, EntityType, UpdateType};
use crate::core::lockfree_memory_system::{LockFreeConversationMemorySystem, SystemConfig};
use crate::core::structured_context::StructuredContext;
use crate::core::timeout_utils::{with_mcp_timeout, with_storage_timeout};
use crate::session::active_session::ActiveSession;
use crate::storage::rocksdb_storage::SessionCheckpoint;
use crate::summary::SummaryGenerator;

#[cfg(feature = "embeddings")]
use crate::core::content_vectorizer::ContentType;
use anyhow::Result;
use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

// Compatibility alias for existing code
type ConversationMemorySystem = LockFreeConversationMemorySystem;

// Helper function to convert String errors to anyhow::Error
fn string_to_anyhow(s: String) -> anyhow::Error {
    anyhow::Error::msg(s)
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ContextQuery {
    GetRecentChanges {
        since: chrono::DateTime<chrono::Utc>,
    },
    FindCodeReferences {
        file_path: String,
    },
    GetStructuredSummary,
    SearchUpdates {
        query: String,
    },
    GetDecisions {
        since: Option<chrono::DateTime<chrono::Utc>>,
    },
    GetOpenQuestions,
    GetChangeHistory {
        file_path: Option<String>,
    },

    // Graph-based queries
    // Entity queries
    FindRelatedEntities {
        entity_name: String,
    },
    GetEntityContext {
        entity_name: String,
    },
    GetAllEntities {
        entity_type: Option<EntityType>,
    },
    TraceRelationships {
        from_entity: String,
        max_depth: usize,
    },

    // Network analysis
    GetEntityNetwork {
        center_entity: String,
        max_depth: usize,
    },
    FindConnectionPath {
        from_entity: String,
        to_entity: String,
        max_depth: usize,
    },
    GetMostImportantEntities {
        limit: usize,
    },
    GetRecentlyMentionedEntities {
        limit: usize,
    },
    AnalyzeEntityImportance,
    FindEntitiesByType {
        entity_type: EntityType,
    },

    // Hierarchical analysis
    GetEntityHierarchy {
        root_entity: String,
        max_depth: usize,
    },
    FindEntityClusters {
        min_cluster_size: usize,
    },

    // Temporal analysis
    GetEntityTimeline {
        entity_name: String,
        start_time: Option<chrono::DateTime<chrono::Utc>>,
        end_time: Option<chrono::DateTime<chrono::Utc>>,
    },
    AnalyzeEntityTrends {
        time_window_days: i64,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ContextResponse {
    RecentChanges(Vec<ContextUpdate>),
    CodeReferences(Vec<CodeReference>),
    StructuredSummary(StructuredContext),
    SearchResults(Vec<ContextUpdate>),
    Decisions(Vec<ContextUpdate>),
    OpenQuestions(Vec<String>),
    ChangeHistory(Vec<ContextUpdate>),
    RelatedEntities(Vec<String>),
    EntityContext(String),
    AllEntities(Vec<String>),
    EntityRelationships(Vec<String>),
    EntityNetwork(String),
    ConnectionPath(String),
    Entities(Vec<String>),
    ImportanceAnalysis(String),
    EntityHierarchy(String),
    EntityClusters(String),
    EntityTimeline(String),
    EntityTrends(String),
}

// Lock-free global system with lazy initialization using ArcSwap + LazyLock
// ArcSwap provides wait-free reads and lock-free writes, eliminating all Mutex contention
// LazyLock allows const initialization with Arc::new()
static MEMORY_SYSTEM: LazyLock<ArcSwap<Option<Arc<ConversationMemorySystem>>>> =
    LazyLock::new(|| ArcSwap::new(Arc::new(None)));

pub async fn get_memory_system_with_config(
    config: SystemConfig,
) -> Result<ConversationMemorySystem> {
    ConversationMemorySystem::new(config)
        .await
        .map_err(anyhow::Error::msg)
}

pub async fn get_memory_system() -> Result<Arc<ConversationMemorySystem>> {
    info!("MCP-TOOLS: get_memory_system() called");

    // Fast path - lock-free read with ArcSwap (wait-free, no blocking)
    if let Some(system) = MEMORY_SYSTEM.load().as_ref() {
        info!("MCP-TOOLS: Using existing system");
        return Ok(system.clone());
    }

    info!("MCP-TOOLS: System not initialized, proceeding with initialization");

    // Slow path - initialize new system
    let mut config = SystemConfig::default();

    #[cfg(feature = "embeddings")]
    {
        config.enable_embeddings = true;
        config.embeddings_model_type = "StaticSimilarityMRL".to_string();
        config.auto_vectorize_on_update = true;
        config.cross_session_search_enabled = true;
        info!("MCP-TOOLS: Embeddings enabled in config");
    }

    #[cfg(not(feature = "embeddings"))]
    {
        info!("MCP-TOOLS: Embeddings not compiled in");
    }

    info!("MCP-TOOLS: About to call ConversationMemorySystem::new()");
    let system = ConversationMemorySystem::new(config)
        .await
        .map_err(anyhow::Error::msg)?;
    info!("MCP-TOOLS: ConversationMemorySystem created successfully");

    let arc_system = Arc::new(system);

    // Store the system using lock-free compare-and-swap (RCU pattern)
    // This handles race conditions gracefully - if another thread won the race,
    // we use their system instead of ours
    let new_option = Arc::new(Some(arc_system.clone()));
    MEMORY_SYSTEM.rcu(|current| {
        if current.is_none() {
            info!("MCP-TOOLS: Storing newly created system");
            new_option.clone()
        } else {
            info!("MCP-TOOLS: Another thread already initialized the system, using existing");
            current.clone()
        }
    });

    info!("MCP-TOOLS: System initialization completed");

    // Return the system (either our newly created one or the one from the race winner)
    // Load returns Arc<Option<Arc<T>>>, deref to Option<Arc<T>>, then unwrap to Arc<T>
    Ok(MEMORY_SYSTEM.load().as_ref().as_ref().unwrap().clone())
}

#[instrument(skip(system, content), fields(
    session_id = %session_id,
    interaction_type = %interaction_type,
    has_code_reference = code_reference.is_some()
))]
pub async fn update_conversation_context_with_system(
    interaction_type: String,
    content: HashMap<String, String>,
    code_reference: Option<CodeReference>,
    session_id: Uuid,
    system: &ConversationMemorySystem,
) -> Result<MCPToolResult> {
    info!("Starting update_conversation_context_with_system");
    debug!("Parsing interaction type: {}", interaction_type);

    // Wrap entire operation in timeout to prevent infinite hangs
    let result = with_mcp_timeout(async {
        let interaction = match interaction_type.as_str() {
            "qa" => {
                let question = content.get("question").cloned().unwrap_or_default();
                let answer = content.get("answer").cloned().unwrap_or_default();
                Interaction::QA { question, answer }
            }
            "code_change" => {
                let description = content.get("description").cloned().unwrap_or_default();
                let change_type = content.get("change_type").cloned().unwrap_or_default();
                Interaction::CodeChange {
                    file_path: description,
                    diff: change_type,
                }
            }
            "problem_solved" => {
                let problem = content.get("problem").cloned().unwrap_or_default();
                let solution = content.get("solution").cloned().unwrap_or_default();
                Interaction::ProblemSolved { problem, solution }
            }
            "decision_made" => {
                let decision = content.get("decision").cloned().unwrap_or_default();
                let rationale = content.get("rationale").cloned().unwrap_or_default();
                Interaction::DecisionMade {
                    decision,
                    rationale,
                }
            }
            _ => {
                error!("Unknown interaction type: {}", interaction_type);
                return Ok(MCPToolResult::error(format!(
                    "Unknown interaction type: {}",
                    interaction_type
                )));
            }
        };

        debug!("Converting interaction to ContextUpdate...");
        let update = interaction_to_context_update(interaction, code_reference)?;
        info!(
            "Created ContextUpdate with {} entities and {} relationships",
            update.creates_entities.len(),
            update.creates_relationships.len()
        );

        debug!("Adding context update to session: {}", session_id);
        let metadata = Some(
            serde_json::to_value(&update)
                .map_err(|e| anyhow::anyhow!("Failed to serialize update metadata: {}", e))?,
        );
        system
            .add_incremental_update(session_id, update.content.description.clone(), metadata)
            .await
            .map_err(string_to_anyhow)?;
        info!("system.add_context_update completed successfully!");

        Ok(MCPToolResult::success(
            "Context updated successfully".to_string(),
            None,
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: update_conversation_context_with_system - session: {}, error: {}",
                session_id, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "Operation timed out: {}",
                timeout_error
            )))
        }
    }
}

pub async fn query_conversation_context_with_system(
    query_type: String,
    parameters: HashMap<String, String>,
    session_id: Uuid,
    system: &ConversationMemorySystem,
) -> Result<MCPToolResult> {
    eprintln!(
        "DEBUG: query_conversation_context_with_system - Looking for session: {}",
        session_id
    );

    let result = with_mcp_timeout(async {
        let session_arc = match system.get_session(session_id).await {
            Ok(session) => {
                eprintln!(
                    "DEBUG: query_conversation_context_with_system - Session found successfully"
                );
                session
            }
            Err(e) => {
                eprintln!(
                    "DEBUG: query_conversation_context_with_system - Session not found: {}",
                    e
                );
                return Err(anyhow::anyhow!("Session not found: {}", e));
            }
        };
        let session = session_arc.load();
        eprintln!("DEBUG: query_conversation_context_with_system - Session loaded");

        let query = match query_type.as_str() {
            "recent_changes" => {
                let since_str = parameters.get("since").cloned().unwrap_or_default();
                let since = parse_datetime(&since_str)?;
                ContextQuery::GetRecentChanges { since }
            }
            "code_references" => {
                let file_path = parameters.get("file_path").cloned().unwrap_or_default();
                ContextQuery::FindCodeReferences { file_path }
            }
            "structured_summary" => ContextQuery::GetStructuredSummary,
            "decisions" => ContextQuery::GetDecisions { since: None },
            "open_questions" => ContextQuery::GetOpenQuestions,
            "related_entities" => {
                let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
                ContextQuery::FindRelatedEntities { entity_name }
            }
            "entity_context" => {
                let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
                ContextQuery::GetEntityContext { entity_name }
            }
            "all_entities" => ContextQuery::GetAllEntities { entity_type: None },
            "trace_relationships" => {
                let from_entity = parameters.get("entity_name").cloned().unwrap_or_default();
                let max_depth = parameters
                    .get("max_depth")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                ContextQuery::TraceRelationships {
                    from_entity,
                    max_depth,
                }
            }
            "find_related_entities" => {
                let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
                ContextQuery::FindRelatedEntities { entity_name }
            }
            "get_entity_context" => {
                let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
                ContextQuery::GetEntityContext { entity_name }
            }
            "get_entity_network" => {
                let center_entity = parameters.get("entity_name").cloned().unwrap_or_default();
                let max_depth = parameters
                    .get("max_depth")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2);
                ContextQuery::GetEntityNetwork {
                    center_entity,
                    max_depth,
                }
            }
            "get_most_important_entities" => {
                let limit = parameters
                    .get("limit")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                ContextQuery::GetMostImportantEntities { limit }
            }
            "get_recently_mentioned_entities" => {
                let limit = parameters
                    .get("limit")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                ContextQuery::GetRecentlyMentionedEntities { limit }
            }
            "analyze_entity_importance" => ContextQuery::AnalyzeEntityImportance,
            "find_entities_by_type" => {
                let entity_type_str = parameters.get("entity_type").cloned().unwrap_or_default();
                let entity_type = match entity_type_str.as_str() {
                    "technology" => EntityType::Technology,
                    "concept" => EntityType::Concept,
                    "problem" => EntityType::Problem,
                    "solution" => EntityType::Solution,
                    "decision" => EntityType::Decision,
                    "code_component" => EntityType::CodeComponent,
                    _ => EntityType::Concept, // default
                };
                ContextQuery::FindEntitiesByType { entity_type }
            }
            "search_updates" => {
                let query = parameters.get("query").cloned().unwrap_or_default();
                ContextQuery::SearchUpdates { query }
            }
            _ => {
                return Ok(MCPToolResult::error(format!(
                    "Unknown query type: {}",
                    query_type
                )));
            }
        };

        let response = query_context(&session, query).await?;
        let json_response = serde_json::to_value(response)?;

        Ok(MCPToolResult::success(
            "Query successful".to_string(),
            Some(json_response),
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: query_conversation_context_with_system - session: {}, error: {}",
                session_id, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "Query timed out: {}",
                timeout_error
            )))
        }
    }
}

pub async fn create_session_checkpoint_with_system(
    session_id: Uuid,
    system: &ConversationMemorySystem,
) -> Result<MCPToolResult> {
    let session_arc = system
        .get_session(session_id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;
    let session = session_arc.load();

    // Create checkpoint
    let checkpoint = create_comprehensive_checkpoint(&*session).await?;

    // Save checkpoint
    let storage_wrapper = system.storage();
    let storage = storage_wrapper.write().await;
    storage
        .save_checkpoint(&checkpoint)
        .await
        .map_err(string_to_anyhow)?;

    Ok(MCPToolResult::success(
        "Checkpoint created successfully".to_string(),
        Some(serde_json::json!({ "checkpoint_id": checkpoint.id.to_string() })),
    ))
}

pub async fn load_session_checkpoint_with_system(
    checkpoint_id: String,
    session_id: Uuid,
    system: &ConversationMemorySystem,
) -> Result<MCPToolResult> {
    eprintln!("Loading checkpoint - step 1: Parsing checkpoint ID");
    let checkpoint_id = Uuid::parse_str(&checkpoint_id)?;

    eprintln!("Loading checkpoint - step 2: Loading checkpoint from storage");
    let storage_wrapper = system.storage();
    let storage = storage_wrapper.read().await;
    let checkpoint = storage
        .load_checkpoint(checkpoint_id)
        .await
        .map_err(string_to_anyhow)?;
    let _ = storage; // Release storage lock

    eprintln!("Loading checkpoint - step 3: Checkpoint loaded successfully");

    // Restore session from checkpoint
    eprintln!("Loading checkpoint - step 4: Creating session from checkpoint");
    let mut session = ActiveSession::new(session_id, None, None);

    eprintln!("Loading checkpoint - step 5: Restoring current state");
    session.current_state = checkpoint.structured_context;

    eprintln!("Loading checkpoint - step 6: Restoring incremental updates");
    session.incremental_updates = checkpoint.recent_updates;

    eprintln!("Loading checkpoint - step 10: Restoring code references");
    session.code_references = checkpoint.code_references;

    eprintln!("Loading checkpoint - step 11: Restoring change history");
    session.change_history = checkpoint.change_history;

    eprintln!("Loading checkpoint - step 12: Entity graph restored");

    eprintln!("Loading checkpoint - step 13: Adding session to session manager");
    system
        .session_manager
        .sessions
        .put(session_id, Arc::new(ArcSwap::new(Arc::new(session))));

    eprintln!("Loading checkpoint - step 14: Updated session manager");

    Ok(MCPToolResult::success(
        "Session loaded from checkpoint successfully".to_string(),
        None,
    ))
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Interaction {
    QA { question: String, answer: String },
    CodeChange { file_path: String, diff: String },
    ProblemSolved { problem: String, solution: String },
    DecisionMade { decision: String, rationale: String },
}

// pub struct LLMClient {
//     api_key: Option<String>,
//     model: String,
//     base_url: String,
// }

// impl LLMClient {
//     pub fn new(api_key: Option<String>, model: String, base_url: String) -> Self {
//         Self {
//             api_key,
//             model,
//             base_url,
//         }
//     }
// }

#[derive(Serialize, Deserialize, Debug)]
pub struct MCPToolResult {
    pub success: bool,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl MCPToolResult {
    pub fn success(message: String, data: Option<serde_json::Value>) -> Self {
        Self {
            success: true,
            message,
            data,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            message,
            data: None,
        }
    }
}

/// Structure for a single context update in bulk operations
#[derive(Serialize, Deserialize, Debug)]
pub struct ContextUpdateItem {
    pub interaction_type: String,
    pub content: HashMap<String, String>,
    pub code_reference: Option<CodeReference>,
}

/// Bulk update conversation context - add multiple context updates at once
pub async fn bulk_update_conversation_context(
    updates: Vec<ContextUpdateItem>,
    session_id: Uuid,
) -> Result<MCPToolResult> {
    info!(
        "MCP-TOOLS: bulk_update_conversation_context() called with {} updates for session {}",
        updates.len(),
        session_id
    );

    let system = get_memory_system().await?;
    let mut success_count = 0;
    let mut error_count = 0;
    let mut errors: Vec<String> = Vec::new();

    for (index, update_item) in updates.into_iter().enumerate() {
        let interaction = match update_item.interaction_type.as_str() {
            "qa" => {
                let question = update_item
                    .content
                    .get("question")
                    .cloned()
                    .unwrap_or_default();
                let answer = update_item
                    .content
                    .get("answer")
                    .cloned()
                    .unwrap_or_default();
                Interaction::QA { question, answer }
            }
            "code_change" => {
                let description = update_item
                    .content
                    .get("description")
                    .cloned()
                    .unwrap_or_default();
                let change_type = update_item
                    .content
                    .get("change_type")
                    .cloned()
                    .unwrap_or_default();
                Interaction::CodeChange {
                    file_path: description,
                    diff: change_type,
                }
            }
            "problem_solved" => {
                let problem = update_item
                    .content
                    .get("problem")
                    .cloned()
                    .unwrap_or_default();
                let solution = update_item
                    .content
                    .get("solution")
                    .cloned()
                    .unwrap_or_default();
                Interaction::ProblemSolved { problem, solution }
            }
            "decision_made" => {
                let decision = update_item
                    .content
                    .get("decision")
                    .cloned()
                    .unwrap_or_default();
                let rationale = update_item
                    .content
                    .get("rationale")
                    .cloned()
                    .unwrap_or_default();
                Interaction::DecisionMade {
                    decision,
                    rationale,
                }
            }
            _ => {
                error_count += 1;
                errors.push(format!(
                    "Update {}: Unknown interaction type: {}",
                    index, update_item.interaction_type
                ));
                continue;
            }
        };

        // Convert to ContextUpdate
        match interaction_to_context_update(interaction, update_item.code_reference) {
            Ok(update) => {
                let description = update.content.description.clone();
                let metadata = match serde_json::to_value(&update) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        error_count += 1;
                        errors.push(format!(
                            "Update {}: Failed to serialize metadata: {}",
                            index, e
                        ));
                        continue;
                    }
                };

                // Add to session
                match system
                    .add_incremental_update(session_id, description, metadata)
                    .await
                {
                    Ok(_) => {
                        success_count += 1;
                        debug!("Update {} added successfully", index);
                    }
                    Err(e) => {
                        error_count += 1;
                        errors.push(format!("Update {}: Failed to add: {}", index, e));
                    }
                }
            }
            Err(e) => {
                error_count += 1;
                errors.push(format!("Update {}: Failed to convert: {}", index, e));
            }
        }
    }

    let message = if error_count == 0 {
        format!(
            "Bulk update completed successfully: {} updates added",
            success_count
        )
    } else {
        format!(
            "Bulk update completed with errors: {} succeeded, {} failed",
            success_count, error_count
        )
    };

    Ok(MCPToolResult::success(
        message,
        Some(serde_json::json!({
            "success_count": success_count,
            "error_count": error_count,
            "errors": errors
        })),
    ))
}

pub async fn update_conversation_context(
    interaction_type: String,
    content: HashMap<String, String>,
    code_reference: Option<CodeReference>,
    session_id: Uuid,
) -> Result<MCPToolResult> {
    info!("MCP-TOOLS: Getting memory system for create_session");
    info!("MCP-TOOLS: Getting memory system for update_conversation_context");
    let system = get_memory_system().await?;
    info!("MCP-TOOLS: Got memory system for update_conversation_context");
    info!("MCP-TOOLS: Got memory system, creating session");

    let interaction = match interaction_type.as_str() {
        "qa" => {
            let question = content.get("question").cloned().unwrap_or_default();
            let answer = content.get("answer").cloned().unwrap_or_default();
            Interaction::QA { question, answer }
        }
        "code_change" => {
            let description = content.get("description").cloned().unwrap_or_default();
            let change_type = content.get("change_type").cloned().unwrap_or_default();
            Interaction::CodeChange {
                file_path: description,
                diff: change_type,
            }
        }
        "problem_solved" => {
            let problem = content.get("problem").cloned().unwrap_or_default();
            let solution = content.get("solution").cloned().unwrap_or_default();
            Interaction::ProblemSolved { problem, solution }
        }
        "decision_made" => {
            let decision = content.get("decision").cloned().unwrap_or_default();
            let rationale = content.get("rationale").cloned().unwrap_or_default();
            Interaction::DecisionMade {
                decision,
                rationale,
            }
        }
        _ => {
            return Ok(MCPToolResult::error(format!(
                "Unknown interaction type: {}",
                interaction_type
            )));
        }
    };

    // Convert to ContextUpdate
    let update = interaction_to_context_update(interaction, code_reference)?;
    eprintln!(
        "MCP: ContextUpdate created with {} entities",
        update.creates_entities.len()
    );

    // Add to session
    eprintln!(
        "MCP: About to call add_incremental_update for session {}",
        session_id
    );
    let description = update.content.description.clone();
    let metadata = Some(
        serde_json::to_value(&update)
            .map_err(|e| anyhow::anyhow!("Failed to serialize update metadata: {}", e))?,
    );
    system
        .add_incremental_update(session_id, description, metadata)
        .await
        .map_err(string_to_anyhow)?;
    eprintln!(
        "MCP: add_incremental_update completed successfully for session {}",
        session_id
    );

    Ok(MCPToolResult::success(
        "Context updated successfully".to_string(),
        None,
    ))
}

pub async fn query_conversation_context(
    query_type: String,
    parameters: HashMap<String, String>,
    session_id: Uuid,
) -> Result<MCPToolResult> {
    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(session_id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;
    let session = session_arc.load();

    let query = match query_type.as_str() {
        "recent_changes" => {
            let since_str = parameters.get("since").cloned().unwrap_or_default();
            let since = parse_datetime(&since_str)?;
            ContextQuery::GetRecentChanges { since }
        }
        "code_references" => {
            let file_path = parameters.get("file_path").cloned().unwrap_or_default();
            ContextQuery::FindCodeReferences { file_path }
        }
        "structured_summary" => ContextQuery::GetStructuredSummary,
        "decisions" => ContextQuery::GetDecisions { since: None },
        "open_questions" => ContextQuery::GetOpenQuestions,
        "related_entities" => {
            let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
            ContextQuery::FindRelatedEntities { entity_name }
        }
        "entity_context" => {
            let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
            ContextQuery::GetEntityContext { entity_name }
        }
        "all_entities" => ContextQuery::GetAllEntities { entity_type: None },
        "trace_relationships" => {
            let from_entity = parameters.get("entity_name").cloned().unwrap_or_default();
            let max_depth = parameters
                .get("max_depth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            ContextQuery::TraceRelationships {
                from_entity,
                max_depth,
            }
        }
        "find_related_entities" => {
            let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
            ContextQuery::FindRelatedEntities { entity_name }
        }
        "get_entity_context" => {
            let entity_name = parameters.get("entity_name").cloned().unwrap_or_default();
            ContextQuery::GetEntityContext { entity_name }
        }
        "get_entity_network" => {
            let center_entity = parameters.get("entity_name").cloned().unwrap_or_default();
            let max_depth = parameters
                .get("max_depth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(2);
            ContextQuery::GetEntityNetwork {
                center_entity,
                max_depth,
            }
        }
        "get_most_important_entities" => {
            let limit = parameters
                .get("limit")
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            ContextQuery::GetMostImportantEntities { limit }
        }
        "get_recently_mentioned_entities" => {
            let limit = parameters
                .get("limit")
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            ContextQuery::GetRecentlyMentionedEntities { limit }
        }
        "analyze_entity_importance" => ContextQuery::AnalyzeEntityImportance,
        "find_entities_by_type" => {
            let entity_type_str = parameters.get("entity_type").cloned().unwrap_or_default();
            let entity_type = match entity_type_str.as_str() {
                "technology" => EntityType::Technology,
                "concept" => EntityType::Concept,
                "problem" => EntityType::Problem,
                "solution" => EntityType::Solution,
                "decision" => EntityType::Decision,
                "code_component" => EntityType::CodeComponent,
                _ => EntityType::Concept, // default
            };
            ContextQuery::FindEntitiesByType { entity_type }
        }
        "search_updates" => {
            let query = parameters.get("query").cloned().unwrap_or_default();
            ContextQuery::SearchUpdates { query }
        }
        _ => {
            return Ok(MCPToolResult::error(format!(
                "Unknown query type: {}",
                query_type
            )));
        }
    };

    let response = query_context(&session, query).await?;
    let json_response = serde_json::to_value(response)?;

    Ok(MCPToolResult::success(
        "Query successful".to_string(),
        Some(json_response),
    ))
}

pub async fn create_session_checkpoint(session_id: Uuid) -> Result<MCPToolResult> {
    let result = with_storage_timeout(async {
        let system = get_memory_system().await?;
        let session_arc = system
            .get_session(session_id)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;
        let session = session_arc.load();

        // Create checkpoint
        let checkpoint = create_comprehensive_checkpoint(&*session).await?;

        // Save checkpoint
        let storage_wrapper = system.storage();
        let storage = storage_wrapper.write().await;
        storage
            .save_checkpoint(&checkpoint)
            .await
            .map_err(string_to_anyhow)?;

        Ok(MCPToolResult::success(
            "Checkpoint created successfully".to_string(),
            Some(serde_json::json!({ "checkpoint_id": checkpoint.id.to_string() })),
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: create_session_checkpoint - session: {}, error: {}",
                session_id, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "Checkpoint creation timed out: {}",
                timeout_error
            )))
        }
    }
}

pub async fn load_session_checkpoint(
    checkpoint_id: String,
    session_id: Uuid,
) -> Result<MCPToolResult> {
    let result = with_storage_timeout(async {
        let system = get_memory_system().await?;
        let checkpoint_id = Uuid::parse_str(&checkpoint_id)?;

        // Load checkpoint from storage
        let storage_wrapper = system.storage();
        let storage = storage_wrapper.read().await;
        let checkpoint = storage
            .load_checkpoint(checkpoint_id)
            .await
            .map_err(string_to_anyhow)?;
        let _ = storage; // Release storage lock

        // Restore session from checkpoint
        let mut session = ActiveSession::new(session_id, None, None);
        session.current_state = checkpoint.structured_context;
        session.incremental_updates = checkpoint.recent_updates;
        session.code_references = checkpoint.code_references;
        session.change_history = checkpoint.change_history;
        // Entity graph will be rebuilt from incremental updates

        // Add session to session manager
        system
            .session_manager
            .sessions
            .put(session_id, Arc::new(ArcSwap::new(Arc::new(session))));

        Ok(MCPToolResult::success(
            "Session loaded from checkpoint successfully".to_string(),
            None,
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: load_session_checkpoint - session: {}, error: {}",
                session_id, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "Checkpoint loading timed out: {}",
                timeout_error
            )))
        }
    }
}

pub async fn mark_important(session_id: Uuid, update_id: String) -> Result<MCPToolResult> {
    let update_id = Uuid::parse_str(&update_id)?;
    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(session_id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    // Find and mark the update as important using ArcSwap update
    let mut found = false;
    session_arc.rcu(|current| {
        let mut updated = (**current).clone();
        for update in &mut updated.incremental_updates {
            if update.id == update_id {
                update.user_marked_important = true;
                found = true;
                break;
            }
        }
        Arc::new(updated)
    });

    if found {
        Ok(MCPToolResult::success(
            "Update marked as important".to_string(),
            None,
        ))
    } else {
        Ok(MCPToolResult::error("Update not found".to_string()))
    }
}

pub async fn list_sessions_with_storage(
    storage: &crate::storage::rocksdb_storage::RealRocksDBStorage,
) -> Result<MCPToolResult> {
    match storage.list_sessions().await {
        Ok(session_ids) => {
            let mut sessions_info = Vec::new();

            // Load each session to get real metadata
            for session_id in session_ids {
                match storage.load_session(session_id).await {
                    Ok(session) => {
                        sessions_info.push(serde_json::json!({
                            "id": session_id.to_string(),
                            "name": session.name,
                            "description": session.description,
                            "created_at": session.created_at.to_rfc3339(),
                            "last_updated": session.last_updated.to_rfc3339(),
                            "update_count": session.incremental_updates.len(),
                            "entity_count": session.entity_graph.entities.len()
                        }));
                    }
                    Err(_) => {
                        // If we can't load the session, still include basic info
                        sessions_info.push(serde_json::json!({
                            "id": session_id.to_string(),
                            "name": null,
                            "description": null,
                            "created_at": "unknown",
                            "last_updated": "unknown",
                            "update_count": 0,
                            "entity_count": 0
                        }));
                    }
                }
            }
            Ok(MCPToolResult::success(
                format!("Found {} sessions", sessions_info.len()),
                Some(serde_json::json!({
                    "sessions": sessions_info
                })),
            ))
        }
        Err(e) => Ok(MCPToolResult::error(format!(
            "Failed to load sessions: {e}"
        ))),
    }
}

pub async fn list_sessions() -> Result<MCPToolResult> {
    info!("MCP-TOOLS: list_sessions() called");
    let result = with_storage_timeout(async {
        info!("MCP-TOOLS: Getting memory system for list_sessions");
        let system = get_memory_system().await?;
        info!("MCP-TOOLS: Got memory system, listing sessions");
        let session_ids = system.list_sessions().await.map_err(string_to_anyhow)?;

        let mut sessions_info = Vec::new();
        for session_id in session_ids {
            match system.get_session(session_id).await {
                Ok(session_arc) => {
                    let session = session_arc.load();
                    sessions_info.push(serde_json::json!({
                        "id": session_id.to_string(),
                        "name": session.name,
                        "description": session.description,
                        "created_at": session.created_at.to_rfc3339(),
                        "last_updated": session.last_updated.to_rfc3339(),
                        "update_count": session.incremental_updates.len(),
                        "entity_count": session.entity_graph.entities.len()
                    }));
                }
                Err(_) => {
                    sessions_info.push(serde_json::json!({
                        "id": session_id.to_string(),
                        "name": null,
                        "description": null,
                        "created_at": "unknown",
                        "last_updated": "unknown",
                        "update_count": 0,
                        "entity_count": 0
                    }));
                }
            }
        }

        Ok(MCPToolResult::success(
            format!("Found {} sessions", sessions_info.len()),
            Some(serde_json::json!({
                "sessions": sessions_info
            })),
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!("TIMEOUT: list_sessions - error: {timeout_error}");
            Ok(MCPToolResult::error(format!(
                "Session listing timed out: {timeout_error}"
            )))
        }
    }
}

pub async fn load_session_with_system(
    session_id: Uuid,
    system: &ConversationMemorySystem,
) -> Result<MCPToolResult> {
    match system.get_session(session_id).await {
        Ok(session_arc) => {
            let session = session_arc.load();
            Ok(MCPToolResult::success(
                "Session loaded successfully".to_string(),
                Some(serde_json::json!({
                    "session": {
                        "id": session.id.to_string(),
                        "created_at": session.created_at.to_rfc3339(),
                        "last_updated": session.last_updated.to_rfc3339(),
                        "update_count": session.incremental_updates.len(),
                        "entity_count": session.entity_graph.entities.len(),
                        "hot_context_size": session.hot_context.len(),
                        "warm_context_size": session.warm_context.len(),
                        "cold_context_size": session.cold_context.len(),
                        "code_references": session.code_references.keys().collect::<Vec<_>>(),
                        "change_history_count": session.change_history.len()
                    }
                })),
            ))
        }
        Err(e) => Ok(MCPToolResult::error(
            format!("Failed to load session: {e}",),
        )),
    }
}

pub async fn load_session(session_id: Uuid) -> Result<MCPToolResult> {
    let result = with_storage_timeout(async {
        let system = get_memory_system().await?;
        load_session_with_system(session_id, &system).await
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: load_session - session: {}, error: {}",
                session_id, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "Session loading timed out: {}",
                timeout_error
            )))
        }
    }
}

// Search for sessions by name or description
pub async fn search_sessions(query: String) -> Result<MCPToolResult> {
    let result = with_storage_timeout(async {
        let system = get_memory_system().await?;
        let session_ids = system
            .find_sessions_by_name_or_description(&query)
            .await
            .map_err(string_to_anyhow)?;

        // Load full session data for each matching session ID
        let mut sessions = Vec::new();
        for session_id in session_ids {
            if let Ok(session_arc) = system.get_session(session_id).await {
                let session = session_arc.load();
                sessions.push(serde_json::json!({
                    "id": session_id.to_string(),
                    "name": session.name,
                    "description": session.description
                }));
            }
        }

        Ok(MCPToolResult::success(
            format!("Found {} sessions matching '{}'", sessions.len(), query),
            Some(serde_json::json!({
                "sessions": sessions
            })),
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: search_sessions - query: {}, error: {}",
                query, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "Session search timed out: {}",
                timeout_error
            )))
        }
    }
}

#[instrument(skip(session_id), fields(session_id = %session_id))]
pub async fn update_session_metadata(
    session_id: Uuid,
    name: Option<String>,
    description: Option<String>,
) -> Result<MCPToolResult> {
    let result = with_storage_timeout(async {
        let system = get_memory_system().await?;
        system
            .update_session_metadata(session_id, name, description)
            .await
            .map_err(string_to_anyhow)?;

        // Get the final values after update
        let session_arc = system
            .get_session(session_id)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;
        let session = session_arc.load();
        let (final_name, final_description) = session.get_metadata();

        Ok(MCPToolResult::success(
            "Session metadata updated successfully".to_string(),
            Some(serde_json::json!({
                "session_id": session_id.to_string(),
                "name": final_name,
                "description": final_description
            })),
        ))
    })
    .await;

    match result {
        Ok(success_result) => success_result,
        Err(timeout_error) => {
            error!(
                "TIMEOUT: update_session_metadata - session_id: {}, error: {}",
                session_id, timeout_error
            );
            Ok(MCPToolResult::error(format!(
                "TIMEOUT: Failed to update session metadata: {}",
                timeout_error
            )))
        }
    }
}

fn interaction_to_context_update(
    interaction: Interaction,
    code_reference: Option<CodeReference>,
) -> Result<ContextUpdate> {
    let id = Uuid::new_v4();
    let timestamp = chrono::Utc::now();

    let (update_type, content) = match interaction {
        Interaction::QA { question, answer } => (
            UpdateType::QuestionAnswered,
            crate::core::context_update::UpdateContent {
                title: question.clone(),
                description: answer.clone(),
                details: vec![],
                examples: vec![],
                implications: vec![],
            },
        ),
        Interaction::CodeChange { file_path, diff } => (
            UpdateType::CodeChanged,
            crate::core::context_update::UpdateContent {
                title: file_path.clone(),
                description: diff.clone(),
                details: vec![],
                examples: vec![],
                implications: vec!["Code functionality updated".to_string()],
            },
        ),
        Interaction::ProblemSolved { problem, solution } => (
            UpdateType::ProblemSolved,
            crate::core::context_update::UpdateContent {
                title: problem.clone(),
                description: solution.clone(),
                details: vec![],
                examples: vec![],
                implications: vec!["Problem resolved".to_string()],
            },
        ),
        Interaction::DecisionMade {
            decision,
            rationale,
        } => (
            UpdateType::DecisionMade,
            crate::core::context_update::UpdateContent {
                title: decision.clone(),
                description: rationale.clone(),
                details: vec![],
                examples: vec![],
                implications: vec!["Decision recorded".to_string()],
            },
        ),
    };

    Ok(ContextUpdate {
        id,
        timestamp,
        update_type,
        content,
        related_code: code_reference,
        parent_update: None,
        user_marked_important: false,
        creates_entities: vec![],
        creates_relationships: vec![],
        references_entities: vec![],
    })
}

async fn query_context(session: &ActiveSession, query: ContextQuery) -> Result<ContextResponse> {
    match query {
        ContextQuery::GetRecentChanges { since } => {
            let recent_updates: Vec<ContextUpdate> = session
                .hot_context
                .iter()
                .chain(session.warm_context.iter().map(|c| &c.update))
                .filter(|u| u.timestamp >= since)
                .cloned()
                .collect();
            Ok(ContextResponse::RecentChanges(recent_updates))
        }
        ContextQuery::FindCodeReferences { file_path } => {
            let refs = session
                .code_references
                .get(&file_path)
                .cloned()
                .unwrap_or_default();
            let converted_refs: Vec<CodeReference> = refs
                .into_iter()
                .map(|r| CodeReference {
                    file_path: r.file_path,
                    start_line: r.start_line,
                    end_line: r.end_line,
                    code_snippet: r.code_snippet,
                    commit_hash: r.commit_hash,
                    branch: r.branch,
                    change_description: r.change_description,
                })
                .collect();
            Ok(ContextResponse::CodeReferences(converted_refs))
        }
        ContextQuery::GetStructuredSummary => Ok(ContextResponse::StructuredSummary(
            session.current_state.clone(),
        )),
        ContextQuery::FindRelatedEntities { entity_name } => {
            let related = session.entity_graph.find_related_entities(&entity_name);
            Ok(ContextResponse::RelatedEntities(related))
        }
        ContextQuery::GetEntityContext { entity_name } => {
            let context = session.entity_graph.get_entity_context(&entity_name);
            Ok(ContextResponse::EntityContext(
                context.unwrap_or("Entity not found".to_string()),
            ))
        }
        ContextQuery::GetAllEntities { entity_type } => {
            let entities: Vec<String> = match entity_type {
                Some(et) => session
                    .entity_graph
                    .get_entities_by_type(&et)
                    .into_iter()
                    .map(|e| e.name.clone())
                    .collect(),
                None => session
                    .entity_graph
                    .get_most_important_entities(50)
                    .into_iter()
                    .map(|e| e.name.clone())
                    .collect(),
            };
            Ok(ContextResponse::AllEntities(entities))
        }
        ContextQuery::TraceRelationships {
            from_entity,
            max_depth,
        } => {
            let trace = session
                .entity_graph
                .trace_entity_relationships(&from_entity, max_depth);
            Ok(ContextResponse::Entities(
                trace.into_iter().map(|(a, _, _)| a).collect(),
            ))
        }
        ContextQuery::GetEntityNetwork {
            center_entity,
            max_depth,
        } => {
            let _network = session
                .entity_graph
                .get_entity_network(&center_entity, max_depth);
            Ok(ContextResponse::EntityNetwork("Network data".to_string()))
        }
        ContextQuery::GetMostImportantEntities { limit } => {
            let entities = session.entity_graph.get_most_important_entities(limit);
            Ok(ContextResponse::Entities(
                entities.into_iter().map(|e| e.name.clone()).collect(),
            ))
        }
        ContextQuery::GetRecentlyMentionedEntities { limit } => {
            let entities = session.entity_graph.get_recently_mentioned_entities(limit);
            Ok(ContextResponse::Entities(
                entities.into_iter().map(|e| e.name.clone()).collect(),
            ))
        }
        ContextQuery::AnalyzeEntityImportance => {
            let _analysis = session.entity_graph.analyze_entity_importance();
            Ok(ContextResponse::ImportanceAnalysis(
                "Analysis complete".to_string(),
            ))
        }
        ContextQuery::FindEntitiesByType { entity_type } => {
            let entities = session.entity_graph.get_entities_by_type(&entity_type);
            Ok(ContextResponse::Entities(
                entities.into_iter().map(|e| e.name.clone()).collect(),
            ))
        }
        ContextQuery::SearchUpdates { query } => {
            let update_results: Vec<ContextUpdate> = session
                .hot_context
                .iter()
                .chain(session.warm_context.iter().map(|c| &c.update))
                .filter(|u| {
                    u.content
                        .title
                        .to_lowercase()
                        .contains(&query.to_lowercase())
                        || u.content
                            .description
                            .to_lowercase()
                            .contains(&query.to_lowercase())
                })
                .cloned()
                .collect();
            Ok(ContextResponse::SearchResults(update_results))
        }
        ContextQuery::GetDecisions { since: _ } => {
            let decisions: Vec<ContextUpdate> = session
                .hot_context
                .iter()
                .filter(|u| matches!(u.update_type, UpdateType::DecisionMade))
                .cloned()
                .collect();
            Ok(ContextResponse::Decisions(decisions))
        }
        ContextQuery::GetOpenQuestions => Ok(ContextResponse::OpenQuestions(vec![
            "No open questions".to_string(),
        ])),
        ContextQuery::GetChangeHistory { file_path: _ } => {
            let changes: Vec<ContextUpdate> = session
                .hot_context
                .iter()
                .filter(|u| matches!(u.update_type, UpdateType::CodeChanged))
                .cloned()
                .collect();
            Ok(ContextResponse::ChangeHistory(changes))
        }
        _ => Ok(ContextResponse::Entities(vec![
            "Not implemented".to_string(),
        ])),
    }
}

async fn create_comprehensive_checkpoint(session: &ActiveSession) -> Result<SessionCheckpoint> {
    Ok(SessionCheckpoint {
        id: Uuid::new_v4(),
        session_id: session.id,
        created_at: chrono::Utc::now(),
        structured_context: session.current_state.clone(),
        recent_updates: session.incremental_updates.clone(),
        code_references: session.code_references.clone(),
        change_history: session.change_history.clone(),
        total_updates: session.incremental_updates.len(),
        context_quality_score: 1.0,
        compression_ratio: 1.0,
    })
}

fn parse_datetime(date_str: &str) -> Result<chrono::DateTime<chrono::Utc>> {
    if date_str.is_empty() {
        return Ok(chrono::Utc::now() - chrono::Duration::days(30));
    }

    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(date_str) {
        return Ok(dt.with_timezone(&chrono::Utc));
    }

    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc());
    }

    if let Ok(dt) = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        return dt
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("Invalid time components for date: {}", date_str))
            .map(|dt| dt.and_utc());
    }

    Err(anyhow::anyhow!("Failed to parse datetime: {}", date_str))
}

/// Get structured summary of a session using existing data
pub async fn get_structured_summary(
    session_id: String,
    decisions_limit: Option<usize>,
    entities_limit: Option<usize>,
    questions_limit: Option<usize>,
    concepts_limit: Option<usize>,
    min_confidence: Option<f32>,
    compact: Option<bool>,
) -> Result<MCPToolResult> {
    let uuid =
        Uuid::parse_str(&session_id).map_err(|e| anyhow::anyhow!("Invalid session ID: {}", e))?;

    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(uuid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    let session = session_arc.load();
    let generator = SummaryGenerator::new();

    // Build summary options
    use crate::summary::SummaryOptions;
    let user_requested_compact = compact.unwrap_or(false);
    let options = if user_requested_compact {
        SummaryOptions::compact()
    } else {
        SummaryOptions {
            decisions_limit,
            entities_limit,
            questions_limit,
            concepts_limit,
            min_confidence,
            compact: false,
        }
    };

    let mut summary = generator.generate_structured_summary_filtered(&session, &options);
    let mut auto_compacted = false;

    // Auto-compact detection: if summary > 25K tokens, enable compact mode
    if !user_requested_compact {
        // Create a test MCPToolResult to estimate full response size
        let test_message = format!(
            "Generated structured summary (decisions: {}, entities: {}, questions: {}, concepts: {})",
            summary.key_decisions.len(),
            summary.entity_summaries.len(),
            summary.open_questions.len(),
            summary.key_concepts.len()
        );
        let test_result =
            MCPToolResult::success(test_message.clone(), Some(serde_json::to_value(&summary)?));
        let full_json = serde_json::to_string(&test_result)?;

        // Conservative estimate: 1 token ≈ 3 characters for JSON (includes wrapper overhead)
        let estimated_tokens = full_json.len() / 3;
        const MAX_TOKENS: usize = 25_000;

        if estimated_tokens > MAX_TOKENS {
            log::warn!(
                "Summary too large ({} estimated tokens > {} max, {} chars). Auto-enabling compact mode.",
                estimated_tokens,
                MAX_TOKENS,
                full_json.len()
            );
            // Re-generate with compact mode
            let compact_options = SummaryOptions::compact();
            summary = generator.generate_structured_summary_filtered(&session, &compact_options);
            auto_compacted = true;
        }
    }

    let message = if user_requested_compact {
        "Generated compact structured summary".to_string()
    } else if auto_compacted {
        format!(
            "Auto-compacted summary (was too large for MCP). Showing: {} decisions, {} entities, {} questions, {} concepts",
            summary.key_decisions.len(),
            summary.entity_summaries.len(),
            summary.open_questions.len(),
            summary.key_concepts.len()
        )
    } else {
        format!(
            "Generated structured summary (decisions: {}, entities: {}, questions: {}, concepts: {})",
            summary.key_decisions.len(),
            summary.entity_summaries.len(),
            summary.open_questions.len(),
            summary.key_concepts.len()
        )
    };

    Ok(MCPToolResult::success(
        message,
        Some(serde_json::to_value(summary)?),
    ))
}

/// Get key decisions from a session
pub async fn get_key_decisions(session_id: String) -> Result<MCPToolResult> {
    let uuid =
        Uuid::parse_str(&session_id).map_err(|e| anyhow::anyhow!("Invalid session ID: {}", e))?;

    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(uuid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    let session = session_arc.load();
    let generator = SummaryGenerator::new();
    let decisions = generator.extract_decision_timeline(&session);

    Ok(MCPToolResult::success(
        format!("Found {} key decisions", decisions.len()),
        Some(serde_json::to_value(decisions)?),
    ))
}

/// Get key insights from session data
pub async fn get_key_insights(session_id: String, limit: Option<usize>) -> Result<MCPToolResult> {
    let uuid =
        Uuid::parse_str(&session_id).map_err(|e| anyhow::anyhow!("Invalid session ID: {}", e))?;

    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(uuid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    let session = session_arc.load();
    let generator = SummaryGenerator::new();
    let insights = generator.extract_key_insights(&session, limit.unwrap_or(5));

    Ok(MCPToolResult::success(
        format!("Found {} key insights", insights.len()),
        Some(serde_json::to_value(insights)?),
    ))
}

/// Get entity importance analysis from existing data
pub async fn get_entity_importance_analysis(
    session_id: String,
    limit: Option<usize>,
    min_importance: Option<f32>,
) -> Result<MCPToolResult> {
    let uuid =
        Uuid::parse_str(&session_id).map_err(|e| anyhow::anyhow!("Invalid session ID: {}", e))?;

    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(uuid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    let session = session_arc.load();
    let mut analysis = session.entity_graph.analyze_entity_importance();

    let total_entities = analysis.len();

    // Apply min_importance filter if specified
    if let Some(min_imp) = min_importance {
        analysis.retain(|entity| entity.importance_score >= min_imp);
    }

    let after_filter = analysis.len();

    // Apply limit if specified (default: 100 for manageable output)
    let entity_limit = limit.unwrap_or(100);
    let truncated = analysis.len() > entity_limit;
    if truncated {
        analysis.truncate(entity_limit);
    }

    // Add pagination metadata
    let result = serde_json::json!({
        "entities": analysis,
        "pagination": {
            "total_entities": total_entities,
            "after_filter": after_filter,
            "returned": analysis.len(),
            "limit": entity_limit,
            "min_importance_threshold": min_importance,
            "truncated": truncated
        }
    });

    Ok(MCPToolResult::success(
        "Generated entity importance analysis with pagination".to_string(),
        Some(result),
    ))
}

/// Get entity network view centered on an entity with pagination
pub async fn get_entity_network_view(
    session_id: String,
    center_entity: Option<String>,
    max_entities: Option<usize>,
    max_relationships: Option<usize>,
) -> Result<MCPToolResult> {
    let uuid =
        Uuid::parse_str(&session_id).map_err(|e| anyhow::anyhow!("Invalid session ID: {}", e))?;

    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(uuid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    let session = session_arc.load();
    let mut network = match center_entity {
        Some(entity) => session.entity_graph.get_entity_network(&entity, 2),
        None => {
            let top_entities = session.entity_graph.get_most_important_entities(1);
            if let Some(top) = top_entities.first() {
                session.entity_graph.get_entity_network(&top.name, 2)
            } else {
                return Ok(MCPToolResult::error(
                    "No entities found in session".to_string(),
                ));
            }
        }
    };

    // Apply pagination limits
    let max_entities_limit = max_entities.unwrap_or(50); // Default: 50 entities
    let max_relationships_limit = max_relationships.unwrap_or(100); // Default: 100 relationships

    let total_entities = network.entities.len();
    let total_relationships = network.relationships.len();

    // Truncate entities if needed (keep most important ones based on importance_score)
    if network.entities.len() > max_entities_limit {
        let mut sorted_entities: Vec<_> = network.entities.iter().collect();
        sorted_entities.sort_by(|a, b| {
            b.1.importance_score
                .partial_cmp(&a.1.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let kept_entities: std::collections::HashSet<String> = sorted_entities
            .into_iter()
            .take(max_entities_limit)
            .map(|(name, _)| name.clone())
            .collect();

        // Filter entities
        network
            .entities
            .retain(|name, _| kept_entities.contains(name));

        // Filter relationships to only include kept entities
        network.relationships.retain(|rel| {
            kept_entities.contains(&rel.from_entity) && kept_entities.contains(&rel.to_entity)
        });
    }

    // Truncate relationships if still needed
    if network.relationships.len() > max_relationships_limit {
        network.relationships.truncate(max_relationships_limit);
    }

    // Add pagination metadata
    let mut result = serde_json::to_value(network)?;

    // Calculate returned counts before taking mutable reference
    let returned_entities = result["entities"].as_object().map(|o| o.len()).unwrap_or(0);
    let returned_relationships = result["relationships"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);

    if let Some(obj) = result.as_object_mut() {
        obj.insert(
            "pagination".to_string(),
            serde_json::json!({
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "returned_entities": returned_entities,
                "returned_relationships": returned_relationships,
                "max_entities_limit": max_entities_limit,
                "max_relationships_limit": max_relationships_limit,
                "truncated": total_entities > max_entities_limit || total_relationships > max_relationships_limit
            }),
        );
    }

    Ok(MCPToolResult::success(
        "Generated entity network view with pagination".to_string(),
        Some(result),
    ))
}

/// Get session statistics from existing data
pub async fn get_session_statistics(session_id: String) -> Result<MCPToolResult> {
    let uuid =
        Uuid::parse_str(&session_id).map_err(|e| anyhow::anyhow!("Invalid session ID: {}", e))?;

    let system = get_memory_system().await?;
    let session_arc = system
        .get_session(uuid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load session: {}", e))?;

    let session = session_arc.load();
    let generator = SummaryGenerator::new();
    let summary = generator.generate_structured_summary(&session);

    Ok(MCPToolResult::success(
        "Generated session statistics".to_string(),
        Some(serde_json::to_value(summary.session_stats)?),
    ))
}

/// Get comprehensive catalog of all available tools organized by category
pub async fn get_tool_catalog() -> Result<MCPToolResult> {
    let catalog = serde_json::json!({
        "total_tools": 20,
        "categories": {
            "Session Management": {
                "description": "Tools for creating, loading, and managing conversation sessions",
                "tool_count": 5,
                "tools": [
                    {
                        "name": "create_session",
                        "description": "Start new conversation session with optional name/description",
                        "use_when": "Beginning a new project, conversation, or knowledge context"
                    },
                    {
                        "name": "load_session",
                        "description": "Resume existing session into active memory",
                        "use_when": "Continuing work on a previous conversation or project"
                    },
                    {
                        "name": "list_sessions",
                        "description": "View all sessions with metadata and statistics",
                        "use_when": "Finding available sessions or checking session details"
                    },
                    {
                        "name": "search_sessions",
                        "description": "Find sessions by name or description (text search)",
                        "use_when": "Looking for specific sessions when you know the name/topic"
                    },
                    {
                        "name": "update_session_metadata",
                        "description": "Change session name or description",
                        "use_when": "Renaming or updating session information"
                    }
                ]
            },
            "Context Operations": {
                "description": "Tools for adding and querying conversation knowledge",
                "tool_count": 3,
                "tools": [
                    {
                        "name": "update_conversation_context",
                        "description": "Add knowledge: QA, decisions, problems solved, code changes",
                        "use_when": "Storing information for future retrieval and learning",
                        "interaction_types": ["qa", "decision_made", "problem_solved", "code_change"]
                    },
                    {
                        "name": "query_conversation_context",
                        "description": "Search using entities, keywords, or structured queries",
                        "use_when": "Finding exact entities or keyword-based searches (faster than semantic)",
                        "query_types": ["find_related_entities", "get_entity_context", "search_updates", "get_most_important_entities"]
                    },
                    {
                        "name": "create_session_checkpoint",
                        "description": "Create snapshot of current session state",
                        "use_when": "Creating restore points before major changes"
                    }
                ]
            },
            "Semantic Search": {
                "description": "AI-powered conceptual search using embeddings (requires embeddings feature)",
                "tool_count": 5,
                "tools": [
                    {
                        "name": "semantic_search_session",
                        "description": "AI search within session - auto-loads and auto-vectorizes!",
                        "use_when": "Finding information by concept/meaning, not exact keywords",
                        "note": "Automatically vectorizes on first use - no manual setup needed"
                    },
                    {
                        "name": "semantic_search_global",
                        "description": "AI search across ALL sessions",
                        "use_when": "Finding related knowledge across multiple conversations"
                    },
                    {
                        "name": "find_related_content",
                        "description": "Discover connections between different sessions",
                        "use_when": "Finding similar discussions from other conversations"
                    },
                    {
                        "name": "vectorize_session",
                        "description": "Generate embeddings for semantic search (optional - auto-called)",
                        "use_when": "Rarely needed - semantic_search_session auto-vectorizes",
                        "note": "Only useful for batch processing or manual control"
                    },
                    {
                        "name": "get_vectorization_stats",
                        "description": "Check embedding statistics and performance metrics",
                        "use_when": "Debugging or monitoring semantic search system"
                    }
                ]
            },
            "Analysis & Insights": {
                "description": "Tools for extracting insights, summaries, and analyzing session data",
                "tool_count": 7,
                "tools": [
                    {
                        "name": "get_structured_summary",
                        "description": "Comprehensive summary with auto-compact for large sessions",
                        "use_when": "Getting overview of decisions, entities, questions, concepts",
                        "note": "Auto-compacts if response > 25K tokens - prevents MCP overflow"
                    },
                    {
                        "name": "get_key_decisions",
                        "description": "Timeline of decisions with confidence levels",
                        "use_when": "Reviewing architectural choices and their rationale"
                    },
                    {
                        "name": "get_key_insights",
                        "description": "Extract top insights from session data",
                        "use_when": "Quick overview of most important discoveries"
                    },
                    {
                        "name": "get_entity_importance_analysis",
                        "description": "Entity rankings with mention counts and relationships",
                        "use_when": "Understanding which concepts/technologies are most central"
                    },
                    {
                        "name": "get_entity_network_view",
                        "description": "Visualize entity relationships as network graph",
                        "use_when": "Understanding how concepts connect to each other"
                    },
                    {
                        "name": "get_session_statistics",
                        "description": "Session metrics: updates, entities, activity level, duration",
                        "use_when": "Checking session health and growth metrics"
                    },
                    {
                        "name": "get_tool_catalog",
                        "description": "View all 20 tools organized by category with usage guidance",
                        "use_when": "First time using post-cortex or discovering available tools",
                        "note": "Returns categories, workflows, tips, and most-used tools list"
                    }
                ]
            }
        },
        "getting_started": [
            "1. create_session → get session_id",
            "2. update_conversation_context → add knowledge (qa, decisions, problems, code)",
            "3. semantic_search_session → AI-powered search (auto-vectorizes on first use!)",
            "4. get_structured_summary → comprehensive overview"
        ],
        "most_used_tools": [
            "create_session",
            "load_session",
            "update_conversation_context",
            "semantic_search_session",
            "get_structured_summary"
        ],
        "tips": [
            "💡 Semantic search auto-vectorizes - no manual vectorize_session needed!",
            "💡 Use semantic_search for concepts, query_conversation_context for keywords",
            "💡 get_structured_summary has auto-compact - safe for large sessions",
            "💡 Similarity scores: 0.65-0.75 = excellent, 0.45-0.55 = good, <0.30 = weak",
            "💡 All sessions persist across Claude Code conversations"
        ]
    });

    Ok(MCPToolResult::success(
        "Retrieved tool catalog with 20 tools across 4 categories".to_string(),
        Some(catalog),
    ))
}

// Semantic Search and Embeddings Tools (requires embeddings feature)

/// Helper function to map interaction type strings to ContentType enums
#[cfg(feature = "embeddings")]
fn interaction_type_to_content_type(interaction_type: &str) -> Option<ContentType> {
    match interaction_type {
        "qa" => Some(ContentType::UserMessage),
        "code_change" => Some(ContentType::CodeSnippet),
        "decision_made" => Some(ContentType::DecisionPoint),
        "problem_solved" => Some(ContentType::UpdateContent),
        _ => None,
    }
}

/// Perform semantic search across all sessions
#[cfg(feature = "embeddings")]
pub async fn semantic_search_global(
    query: String,
    limit: Option<usize>,
    date_from: Option<String>,
    date_to: Option<String>,
    interaction_type: Option<Vec<String>>,
) -> Result<MCPToolResult> {
    info!(
        "MCP-TOOLS: semantic_search_global() called with query: '{}'",
        query
    );
    let system = get_memory_system().await?;
    info!("MCP-TOOLS: Got memory system for semantic_search_global");

    if !system.config.enable_embeddings {
        return Ok(MCPToolResult::error(
            "Embeddings not enabled or initialized".to_string(),
        ));
    }

    // Parse date range if provided
    let date_range = match (date_from, date_to) {
        (Some(from_str), Some(to_str)) => {
            let from = chrono::DateTime::parse_from_rfc3339(&from_str)
                .map_err(|e| anyhow::anyhow!("Invalid date_from format: {}", e))?
                .with_timezone(&chrono::Utc);
            let to = chrono::DateTime::parse_from_rfc3339(&to_str)
                .map_err(|e| anyhow::anyhow!("Invalid date_to format: {}", e))?
                .with_timezone(&chrono::Utc);
            Some((from, to))
        }
        (Some(_), None) | (None, Some(_)) => {
            return Ok(MCPToolResult::error(
                "Both date_from and date_to must be provided together".to_string(),
            ));
        }
        (None, None) => None,
    };

    match system
        .semantic_search_global(&query, limit, date_range)
        .await
    {
        Ok(mut results) => {
            let results_before_filter = results.len();

            // Filter by interaction_type if provided
            if let Some(ref types) = interaction_type {
                info!("Filtering by interaction_types: {:?}", types);

                // Convert interaction types to ContentType
                let content_types: Vec<ContentType> = types
                    .iter()
                    .filter_map(|t| {
                        let ct = interaction_type_to_content_type(t);
                        info!("Mapping '{}' -> {:?}", t, ct);
                        ct
                    })
                    .collect();

                info!("Mapped to ContentTypes: {:?}", content_types);

                if !content_types.is_empty() {
                    results.retain(|r| content_types.contains(&r.content_type));
                    info!(
                        "Filtered from {} to {} results",
                        results_before_filter,
                        results.len()
                    );
                } else {
                    info!("No valid ContentTypes mapped, skipping filter");
                }
            } else {
                info!("No interaction_type filter provided");
            }

            let search_results: Vec<serde_json::Value> = results
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "content_id": r.content_id,
                        "session_id": r.session_id.to_string(),
                        "content_type": format!("{:?}", r.content_type),
                        "text_content": r.text_content,
                        "similarity_score": r.similarity_score,
                        "similarity_quality": r.similarity_quality(),
                        "importance_score": r.importance_score,
                        "timestamp": r.timestamp.to_rfc3339(),
                        "combined_score": r.combined_score,
                        "score_explanation": r.score_explanation()
                    })
                })
                .collect();

            Ok(MCPToolResult::success(
                format!("Found {} semantic search results", search_results.len()),
                Some(serde_json::json!({
                    "query": query,
                    "results": search_results,
                    "scoring_info": {
                        "algorithm": "combined_score = (similarity_score × 0.7) + (importance_weight × 0.3)",
                        "quality_levels": {
                            "Excellent": "≥ 0.85",
                            "Very Good": "0.70 - 0.84",
                            "Good": "0.55 - 0.69",
                            "Moderate": "0.40 - 0.54",
                            "Fair": "0.30 - 0.39",
                            "Weak": "< 0.30"
                        }
                    }
                })),
            ))
        }
        Err(e) => Ok(MCPToolResult::error(format!(
            "Semantic search failed: {}",
            e
        ))),
    }
}

/// Perform semantic search within a specific session
#[cfg(feature = "embeddings")]
pub async fn semantic_search_session(
    session_id: Uuid,
    query: String,
    limit: Option<usize>,
    date_from: Option<String>,
    date_to: Option<String>,
    interaction_type: Option<Vec<String>>,
) -> Result<MCPToolResult> {
    info!(
        "MCP-TOOLS: semantic_search_session() called for session {} with query: '{}'",
        session_id, query
    );
    let system = get_memory_system().await?;
    info!("MCP-TOOLS: Got memory system for semantic_search_session");

    if !system.config.enable_embeddings {
        return Ok(MCPToolResult::error(
            "Embeddings not enabled or initialized".to_string(),
        ));
    }

    // Parse date range if provided
    let date_range = match (date_from, date_to) {
        (Some(from_str), Some(to_str)) => {
            let from = chrono::DateTime::parse_from_rfc3339(&from_str)
                .map_err(|e| anyhow::anyhow!("Invalid date_from format: {}", e))?
                .with_timezone(&chrono::Utc);
            let to = chrono::DateTime::parse_from_rfc3339(&to_str)
                .map_err(|e| anyhow::anyhow!("Invalid date_to format: {}", e))?
                .with_timezone(&chrono::Utc);
            Some((from, to))
        }
        (Some(_), None) | (None, Some(_)) => {
            return Ok(MCPToolResult::error(
                "Both date_from and date_to must be provided together".to_string(),
            ));
        }
        (None, None) => None,
    };

    match system
        .semantic_search_session(session_id, &query, limit, date_range)
        .await
    {
        Ok(mut results) => {
            let results_before_filter = results.len();

            // Filter by interaction_type if provided
            if let Some(ref types) = interaction_type {
                info!("Filtering by interaction_types: {:?}", types);

                // Convert interaction types to ContentType
                let content_types: Vec<ContentType> = types
                    .iter()
                    .filter_map(|t| {
                        let ct = interaction_type_to_content_type(t);
                        info!("Mapping '{}' -> {:?}", t, ct);
                        ct
                    })
                    .collect();

                info!("Mapped to ContentTypes: {:?}", content_types);

                if !content_types.is_empty() {
                    results.retain(|r| content_types.contains(&r.content_type));
                    info!(
                        "Filtered from {} to {} results",
                        results_before_filter,
                        results.len()
                    );
                } else {
                    info!("No valid ContentTypes mapped, skipping filter");
                }
            } else {
                info!("No interaction_type filter provided");
            }

            let search_results: Vec<serde_json::Value> = results
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "content_id": r.content_id,
                        "session_id": r.session_id.to_string(),
                        "content_type": format!("{:?}", r.content_type),
                        "text_content": r.text_content,
                        "similarity_score": r.similarity_score,
                        "similarity_quality": r.similarity_quality(),
                        "importance_score": r.importance_score,
                        "timestamp": r.timestamp.to_rfc3339(),
                        "combined_score": r.combined_score,
                        "score_explanation": r.score_explanation()
                    })
                })
                .collect();

            Ok(MCPToolResult::success(
                format!(
                    "Found {} semantic search results in session",
                    search_results.len()
                ),
                Some(serde_json::json!({
                    "session_id": session_id.to_string(),
                    "query": query,
                    "results": search_results,
                    "scoring_info": {
                        "algorithm": "combined_score = (similarity_score × 0.7) + (importance_weight × 0.3)",
                        "quality_levels": {
                            "Excellent": "≥ 0.85",
                            "Very Good": "0.70 - 0.84",
                            "Good": "0.55 - 0.69",
                            "Moderate": "0.40 - 0.54",
                            "Fair": "0.30 - 0.39",
                            "Weak": "< 0.30"
                        }
                    }
                })),
            ))
        }
        Err(e) => Ok(MCPToolResult::error(format!(
            "Session semantic search failed: {}",
            e
        ))),
    }
}

/// Find related content across sessions
#[cfg(feature = "embeddings")]
pub async fn find_related_content(
    session_id: Uuid,
    topic: String,
    limit: Option<usize>,
) -> Result<MCPToolResult> {
    info!(
        "MCP-TOOLS: find_related_content() called for session {} with topic: '{}'",
        session_id, topic
    );
    let system = get_memory_system().await?;
    info!("MCP-TOOLS: Got memory system for find_related_content");

    if !system.config.enable_embeddings {
        return Ok(MCPToolResult::error(
            "Embeddings not enabled or initialized".to_string(),
        ));
    }

    match system.find_related_content(session_id, &topic, limit).await {
        Ok(results) => {
            let related_content: Vec<serde_json::Value> = results
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "content_id": r.content_id,
                        "session_id": r.session_id.to_string(),
                        "content_type": format!("{:?}", r.content_type),
                        "text_content": r.text_content,
                        "similarity_score": r.similarity_score,
                        "importance_score": r.importance_score,
                        "timestamp": r.timestamp.to_rfc3339(),
                        "combined_score": r.combined_score
                    })
                })
                .collect();

            Ok(MCPToolResult::success(
                format!("Found {} related content items", related_content.len()),
                Some(serde_json::json!({
                    "session_id": session_id.to_string(),
                    "topic": topic,
                    "related_content": related_content
                })),
            ))
        }
        Err(e) => Ok(MCPToolResult::error(format!(
            "Related content search failed: {}",
            e
        ))),
    }
}

/// Vectorize a session's content
#[cfg(feature = "embeddings")]
pub async fn vectorize_session(session_id: Uuid) -> Result<MCPToolResult> {
    info!(
        "MCP-TOOLS: vectorize_session() called for session {}",
        session_id
    );
    let system = get_memory_system().await?;
    info!("MCP-TOOLS: Got memory system for vectorize_session");

    // Check if embeddings are enabled in config (lazy init will happen automatically)
    if !system.config.enable_embeddings {
        return Ok(MCPToolResult::error(
            "Embeddings not enabled in configuration".to_string(),
        ));
    }

    match system.vectorize_session(session_id).await {
        Ok(count) => Ok(MCPToolResult::success(
            format!("Successfully vectorized {} items", count),
            Some(serde_json::json!({
                "session_id": session_id.to_string(),
                "vectorized_count": count
            })),
        )),
        Err(e) => Ok(MCPToolResult::error(format!("Vectorization failed: {}", e))),
    }
}

/// Get vectorization statistics
#[cfg(feature = "embeddings")]
pub async fn get_vectorization_stats() -> Result<MCPToolResult> {
    info!("MCP-TOOLS: get_vectorization_stats() called");
    let system = get_memory_system().await?;
    info!("MCP-TOOLS: Got memory system for get_vectorization_stats");

    if !system.config.enable_embeddings {
        return Ok(MCPToolResult::error(
            "Embeddings not enabled or initialized".to_string(),
        ));
    }

    match system.get_vectorization_stats() {
        Ok(stats) => Ok(MCPToolResult::success(
            "Retrieved vectorization statistics".to_string(),
            Some(serde_json::json!({
                "stats": stats
            })),
        )),
        Err(e) => Ok(MCPToolResult::error(format!("Failed to get stats: {}", e))),
    }
}

/// Enable embeddings configuration
pub async fn enable_embeddings(model_type: Option<String>) -> Result<MCPToolResult> {
    if !cfg!(feature = "embeddings") {
        return Ok(MCPToolResult::error(
            "Embeddings feature not compiled in. Please rebuild with --features embeddings"
                .to_string(),
        ));
    }

    // Note: This would require restarting the system to take effect
    // For now, we return information about embeddings support
    Ok(MCPToolResult::success(
        "Embeddings feature is available".to_string(),
        Some(serde_json::json!({
            "embeddings_compiled": cfg!(feature = "embeddings"),
            "available_models": ["StaticSimilarityMRL", "MiniLM", "TinyBERT", "BGESmall"],
            "default_model": model_type.unwrap_or_else(|| "StaticSimilarityMRL".to_string()),
            "note": "Embeddings must be enabled in system configuration and requires restart"
        })),
    ))
}

// Non-embeddings fallback versions for when embeddings feature is disabled
#[cfg(not(feature = "embeddings"))]
pub async fn semantic_search_global(
    _query: String,
    _limit: Option<usize>,
    _date_from: Option<String>,
    _date_to: Option<String>,
    _interaction_type: Option<Vec<String>>,
) -> Result<MCPToolResult> {
    Ok(MCPToolResult::error(
        "Semantic search requires the 'embeddings' feature to be enabled. Please rebuild with --features embeddings".to_string()
    ))
}

#[cfg(not(feature = "embeddings"))]
pub async fn semantic_search_session(
    _session_id: Uuid,
    _query: String,
    _limit: Option<usize>,
    _date_from: Option<String>,
    _date_to: Option<String>,
    _interaction_type: Option<Vec<String>>,
) -> Result<MCPToolResult> {
    Ok(MCPToolResult::error(
        "Semantic search requires the 'embeddings' feature to be enabled. Please rebuild with --features embeddings".to_string()
    ))
}

#[cfg(not(feature = "embeddings"))]
pub async fn find_related_content(
    _session_id: Uuid,
    _topic: String,
    _limit: Option<usize>,
) -> Result<MCPToolResult> {
    Ok(MCPToolResult::error(
        "Related content search requires the 'embeddings' feature to be enabled. Please rebuild with --features embeddings".to_string()
    ))
}

#[cfg(not(feature = "embeddings"))]
pub async fn vectorize_session(_session_id: Uuid) -> Result<MCPToolResult> {
    Ok(MCPToolResult::error(
        "Vectorization requires the 'embeddings' feature to be enabled. Please rebuild with --features embeddings".to_string()
    ))
}

#[cfg(not(feature = "embeddings"))]
pub async fn get_vectorization_stats() -> Result<MCPToolResult> {
    Ok(MCPToolResult::error(
        "Vectorization stats require the 'embeddings' feature to be enabled. Please rebuild with --features embeddings".to_string()
    ))
}
