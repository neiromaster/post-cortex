// Copyright (c) 2025 Julius ML
// MIT License

//! Post-Cortex MCP Service - Consolidated 6-tool API
//!
//! Tools:
//! 1. session - create/list sessions
//! 2. update_conversation_context - single or bulk updates
//! 3. semantic_search - unified search with scope
//! 4. get_structured_summary - summary with optional sections
//! 5. query_conversation_context - flexible queries
//! 6. manage_workspace - workspace operations

use crate::ConversationMemorySystem;
use rmcp::{
    RoleServer, ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ErrorData as McpError, *},
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

/// Post-Cortex MCP Service
#[derive(Clone)]
pub struct PostCortexService {
    memory_system: Arc<ConversationMemorySystem>,
    tool_router: ToolRouter<PostCortexService>,
}

// =============================================================================
// Tool 1: session (create/list)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SessionRequest {
    #[schemars(description = "Action: create | list")]
    pub action: String,
    #[schemars(description = "Session name (for create action)")]
    pub name: Option<String>,
    #[schemars(description = "Session description (for create action)")]
    pub description: Option<String>,
}

// =============================================================================
// Tool 2: update_conversation_context (single + bulk)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug, Clone)]
pub struct ContextUpdateItem {
    #[schemars(description = "Type of interaction: qa, decision_made, problem_solved, code_change")]
    pub interaction_type: String,
    #[schemars(description = "Content as key-value pairs")]
    pub content: std::collections::HashMap<String, String>,
    #[schemars(description = "Optional code reference")]
    pub code_reference: Option<serde_json::Value>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct UpdateConversationContextRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Type of interaction: qa, decision_made, problem_solved, code_change (for single update)")]
    pub interaction_type: Option<String>,
    #[schemars(description = "Content as key-value pairs (for single update)")]
    pub content: Option<std::collections::HashMap<String, String>>,
    #[schemars(description = "Optional code reference (for single update)")]
    pub code_reference: Option<serde_json::Value>,
    #[schemars(description = "Array of updates for bulk operation (overrides single fields if provided)")]
    pub updates: Option<Vec<ContextUpdateItem>>,
}

// =============================================================================
// Tool 3: semantic_search (unified)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SemanticSearchRequest {
    #[schemars(description = "Search query")]
    pub query: String,
    #[schemars(description = "Scope: session | workspace | global (default: global)")]
    pub scope: Option<String>,
    #[schemars(description = "Session ID or Workspace ID (required when scope is session/workspace)")]
    pub scope_id: Option<String>,
    #[schemars(description = "Maximum number of results (default: 10)")]
    pub limit: Option<usize>,
    #[schemars(description = "Filter from date (ISO 8601)")]
    pub date_from: Option<String>,
    #[schemars(description = "Filter to date (ISO 8601)")]
    pub date_to: Option<String>,
    #[schemars(description = "Filter by interaction types")]
    pub interaction_type: Option<Vec<String>>,
}

// =============================================================================
// Tool 4: get_structured_summary (extended)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetStructuredSummaryRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Sections to include: decisions, insights, entities, all (default: all)")]
    pub include: Option<Vec<String>>,
    #[schemars(description = "Maximum decisions to include")]
    pub decisions_limit: Option<usize>,
    #[schemars(description = "Maximum entities to include")]
    pub entities_limit: Option<usize>,
    #[schemars(description = "Maximum questions to include")]
    pub questions_limit: Option<usize>,
    #[schemars(description = "Maximum concepts to include")]
    pub concepts_limit: Option<usize>,
    #[schemars(description = "Minimum confidence level")]
    pub min_confidence: Option<f32>,
    #[schemars(description = "Use compact format for large sessions")]
    pub compact: Option<bool>,
}

// =============================================================================
// Tool 5: query_conversation_context
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct QueryConversationContextRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Query type: find_related_entities, get_entity_context, search_updates, entity_importance, entity_network, etc.")]
    pub query_type: String,
    #[schemars(description = "Query parameters as key-value pairs")]
    pub parameters: std::collections::HashMap<String, String>,
}

// =============================================================================
// Tool 6: manage_workspace
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct ManageWorkspaceRequest {
    #[schemars(description = "Action: create | list | get | delete | add_session | remove_session")]
    pub action: String,
    #[schemars(description = "Workspace ID (for get/delete/add_session/remove_session)")]
    pub workspace_id: Option<String>,
    #[schemars(description = "Session ID (for add_session/remove_session)")]
    pub session_id: Option<String>,
    #[schemars(description = "Workspace name (for create)")]
    pub name: Option<String>,
    #[schemars(description = "Workspace description (for create)")]
    pub description: Option<String>,
    #[schemars(description = "Session role: primary | related | dependency | shared (for add_session)")]
    pub role: Option<String>,
}

// =============================================================================
// Tool Implementations
// =============================================================================

#[tool_router]
impl PostCortexService {
    pub fn new(memory_system: Arc<ConversationMemorySystem>) -> Self {
        info!("Initializing Post-Cortex MCP Service (6 consolidated tools)");
        Self {
            memory_system,
            tool_router: Self::tool_router(),
        }
    }

    // =========================================================================
    // Tool 1: session
    // =========================================================================
    #[tool(description = "Manage sessions: create new session or list all sessions")]
    async fn session(
        &self,
        params: Parameters<SessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;

        match req.action.to_lowercase().as_str() {
            "create" => {
                match self.memory_system.create_session(
                    req.name.clone(),
                    req.description.clone(),
                ).await {
                    Ok(session_id) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Created session: {}{}{}",
                        session_id,
                        req.name.as_ref().map(|n| format!(" (name: {})", n)).unwrap_or_default(),
                        req.description.as_ref().map(|d| format!(" - {}", d)).unwrap_or_default(),
                    ))])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "list" => {
                match crate::tools::mcp::list_sessions().await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            _ => Err(McpError::invalid_params(
                format!("Invalid action '{}'. Use: create | list", req.action),
                None,
            )),
        }
    }

    // =========================================================================
    // Tool 2: update_conversation_context (single + bulk)
    // =========================================================================
    #[tool(description = "Add context updates to conversation. Supports single update or bulk mode with updates array.")]
    async fn update_conversation_context(
        &self,
        params: Parameters<UpdateConversationContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        // Bulk mode: if updates array is provided
        if let Some(ref updates) = req.updates {
            let items: Vec<crate::tools::mcp::ContextUpdateItem> = updates
                .iter()
                .map(|u| crate::tools::mcp::ContextUpdateItem {
                    interaction_type: u.interaction_type.clone(),
                    content: u.content.clone(),
                    code_reference: u.code_reference.as_ref()
                        .and_then(|v| serde_json::from_value(v.clone()).ok()),
                })
                .collect();

            match crate::tools::mcp::bulk_update_conversation_context(items, uuid).await {
                Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                Err(e) => Err(McpError::internal_error(e.to_string(), None)),
            }
        } else {
            // Single update mode
            let interaction_type = req.interaction_type.as_ref()
                .ok_or_else(|| McpError::invalid_params(
                    "interaction_type is required for single update (or use updates array for bulk)".to_string(),
                    None,
                ))?;
            let content = req.content.as_ref()
                .ok_or_else(|| McpError::invalid_params(
                    "content is required for single update (or use updates array for bulk)".to_string(),
                    None,
                ))?;

            let code_ref = req
                .code_reference
                .as_ref()
                .and_then(|v| serde_json::from_value(v.clone()).ok());

            match crate::tools::mcp::update_conversation_context(
                interaction_type.clone(),
                content.clone(),
                code_ref,
                uuid,
            )
            .await
            {
                Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                Err(e) => Err(McpError::internal_error(e.to_string(), None)),
            }
        }
    }

    // =========================================================================
    // Tool 3: semantic_search (unified)
    // =========================================================================
    #[tool(description = "Universal semantic search. Scope: session (requires scope_id), workspace (requires scope_id), or global (default).")]
    async fn semantic_search(
        &self,
        params: Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let scope = req.scope.as_deref().unwrap_or("global");

        match scope.to_lowercase().as_str() {
            "session" => {
                let session_id = req.scope_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "scope_id (session_id) is required when scope is 'session'".to_string(),
                        None,
                    ))?;
                let uuid = Uuid::parse_str(session_id)
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

                match crate::tools::mcp::semantic_search_session(
                    uuid,
                    req.query.clone(),
                    req.limit,
                    req.date_from.clone(),
                    req.date_to.clone(),
                    req.interaction_type.clone(),
                )
                .await
                {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "workspace" => {
                // Build scope JSON for workspace search
                let scope_json = {
                    let mut scope = serde_json::Map::new();
                    scope.insert("scope_type".to_string(), serde_json::Value::String("workspace".to_string()));
                    if let Some(ref id) = req.scope_id {
                        scope.insert("id".to_string(), serde_json::Value::String(id.clone()));
                    }
                    Some(serde_json::Value::Object(scope))
                };

                match crate::tools::mcp::semantic_search(req.query.clone(), scope_json).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "global" | _ => {
                match crate::tools::mcp::semantic_search_global(
                    req.query.clone(),
                    req.limit,
                    req.date_from.clone(),
                    req.date_to.clone(),
                    req.interaction_type.clone(),
                )
                .await
                {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
        }
    }

    // =========================================================================
    // Tool 4: get_structured_summary (extended)
    // =========================================================================
    #[tool(description = "Get session summary. Use 'include' to specify sections: decisions, insights, entities, or all (default).")]
    async fn get_structured_summary(
        &self,
        params: Parameters<GetStructuredSummaryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        // Validate session_id format
        let _uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        // Determine which sections to include
        let include = req.include.as_ref().map(|v| {
            v.iter().map(|s| s.to_lowercase()).collect::<Vec<_>>()
        });

        let include_all = include.is_none() ||
            include.as_ref().is_some_and(|v| v.contains(&"all".to_string()));
        let include_decisions = include_all ||
            include.as_ref().is_some_and(|v| v.contains(&"decisions".to_string()));
        let include_insights = include_all ||
            include.as_ref().is_some_and(|v| v.contains(&"insights".to_string()));
        let include_entities = include_all ||
            include.as_ref().is_some_and(|v| v.contains(&"entities".to_string()));

        let mut result_parts = Vec::new();

        // Get full summary if all or no specific sections
        if include_all {
            match crate::tools::mcp::get_structured_summary(
                req.session_id.clone(),
                req.decisions_limit,
                req.entities_limit,
                req.questions_limit,
                req.concepts_limit,
                req.min_confidence,
                req.compact,
            )
            .await
            {
                Ok(result) => {
                    result_parts.push(result.message);
                    if let Some(data) = result.data {
                        result_parts.push(format!(
                            "\n\nStructured Data:\n{}",
                            serde_json::to_string_pretty(&data).unwrap_or_default()
                        ));
                    }
                }
                Err(e) => return Err(McpError::internal_error(e.to_string(), None)),
            }
        } else {
            // Get specific sections only
            if include_decisions {
                match crate::tools::mcp::get_key_decisions(req.session_id.clone()).await {
                    Ok(result) => result_parts.push(format!("## Decisions\n{}", result.message)),
                    Err(e) => result_parts.push(format!("## Decisions\nError: {}", e)),
                }
            }

            if include_insights {
                match crate::tools::mcp::get_key_insights(
                    req.session_id.clone(),
                    req.decisions_limit, // reuse limit
                ).await {
                    Ok(result) => result_parts.push(format!("## Insights\n{}", result.message)),
                    Err(e) => result_parts.push(format!("## Insights\nError: {}", e)),
                }
            }

            if include_entities {
                match crate::tools::mcp::get_entity_importance_analysis(
                    req.session_id.clone(),
                    req.entities_limit,
                    req.min_confidence,
                ).await {
                    Ok(result) => result_parts.push(format!("## Entities\n{}", result.message)),
                    Err(e) => result_parts.push(format!("## Entities\nError: {}", e)),
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(result_parts.join("\n\n"))]))
    }

    // =========================================================================
    // Tool 5: query_conversation_context
    // =========================================================================
    #[tool(description = "Query session data. Types: find_related_entities, get_entity_context, search_updates, entity_importance, entity_network, etc.")]
    async fn query_conversation_context(
        &self,
        params: Parameters<QueryConversationContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        // Handle special query types that map to deprecated tools
        match req.query_type.to_lowercase().as_str() {
            "entity_importance" => {
                let limit = req.parameters.get("limit")
                    .and_then(|s| s.parse().ok());
                let min_importance = req.parameters.get("min_importance")
                    .and_then(|s| s.parse().ok());

                match crate::tools::mcp::get_entity_importance_analysis(
                    req.session_id.clone(),
                    limit,
                    min_importance,
                ).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "entity_network" => {
                let center_entity = req.parameters.get("center_entity").cloned();
                let max_entities = req.parameters.get("max_entities")
                    .and_then(|s| s.parse().ok());
                let max_relationships = req.parameters.get("max_relationships")
                    .and_then(|s| s.parse().ok());

                match crate::tools::mcp::get_entity_network_view(
                    req.session_id.clone(),
                    center_entity,
                    max_entities,
                    max_relationships,
                ).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            _ => {
                // Default: use the original query_conversation_context
                match crate::tools::mcp::query_conversation_context(
                    req.query_type.clone(),
                    req.parameters.clone(),
                    uuid,
                )
                .await
                {
                    Ok(result) => {
                        let mut contents = vec![Content::text(result.message)];
                        if let Some(data) = result.data {
                            contents.push(Content::text(format!(
                                "\n\nResults:\n{}",
                                serde_json::to_string_pretty(&data).unwrap_or_default()
                            )));
                        }
                        Ok(CallToolResult::success(contents))
                    }
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
        }
    }

    // =========================================================================
    // Tool 6: manage_workspace
    // =========================================================================
    #[tool(description = "Manage workspaces. Actions: create, list, get, delete, add_session, remove_session")]
    async fn manage_workspace(
        &self,
        params: Parameters<ManageWorkspaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;

        match req.action.to_lowercase().as_str() {
            "create" => {
                let name = req.name.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "name is required for create action".to_string(),
                        None,
                    ))?;
                let description = req.description.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "description is required for create action".to_string(),
                        None,
                    ))?;

                match crate::tools::mcp::create_workspace(name.clone(), description.clone()).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "list" => {
                match crate::tools::mcp::list_workspaces().await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "get" => {
                let workspace_id = req.workspace_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "workspace_id is required for get action".to_string(),
                        None,
                    ))?;
                let uuid = Uuid::parse_str(workspace_id)
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

                match crate::tools::mcp::get_workspace(uuid).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "delete" => {
                let workspace_id = req.workspace_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "workspace_id is required for delete action".to_string(),
                        None,
                    ))?;
                let uuid = Uuid::parse_str(workspace_id)
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

                match crate::tools::mcp::delete_workspace(uuid).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "add_session" => {
                let workspace_id = req.workspace_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "workspace_id is required for add_session action".to_string(),
                        None,
                    ))?;
                let session_id = req.session_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "session_id is required for add_session action".to_string(),
                        None,
                    ))?;
                let role = req.role.clone().unwrap_or_else(|| "related".to_string());

                let ws_uuid = Uuid::parse_str(workspace_id)
                    .map_err(|e| McpError::invalid_params(format!("Invalid workspace_id: {}", e), None))?;
                let sess_uuid = Uuid::parse_str(session_id)
                    .map_err(|e| McpError::invalid_params(format!("Invalid session_id: {}", e), None))?;

                match crate::tools::mcp::add_session_to_workspace(ws_uuid, sess_uuid, role).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "remove_session" => {
                let workspace_id = req.workspace_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "workspace_id is required for remove_session action".to_string(),
                        None,
                    ))?;
                let session_id = req.session_id.as_ref()
                    .ok_or_else(|| McpError::invalid_params(
                        "session_id is required for remove_session action".to_string(),
                        None,
                    ))?;

                let ws_uuid = Uuid::parse_str(workspace_id)
                    .map_err(|e| McpError::invalid_params(format!("Invalid workspace_id: {}", e), None))?;
                let sess_uuid = Uuid::parse_str(session_id)
                    .map_err(|e| McpError::invalid_params(format!("Invalid session_id: {}", e), None))?;

                match crate::tools::mcp::remove_session_from_workspace(ws_uuid, sess_uuid).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            _ => Err(McpError::invalid_params(
                format!("Invalid action '{}'. Use: create | list | get | delete | add_session | remove_session", req.action),
                None,
            )),
        }
    }
}

#[tool_handler]
impl ServerHandler for PostCortexService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Post-Cortex: Intelligent conversation memory system with 6 consolidated MCP tools. \
                 Tools: session, update_conversation_context, semantic_search, get_structured_summary, \
                 query_conversation_context, manage_workspace. All use shared RocksDB for centralized \
                 knowledge management.".to_string()
            ),
        }
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        if let Some(parts) = context.extensions.get::<axum::http::request::Parts>() {
            info!("Client initialized from {}", parts.uri);
        }
        Ok(self.get_info())
    }
}
