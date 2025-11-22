// Copyright (c) 2025 Julius ML
// MIT License

//! Post-Cortex MCP Service with rmcp 0.9 using Parameters<T>

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
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

/// Post-Cortex MCP Service
#[derive(Clone)]
pub struct PostCortexService {
    memory_system: Arc<ConversationMemorySystem>,
    tool_router: ToolRouter<PostCortexService>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct LoadSessionRequest {
    #[schemars(description = "Session ID to load")]
    pub session_id: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct UpdateConversationContextRequest {
    #[schemars(
        description = "Type of interaction: qa, decision_made, problem_solved, code_change"
    )]
    pub interaction_type: String,
    #[schemars(description = "Content as key-value pairs")]
    pub content: std::collections::HashMap<String, String>,
    #[schemars(description = "Optional code reference")]
    pub code_reference: Option<serde_json::Value>,
    #[schemars(description = "Session ID")]
    pub session_id: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SemanticSearchSessionRequest {
    #[schemars(description = "Session ID to search")]
    pub session_id: String,
    #[schemars(description = "Search query")]
    pub query: String,
    #[schemars(description = "Maximum number of results (default: 10)")]
    pub limit: Option<usize>,
    #[schemars(description = "Filter from date (ISO 8601)")]
    pub date_from: Option<String>,
    #[schemars(description = "Filter to date (ISO 8601)")]
    pub date_to: Option<String>,
    #[schemars(description = "Filter by interaction types")]
    pub interaction_type: Option<Vec<String>>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SearchSessionsRequest {
    #[schemars(description = "Search query")]
    pub query: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetStructuredSummaryRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
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

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetKeyDecisionsRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct QueryConversationContextRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(
        description = "Query type: find_related_entities, get_entity_context, search_updates, etc."
    )]
    pub query_type: String,
    #[schemars(description = "Query parameters as key-value pairs")]
    pub parameters: std::collections::HashMap<String, String>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct UpdateSessionMetadataRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "New session name")]
    pub name: Option<String>,
    #[schemars(description = "New session description")]
    pub description: Option<String>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct BulkUpdateContextRequest {
    #[schemars(description = "Array of context updates")]
    pub updates: Vec<serde_json::Value>,
    #[schemars(description = "Session ID")]
    pub session_id: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetKeyInsightsRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Maximum number of insights to return")]
    pub limit: Option<usize>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetEntityImportanceRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Maximum entities to return")]
    pub limit: Option<usize>,
    #[schemars(description = "Minimum importance threshold")]
    pub min_importance: Option<f32>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetEntityNetworkRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Center entity for network view")]
    pub center_entity: Option<String>,
    #[schemars(description = "Maximum entities to include")]
    pub max_entities: Option<usize>,
    #[schemars(description = "Maximum relationships to show")]
    pub max_relationships: Option<usize>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SemanticSearchGlobalRequest {
    #[schemars(description = "Search query")]
    pub query: String,
    #[schemars(description = "Maximum results")]
    pub limit: Option<usize>,
    #[schemars(description = "Filter from date (ISO 8601)")]
    pub date_from: Option<String>,
    #[schemars(description = "Filter to date (ISO 8601)")]
    pub date_to: Option<String>,
    #[schemars(description = "Filter by interaction types")]
    pub interaction_type: Option<Vec<String>>,
}

#[derive(Deserialize, JsonSchema, Debug, Serialize)]
pub struct SearchScope {
    #[schemars(description = "Scope type: session, workspace, global")]
    #[serde(rename = "type")]
    pub type_: String,
    #[schemars(description = "ID for session/workspace scope")]
    pub id: Option<String>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SemanticSearchRequest {
    #[schemars(description = "Search query")]
    pub query: String,
    #[schemars(description = "Search scope")]
    pub scope: Option<SearchScope>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct FindRelatedContentRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Topic to find related content")]
    pub topic: String,
    #[schemars(description = "Maximum results")]
    pub limit: Option<usize>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct CreateWorkspaceRequest {
    #[schemars(description = "Workspace name")]
    pub name: String,
    #[schemars(description = "Workspace description")]
    pub description: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetWorkspaceRequest {
    #[schemars(description = "Workspace ID")]
    pub workspace_id: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct DeleteWorkspaceRequest {
    #[schemars(description = "Workspace ID")]
    pub workspace_id: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct AddSessionToWorkspaceRequest {
    #[schemars(description = "Workspace ID")]
    pub workspace_id: String,
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(description = "Session role (primary/related/dependency/shared)")]
    pub role: String,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct RemoveSessionFromWorkspaceRequest {
    #[schemars(description = "Workspace ID")]
    pub workspace_id: String,
    #[schemars(description = "Session ID")]
    pub session_id: String,
}

#[tool_router]
impl PostCortexService {
    pub fn new(memory_system: Arc<ConversationMemorySystem>) -> Self {
        info!("Initializing Post-Cortex MCP Service with rmcp 0.9");
        Self {
            memory_system,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Create a new conversation session")]
    async fn create_session(&self) -> Result<CallToolResult, McpError> {
        match self.memory_system.create_session(None, None).await {
            Ok(session_id) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Created session: {}",
                session_id
            ))])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "List all sessions")]
    async fn list_sessions(&self) -> Result<CallToolResult, McpError> {
        match self.memory_system.list_sessions().await {
            Ok(session_ids) => {
                let sessions_text = session_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Sessions ({}): \n{}",
                    session_ids.len(),
                    sessions_text
                ))]))
            }
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Load session into active memory")]
    async fn load_session(
        &self,
        params: Parameters<LoadSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        let uuid = Uuid::parse_str(&params.0.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match crate::tools::mcp::load_session(uuid).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(
        description = "Add context update to conversation (qa, decision_made, problem_solved, code_change)"
    )]
    async fn update_conversation_context(
        &self,
        params: Parameters<UpdateConversationContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        // Convert code_reference from Option<Value> to Option<CodeReference>
        let code_ref = req
            .code_reference
            .as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        match crate::tools::mcp::update_conversation_context(
            req.interaction_type.clone(),
            req.content.clone(),
            code_ref,
            uuid,
        )
        .await
        {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Semantic search within a session (auto-vectorizes if needed)")]
    async fn semantic_search_session(
        &self,
        params: Parameters<SemanticSearchSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
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

    #[tool(description = "Search sessions by name or description")]
    async fn search_sessions(
        &self,
        params: Parameters<SearchSessionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        match crate::tools::mcp::search_sessions(params.0.query.clone()).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Get comprehensive session summary with auto-compact for large sessions")]
    async fn get_structured_summary(
        &self,
        params: Parameters<GetStructuredSummaryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;

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
                let mut contents = vec![Content::text(result.message)];
                if let Some(data) = result.data {
                    contents.push(Content::text(format!(
                        "\n\nStructured Data:\n{}",
                        serde_json::to_string_pretty(&data).unwrap_or_default()
                    )));
                }
                Ok(CallToolResult::success(contents))
            }
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Get timeline of key decisions with confidence levels")]
    async fn get_key_decisions(
        &self,
        params: Parameters<GetKeyDecisionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        match crate::tools::mcp::get_key_decisions(params.0.session_id.clone()).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Search using entities, keywords, or structured queries")]
    async fn query_conversation_context(
        &self,
        params: Parameters<QueryConversationContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

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

    #[tool(description = "Update session name or description")]
    async fn update_session_metadata(
        &self,
        params: Parameters<UpdateSessionMetadataRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match crate::tools::mcp::update_session_metadata(
            uuid,
            req.name.clone(),
            req.description.clone(),
        )
        .await
        {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Batch update conversation context with multiple items")]
    async fn bulk_update_conversation_context(
        &self,
        params: Parameters<BulkUpdateContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        let updates: Vec<crate::tools::mcp::ContextUpdateItem> = req
            .updates
            .iter()
            .filter_map(|v| serde_json::from_value(v.clone()).ok())
            .collect();

        match crate::tools::mcp::bulk_update_conversation_context(updates, uuid).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Extract top insights from session data")]
    async fn get_key_insights(
        &self,
        params: Parameters<GetKeyInsightsRequest>,
    ) -> Result<CallToolResult, McpError> {
        match crate::tools::mcp::get_key_insights(params.0.session_id.clone(), params.0.limit).await
        {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Entity rankings with mention counts and relationships")]
    async fn get_entity_importance_analysis(
        &self,
        params: Parameters<GetEntityImportanceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        match crate::tools::mcp::get_entity_importance_analysis(
            req.session_id.clone(),
            req.limit,
            req.min_importance,
        )
        .await
        {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Visualize entity relationships as network graph")]
    async fn get_entity_network_view(
        &self,
        params: Parameters<GetEntityNetworkRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        match crate::tools::mcp::get_entity_network_view(
            req.session_id.clone(),
            req.center_entity.clone(),
            req.max_entities,
            req.max_relationships,
        )
        .await
        {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Search across ALL sessions with AI-powered semantic search")]
    async fn semantic_search_global(
        &self,
        params: Parameters<SemanticSearchGlobalRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
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

    #[tool(description = "Universal semantic search with scope (session, workspace, global)")]
    async fn semantic_search(
        &self,
        params: Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;

        // Convert scope struct to JSON for the tool function
        let scope_json = req.scope.as_ref().map(|s| serde_json::to_value(s).unwrap());

        match crate::tools::mcp::semantic_search(req.query.clone(), scope_json).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Find related content across different sessions")]
    async fn find_related_content(
        &self,
        params: Parameters<FindRelatedContentRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let uuid = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match crate::tools::mcp::find_related_content(uuid, req.topic.clone(), req.limit).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Create workspace to group related sessions")]
    async fn create_workspace(
        &self,
        params: Parameters<CreateWorkspaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        match crate::tools::mcp::create_workspace(req.name.clone(), req.description.clone()).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Retrieve workspace details including all sessions")]
    async fn get_workspace(
        &self,
        params: Parameters<GetWorkspaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let uuid = Uuid::parse_str(&params.0.workspace_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match crate::tools::mcp::get_workspace(uuid).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "List all workspaces with session counts")]
    async fn list_workspaces(&self) -> Result<CallToolResult, McpError> {
        match crate::tools::mcp::list_workspaces().await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Delete workspace (sessions remain intact)")]
    async fn delete_workspace(
        &self,
        params: Parameters<DeleteWorkspaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let uuid = Uuid::parse_str(&params.0.workspace_id)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))?;

        match crate::tools::mcp::delete_workspace(uuid).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Add session to workspace with role (primary/related/dependency/shared)")]
    async fn add_session_to_workspace(
        &self,
        params: Parameters<AddSessionToWorkspaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let workspace_id = Uuid::parse_str(&req.workspace_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid workspace ID: {}", e), None))?;
        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid session ID: {}", e), None))?;

        match crate::tools::mcp::add_session_to_workspace(
            workspace_id,
            session_id,
            req.role.clone(),
        )
        .await
        {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Remove session from workspace")]
    async fn remove_session_from_workspace(
        &self,
        params: Parameters<RemoveSessionFromWorkspaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let req = &params.0;
        let workspace_id = Uuid::parse_str(&req.workspace_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid workspace ID: {}", e), None))?;
        let session_id = Uuid::parse_str(&req.session_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid session ID: {}", e), None))?;

        match crate::tools::mcp::remove_session_from_workspace(workspace_id, session_id).await {
            Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
        }
    }

    #[tool(description = "Get daemon status")]
    async fn get_status(&self) -> Result<CallToolResult, McpError> {
        let msg = format!(
            "Post-Cortex daemon running. Version: {} | rmcp 0.9 with Parameters<T>",
            env!("CARGO_PKG_VERSION")
        );
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(description = "Get tool catalog")]
    async fn get_tool_catalog(&self) -> Result<CallToolResult, McpError> {
        match crate::tools::mcp::get_tool_catalog().await {
            Ok(result) => {
                let json = serde_json::to_string_pretty(&result).unwrap_or_default();
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Err(McpError::internal_error(e.to_string(), None)),
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
                "Post-Cortex: Intelligent conversation memory system with 24 RMCP tools (rmcp 0.9 with Parameters<T>). \
                 All tools use shared RocksDB for centralized knowledge management across multiple clients.".to_string()
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
