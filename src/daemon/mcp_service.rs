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
use crate::daemon::coerce::{CoercionError, coerce_and_validate};
use crate::daemon::validate::*;
use crate::tools::mcp::{get_memory_system, MCPToolResult};
use uuid::Uuid;
use rmcp::{
    RoleServer, ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{
        CallToolRequestParam, CallToolResult, Content, ErrorData as McpError, ListToolsResult,
        PaginatedRequestParam, *,
    },
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;
use tracing::info;

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
    #[schemars(description = r#"Action to perform: 'create' or 'list'.

Examples:
✅ "create" - Creates a new session and returns a UUID
✅ "list" - Lists all existing sessions

Note: Must be lowercase. Use 'create' to get a session_id before using other tools."#)]
    pub action: String,
    #[schemars(description = r#"Optional session name (for create action).

Examples:
- "Feature: Authentication"
- "Bug Fix: Memory Leak"
- "Planning: API Design"

Note: Only used when action='create'. Helps identify sessions later."#)]
    pub name: Option<String>,
    #[schemars(description = r#"Optional session description (for create action).

Example: "Working on implementing OAuth2 login flow"

Note: Only used when action='create'. Provides context for the session."#)]
    pub description: Option<String>,
}

// =============================================================================
// Tool 2: update_conversation_context (single + bulk)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug, Clone)]
pub struct ContextUpdateItem {
    #[schemars(description = r#"Type of interaction (must be exact lowercase value).

Valid values:
- qa: Questions and answers about the codebase
- decision_made: Architectural decisions and trade-offs
- problem_solved: Bug fixes and solutions to technical problems
- code_change: Code modifications and refactoring
- requirement_added: New requirements or constraints
- concept_defined: Technical concepts and patterns explained

Examples:
✅ "decision_made" (correct)
❌ "DecisionMade" (wrong - must be lowercase)
❌ "made_decision" (wrong - use exact term)"#)]
    pub interaction_type: String,
    #[schemars(
        description = r#"Content as key-value pairs (all values must be strings).

Format: HashMap<String, String> where both keys and values are strings.

Examples:
Simple: {"decision": "Use Rust", "rationale": "Performance"}
Complex: {"criteria": "{\"performance\": 9, \"safety\": 10}", "date": "2025-01-12"}

Note: For complex nested data, stringify as JSON first. Do not pass nested objects directly."#
    )]
    pub content: std::collections::HashMap<String, String>,
    #[schemars(description = r#"Optional code reference for context.

Can be a simple string or complex object:

Examples:
- Simple: "src/main.rs:42"
- Complex: {"file": "src/main.rs", "line": 42, "function": "process_data"}

Note: Helps link context to specific code locations."#)]
    pub code_reference: Option<serde_json::Value>,
}

#[derive(Deserialize, JsonSchema, Debug)]
pub struct UpdateConversationContextRequest {
    #[schemars(description = r#"Session ID (36-char UUID format with hyphens).

Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

How to get:
1. Create session: Use session tool with action='create'
2. Find session: Use semantic_search to search for previous sessions

Example: "60c598e2-d602-4e07-a328-c458006d48c7"

Note: Must be a string, not a number. UUIDs are always strings."#)]
    pub session_id: String,
    #[schemars(
        description = r#"Type of interaction for single update (must be exact lowercase value).

Valid values:
- qa: Questions and answers about the codebase
- decision_made: Architectural decisions and trade-offs
- problem_solved: Bug fixes and solutions to technical problems
- code_change: Code modifications and refactoring
- requirement_added: New requirements or constraints
- concept_defined: Technical concepts and patterns explained

Note: Required for single update mode (when 'updates' array is not provided)."#
    )]
    pub interaction_type: Option<String>,
    #[schemars(
        description = r#"Content as key-value pairs for single update (all values must be strings).

Format: HashMap<String, String> where both keys and values are strings.

Examples:
Simple: {"decision": "Use Rust", "rationale": "Performance"}
Complex: {"criteria": "{\"performance\": 9}", "date": "2025-01-12"}

Note: For complex nested data, stringify as JSON first. Required for single update mode."#
    )]
    pub content: Option<std::collections::HashMap<String, String>>,
    #[schemars(description = r#"Optional code reference for single update.

Can be a simple string or complex object:

Examples:
- Simple: "src/main.rs:42"
- Complex: {"file": "src/main.rs", "line": 42, "function": "process_data"}"#)]
    pub code_reference: Option<serde_json::Value>,
    #[schemars(
        description = r#"Array of updates for bulk operation (overrides single fields if provided).

Use this mode to add multiple updates at once.

Format: Array of objects, each with interaction_type and content.

Example:
{
  "session_id": "60c598e2-d602-4e07-a328-c458006d48c7",
  "updates": [
    {
      "interaction_type": "decision_made",
      "content": {"decision": "Use Rust", "rationale": "Performance"}
    },
    {
      "interaction_type": "code_change",
      "content": {"file": "main.rs", "change": "Add error handling"}
    }
  ]
}

Note: When provided, 'interaction_type' and 'content' fields are ignored."#
    )]
    pub updates: Option<Vec<ContextUpdateItem>>,
}

// =============================================================================
// Tool 3: semantic_search (unified)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct SemanticSearchRequest {
    #[schemars(description = r#"Search query for semantic search.

Examples:
- "How did we handle authentication?"
- "What were the performance issues?"
- "decision_made API design"

Note: Natural language queries work best. The search understands context."#)]
    pub query: String,
    #[schemars(
        description = r#"Search scope: 'session', 'workspace', or 'global' (default: 'global').

Valid values:
- session: Search within a specific session (requires scope_id)
- workspace: Search within a workspace (requires scope_id)
- global: Search across all data (default, no scope_id needed)

Examples:
✅ {"query": "performance", "scope": "global"} (default, searches everywhere)
✅ {"query": "auth", "scope": "session", "scope_id": "60c598e2-d602-4e07-a328-c458006d48c7"} (search specific session)
✅ {"query": "API", "scope": "workspace", "scope_id": "f1d2e3a4-b5c6-7d8e-9f0a-1b2c3d4e5f6f"} (search specific workspace)

Note: scope must be lowercase. When using 'session' or 'workspace', you must provide scope_id."#
    )]
    pub scope: Option<String>,
    #[schemars(
        description = r#"Session ID or Workspace ID (required when scope is 'session' or 'workspace').

Format: 36-char UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)

When to use:
- Required when scope='session' (provide session UUID)
- Required when scope='workspace' (provide workspace UUID)
- Ignored when scope='global' or not specified

Examples:
✅ {"scope": "session", "scope_id": "60c598e2-d602-4e07-a328-c458006d48c7"} (correct)
❌ {"scope": "session"} (missing scope_id - will error)
❌ {"scope": "global", "scope_id": "..."} (scope_id ignored for global)

Note: Must be a valid UUID string. Use 'session' or 'semantic_search' tools to find UUIDs."#
    )]
    pub scope_id: Option<String>,
    #[schemars(
        description = r#"Maximum number of results to return (default: 10, max: 100).

Examples:
- 5: Return top 5 most relevant results
- 20: Return top 20 results
- null or omit: Use default of 10

Note: Higher values may slow down search. Maximum allowed is 100."#
    )]
    pub limit: Option<usize>,
    #[schemars(
        description = r#"Filter results from this date onwards (ISO 8601 format).

Format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ

Examples:
- "2025-01-01": Results from January 1st, 2025 onwards
- "2025-01-15T10:00:00Z": Results from January 15th, 2025 at 10am UTC onwards

Note: Use with date_to to specify a date range."#
    )]
    pub date_from: Option<String>,
    #[schemars(description = r#"Filter results up to this date (ISO 8601 format).

Format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ

Examples:
- "2025-01-31": Results up to January 31st, 2025
- "2025-01-15T10:00:00Z": Results up to January 15th, 2025 at 10am UTC

Note: Use with date_from to specify a date range."#)]
    pub date_to: Option<String>,
    #[schemars(description = r#"Filter by interaction types (array of strings).

Valid types:
- qa: Questions and answers
- decision_made: Architectural decisions
- problem_solved: Bug fixes and solutions
- code_change: Code modifications
- requirement_added: New requirements
- concept_defined: Technical concepts

Examples:
- ["decision_made"]: Only show decisions
- ["decision_made", "problem_solved"]: Show decisions and solutions
- null or omit: Show all interaction types

Note: Must be exact lowercase values. Filters reduce results to only these types."#)]
    pub interaction_type: Option<Vec<String>>,
    #[schemars(
        description = r#"Temporal decay factor to prioritize recent content (default: 0.0 = disabled).

Valid values:
- 0.0: Disabled - pure relevance ranking (default, backward compatible)
- 0.1 - 0.5: Soft bias toward recent content
- 0.5 - 1.0: Moderate bias toward recent content
- 1.0+: Aggressive bias toward recent content

How it works:
- Uses exponential decay: score × e^(-λ × days/365)
- Older content gets progressively lower scores
- Fresh content (1 day) retains ~99.9% score at λ=0.5
- Year-old content retains ~60.6% score at λ=0.5, ~36.8% at λ=1.0

When to use:
- Debugging recent issues: 0.5 - 1.0
- Finding latest solutions: 0.3 - 0.7
- Architecture docs (timeless): 0.0 or omit
- Current context: 1.0+

Examples:
✅ {"query": "timeout error", "recency_bias": 0.5} (recent bugs prioritized)
✅ {"query": "authentication", "recency_bias": 0.0} (all docs equal)
✅ {"query": "performance", "recency_bias": 1.0} (very recent)

Note: Only affects ranking order, doesn't filter out old results."#
    )]
    pub recency_bias: Option<f32>,
}

// =============================================================================
// Tool 4: get_structured_summary (extended)
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct GetStructuredSummaryRequest {
    #[schemars(description = "Session ID")]
    pub session_id: String,
    #[schemars(
        description = "Sections to include: decisions, insights, entities, all (default: all)"
    )]
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
    #[schemars(
        description = "Query type: find_related_entities, get_entity_context, search_updates, entity_importance, entity_network, etc."
    )]
    pub query_type: String,
    #[schemars(description = "Query parameters as key-value pairs")]
    pub parameters: std::collections::HashMap<String, String>,
}

// =============================================================================
// Tool 6: manage_workspace
// =============================================================================

#[derive(Deserialize, JsonSchema, Debug)]
pub struct ManageWorkspaceRequest {
    #[schemars(description = r#"Action to perform on workspaces.

Valid values:
- create: Create a new workspace (requires name and description)
- list: List all workspaces
- get: Get workspace details (requires workspace_id)
- delete: Delete a workspace (requires workspace_id)
- add_session: Add a session to workspace (requires workspace_id, session_id, optional role)
- remove_session: Remove a session from workspace (requires workspace_id, session_id)

Examples:
✅ {"action": "create", "name": "Auth Feature", "description": "OAuth2 work"}
✅ {"action": "get", "workspace_id": "f1d2e3a4-..."}
✅ {"action": "add_session", "workspace_id": "...", "session_id": "...", "role": "primary"}

Note: Must be lowercase. Different actions require different parameters."#)]
    pub action: String,
    #[schemars(
        description = r#"Workspace ID (36-char UUID) for get/delete/add_session/remove_session actions.

Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

When to use:
- Required for: get, delete, add_session, remove_session
- Not used for: create, list

How to get:
- Use manage_workspace with action='list' to see all workspaces
- Use semantic_search to find workspaces by name

Example: "f1d2e3a4-b5c6-7d8e-9f0a-1b2c3d4e5f6f"

Note: Must be a valid UUID string. Not a number."#
    )]
    pub workspace_id: Option<String>,
    #[schemars(
        description = r#"Session ID (36-char UUID) for add_session/remove_session actions.

Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

When to use:
- Required for: add_session, remove_session
- Not used for: create, list, get, delete

How to get:
- Use session tool with action='list' to see all sessions
- Use semantic_search to find sessions

Example: "60c598e2-d602-4e07-a328-c458006d48c7"

Note: Must be a valid UUID string. Not a number."#
    )]
    pub session_id: Option<String>,
    #[schemars(description = r#"Workspace name (for create action).

Example: "Authentication Feature" or "API Design"

Note: Only used when action='create'. Helps identify the workspace."#)]
    pub name: Option<String>,
    #[schemars(description = r#"Workspace description (for create action).

Example: "Working on OAuth2 authentication and user management"

Note: Only used when action='create'. Provides context for the workspace."#)]
    pub description: Option<String>,
    #[schemars(
        description = r#"Session role in workspace (for add_session action, default: 'related').

Valid values:
- primary: Main session for this workspace
- related: Related context session
- dependency: Required dependency session
- shared: Shared reference session

Examples:
✅ {"action": "add_session", "workspace_id": "...", "session_id": "...", "role": "primary"}
✅ {"action": "add_session", "workspace_id": "...", "session_id": "...", "role": "related"}
✅ {"action": "add_session", "workspace_id": "...", "session_id": "..."} (defaults to "related")

Note: Only used when action='add_session'. Defaults to 'related' if omitted."#
    )]
    pub role: Option<String>,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse date range from request parameters
///
/// Returns Ok(Some((from, to))) if both dates are provided and valid
/// Returns Ok(None) if neither date is provided
/// Returns Err if only one date is provided or if parsing fails
fn parse_date_range(
    date_from: Option<String>,
    date_to: Option<String>,
) -> Result<Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>, McpError> {
    match (date_from, date_to) {
        (Some(from_str), Some(to_str)) => {
            let from = chrono::DateTime::parse_from_rfc3339(&from_str)
                .map_err(|_| McpError::invalid_params(
                    format!("Invalid date_from format: '{}'. Expected RFC3339 format (e.g., 2024-01-01T00:00:00Z)", from_str),
                    Some(serde_json::Value::String("date_from".to_string())),
                ))?
                .with_timezone(&chrono::Utc);
            let to = chrono::DateTime::parse_from_rfc3339(&to_str)
                .map_err(|_| McpError::invalid_params(
                    format!("Invalid date_to format: '{}'. Expected RFC3339 format (e.g., 2024-12-31T23:59:59Z)", to_str),
                    Some(serde_json::Value::String("date_to".to_string())),
                ))?
                .with_timezone(&chrono::Utc);
            Ok(Some((from, to)))
        }
        (Some(_), None) | (None, Some(_)) => {
            Err(McpError::invalid_params(
                "Both date_from and date_to must be provided together".to_string(),
                Some(serde_json::Value::String("date_from,date_to".to_string())),
            ))
        }
        (None, None) => Ok(None),
    }
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
    #[tool(
        description = "Manage sessions: create new session or list all sessions",
        input_schema = rmcp::handler::server::common::schema_for_type::<SessionRequest>()
    )]
    async fn session(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        let req: SessionRequest = coerce_and_validate(params.0).map_err(|e| {
            e.with_parameter_path("action".to_string())
                .with_expected_type("one of: create, list")
                .with_hint("Use 'create' to create a new session or 'list' to see all sessions")
                .to_mcp_error()
        })?;

        validate_session_action(&req.action).map_err(|e| e.to_mcp_error())?;

        match req.action.to_lowercase().as_str() {
            "create" => {
                match self
                    .memory_system
                    .create_session(req.name.clone(), req.description.clone())
                    .await
                {
                    Ok(session_id) => Ok(CallToolResult::success(vec![Content::text(format!(
                        "Created session: {}{}{}",
                        session_id,
                        req.name
                            .as_ref()
                            .map(|n| format!(" (name: {})", n))
                            .unwrap_or_default(),
                        req.description
                            .as_ref()
                            .map(|d| format!(" - {}", d))
                            .unwrap_or_default(),
                    ))])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "list" => match crate::tools::mcp::list_sessions().await {
                Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                Err(e) => Err(McpError::internal_error(e.to_string(), None)),
            },
            _ => Err(McpError::invalid_params(
                format!("Invalid action '{}'. Use: create | list", req.action),
                None,
            )),
        }
    }

    // =========================================================================
    // Tool 2: update_conversation_context (single + bulk)
    // =========================================================================
    #[tool(
        description = "Add context updates to conversation. Supports single update or bulk mode with updates array.",
        input_schema = rmcp::handler::server::common::schema_for_type::<UpdateConversationContextRequest>()
    )]
    async fn update_conversation_context(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        let req: UpdateConversationContextRequest = coerce_and_validate(params.0)
            .map_err(|e| {
                // Enhance error with parameter-specific hints
                if e.message.contains("session_id") {
                    e.clone()
                        .with_parameter_path("session_id".to_string())
                        .with_expected_type("UUID string (36 chars with hyphens)")
                        .with_hint("Create a session first using the 'session' tool with action='create', then use the returned UUID")
                        .to_mcp_error()
                } else if e.message.contains("interaction_type") {
                    e.clone()
                        .with_parameter_path("interaction_type".to_string())
                        .with_expected_type("one of: qa, decision_made, problem_solved, code_change, requirement_added, concept_defined")
                        .with_hint("Valid interaction types: qa, decision_made, problem_solved, code_change, requirement_added, concept_defined")
                        .to_mcp_error()
                } else if e.message.contains("content") {
                    e.clone()
                        .with_parameter_path("content".to_string())
                        .with_expected_type("object with string key-value pairs")
                        .with_hint("Content must be a map of string keys to string values. For complex data, stringify as JSON first.")
                        .to_mcp_error()
                } else {
                    e.to_mcp_error()
                }
            })?;

        let uuid = validate_session_id(&req.session_id).map_err(|e| e.to_mcp_error())?;

        // Bulk mode: if updates array is provided
        if let Some(ref updates) = req.updates {
            let items: Vec<crate::tools::mcp::ContextUpdateItem> = updates
                .iter()
                .map(|u| crate::tools::mcp::ContextUpdateItem {
                    interaction_type: u.interaction_type.clone(),
                    content: u.content.clone(),
                    code_reference: u
                        .code_reference
                        .as_ref()
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
                .ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "interaction_type required"),
                        None,
                    )
                    .with_parameter_path("interaction_type".to_string())
                    .with_expected_type("one of: qa, decision_made, problem_solved, code_change, requirement_added, concept_defined")
                    .with_hint("For single update, provide 'interaction_type' and 'content'. For bulk updates, use 'updates' array instead.")
                    .to_mcp_error()
                })?;

            validate_interaction_type(interaction_type).map_err(|e| e.to_mcp_error())?;
            let content = req.content.as_ref()
                .ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "content required"),
                        None,
                    )
                    .with_parameter_path("content".to_string())
                    .with_expected_type("object with string key-value pairs")
                    .with_hint("For single update, provide 'content' and 'interaction_type'. For bulk updates, use 'updates' array instead.")
                    .to_mcp_error()
                })?;

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
    #[tool(
        description = "Universal semantic search. Scope: session (requires scope_id), workspace (requires scope_id), or global (default).",
        input_schema = rmcp::handler::server::common::schema_for_type::<SemanticSearchRequest>()
    )]
    async fn semantic_search(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        let req: SemanticSearchRequest = coerce_and_validate(params.0)
            .map_err(|e| {
                if e.message.contains("scope") {
                    e.clone()
                        .with_parameter_path("scope".to_string())
                        .with_expected_type("one of: session, workspace, global")
                        .with_hint("Valid scopes: 'session' (requires scope_id), 'workspace' (requires scope_id), 'global' (default)")
                        .to_mcp_error()
                } else if e.message.contains("scope_id") {
                    e.clone()
                        .with_parameter_path("scope_id".to_string())
                        .with_expected_type("UUID string (required when scope is 'session' or 'workspace')")
                        .with_hint("When scope is 'session' or 'workspace', provide the session/workspace UUID as scope_id")
                        .to_mcp_error()
                } else if e.message.contains("query") {
                    e.clone()
                        .with_parameter_path("query".to_string())
                        .with_expected_type("search query string")
                        .with_hint("Provide a text query to search for in the conversation history")
                        .to_mcp_error()
                } else {
                    e.to_mcp_error()
                }
            })?;

        let scope = req.scope.as_deref().unwrap_or("global");

        validate_scope(scope).map_err(|e| e.to_mcp_error())?;

        // Validate limit parameter (default: 10, max: 1000)
        let validated_limit = validate_limits(req.limit, 10, 1000)
            .map_err(|e| e.with_parameter_path("limit".to_string()).to_mcp_error())?;

        match scope.to_lowercase().as_str() {
            "session" => {
                let session_id = req.scope_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "scope_id required"),
                        None,
                    )
                    .with_parameter_path("scope_id".to_string())
                    .with_expected_type("UUID string")
                    .with_hint(
                        "When scope is 'session', you must provide scope_id (the session UUID)",
                    )
                    .to_mcp_error()
                })?;

                let uuid = validate_session_id(session_id).map_err(|e| e.to_mcp_error())?;

                // Parse date range if provided
                let date_range = parse_date_range(req.date_from.clone(), req.date_to.clone())?;

                // Validate recency_bias if provided
                let validated_recency_bias = validate_recency_bias(req.recency_bias)
                    .map_err(|e| e.to_mcp_error())?;

                // Use recency_bias if provided
                let search_results = if let Some(bias) = validated_recency_bias {
                    // Get engine and use _with_recency method
                    let system = get_memory_system().await
                        .map_err(|e| McpError::internal_error(format!("Failed to get memory system: {}", e), None))?;

                    let engine = system
                        .semantic_query_engine
                        .get()
                        .ok_or_else(|| McpError::internal_error("Semantic engine not initialized".to_string(), None))?;

                    let results = engine
                        .semantic_search_session_with_recency_bias(
                            uuid,
                            &req.query,
                            Some(validated_limit),
                            date_range,
                            bias,
                        )
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

                    // Format results consistently with JSON
                    let formatted: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "content": r.text_content,
                                "score": r.combined_score,
                                "session_id": r.session_id,
                                "type": format!("{:?}", r.content_type),
                                "timestamp": r.timestamp.to_rfc3339()
                            })
                        })
                        .collect();

                    MCPToolResult::success(
                        format!("Found {} results", results.len()),
                        Some(serde_json::json!({ "results": formatted })),
                    )
                } else {
                    crate::tools::mcp::semantic_search_session(
                        uuid,
                        req.query.clone(),
                        Some(validated_limit),
                        req.date_from.clone(),
                        req.date_to.clone(),
                        req.interaction_type.clone(),
                        None,
                    )
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?
                };

                Ok(CallToolResult::success(vec![Content::text(search_results.message)]))
            }
            "workspace" => {
                // For workspace search, we need to get workspace sessions and search with recency bias
                let ws_id = req.scope_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "scope_id required"),
                        None,
                    )
                    .with_parameter_path("scope_id".to_string())
                    .with_expected_type("UUID string")
                    .with_hint("When scope is 'workspace', you must provide scope_id (the workspace UUID)")
                    .to_mcp_error()
                })?;

                let ws_uuid = Uuid::parse_str(ws_id).map_err(|_| {
                    CoercionError::new(
                        "Invalid UUID",
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid UUID format"),
                        None,
                    )
                    .with_parameter_path("scope_id".to_string())
                    .with_expected_type("Valid UUID string")
                    .to_mcp_error()
                })?;

                // Get system and workspace
                let system = get_memory_system().await
                    .map_err(|e| McpError::internal_error(format!("Failed to get memory system: {}", e), None))?;

                let workspace = system
                    .workspace_manager
                    .get_workspace(&ws_uuid)
                    .ok_or_else(|| McpError::internal_error(format!("Workspace {} not found", ws_uuid), None))?;

                let session_ids: Vec<Uuid> = workspace
                    .get_all_sessions()
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();

                // Parse date range if provided
                let date_range = parse_date_range(req.date_from.clone(), req.date_to.clone())?;

                // Validate recency_bias if provided
                let validated_recency_bias = validate_recency_bias(req.recency_bias)
                    .map_err(|e| e.to_mcp_error())?;

                // Use recency_bias if provided
                let search_results = if let Some(bias) = validated_recency_bias {
                    let engine = system
                        .semantic_query_engine
                        .get()
                        .ok_or_else(|| McpError::internal_error("Semantic engine not initialized".to_string(), None))?;

                    let results = engine
                        .semantic_search_multisession_with_recency_bias(
                            &session_ids,
                            &req.query,
                            Some(validated_limit),
                            date_range,
                            bias,
                        )
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

                    // Format results consistently
                    let formatted: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "content": r.text_content,
                                "score": r.combined_score,
                                "session_id": r.session_id,
                                "type": format!("{:?}", r.content_type),
                                "timestamp": r.timestamp.to_rfc3339()
                            })
                        })
                        .collect();

                    MCPToolResult::success(
                        format!("Found {} results", results.len()),
                        Some(serde_json::json!({ "results": formatted })),
                    )
                } else {
                    // Use existing universal semantic_search for backward compatibility
                    let scope_json = {
                        let mut scope = serde_json::Map::new();
                        scope.insert(
                            "scope_type".to_string(),
                            serde_json::Value::String("workspace".to_string()),
                        );
                        scope.insert("id".to_string(), serde_json::Value::String(ws_id.clone()));
                        Some(serde_json::Value::Object(scope))
                    };

                    crate::tools::mcp::semantic_search(req.query.clone(), scope_json)
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?
                };

                Ok(CallToolResult::success(vec![Content::text(search_results.message)]))
            }
            "global" | _ => {
                // Parse date range if provided
                let date_range = parse_date_range(req.date_from.clone(), req.date_to.clone())?;

                // Validate recency_bias if provided
                let validated_recency_bias = validate_recency_bias(req.recency_bias)
                    .map_err(|e| e.to_mcp_error())?;

                // Use recency_bias if provided
                let search_results = if let Some(bias) = validated_recency_bias {
                    let system = get_memory_system().await
                        .map_err(|e| McpError::internal_error(format!("Failed to get memory system: {}", e), None))?;

                    let engine = system
                        .semantic_query_engine
                        .get()
                        .ok_or_else(|| McpError::internal_error("Semantic engine not initialized".to_string(), None))?;

                    let results = engine
                        .semantic_search_global_with_recency_bias(
                            &req.query,
                            Some(validated_limit),
                            date_range,
                            bias,
                        )
                        .await
                        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

                    // Format results consistently with JSON
                    let formatted: Vec<serde_json::Value> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "content": r.text_content,
                                "score": r.combined_score,
                                "session_id": r.session_id,
                                "type": format!("{:?}", r.content_type),
                                "timestamp": r.timestamp.to_rfc3339()
                            })
                        })
                        .collect();

                    MCPToolResult::success(
                        format!("Found {} results", results.len()),
                        Some(serde_json::json!({ "results": formatted })),
                    )
                } else {
                    crate::tools::mcp::semantic_search_global(
                        req.query.clone(),
                        Some(validated_limit),
                        req.date_from.clone(),
                        req.date_to.clone(),
                        req.interaction_type.clone(),
                        None,
                    )
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?
                };

                Ok(CallToolResult::success(vec![Content::text(search_results.message)]))
            }
        }
    }

    // =========================================================================
    // Tool 4: get_structured_summary (extended)
    // =========================================================================
    #[tool(
        description = "Get session summary. Use 'include' to specify sections: decisions, insights, entities, or all (default).",
        input_schema = rmcp::handler::server::common::schema_for_type::<GetStructuredSummaryRequest>()
    )]
    async fn get_structured_summary(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        let req: GetStructuredSummaryRequest = coerce_and_validate(params.0)
            .map_err(|e| {
                if e.message.contains("session_id") {
                    e.clone()
                        .with_parameter_path("session_id".to_string())
                        .with_expected_type("UUID string (36 chars with hyphens)")
                        .with_hint("Provide a valid session UUID to get its summary")
                        .to_mcp_error()
                } else if e.message.contains("include") {
                    e.clone()
                        .with_parameter_path("include".to_string())
                        .with_expected_type("array of: decisions, insights, entities, questions, all")
                        .with_hint("Specify which sections to include, or omit for all. Valid values: decisions, insights, entities, questions, all")
                        .to_mcp_error()
                } else {
                    e.to_mcp_error()
                }
            })?;

        validate_session_id(&req.session_id).map_err(|e| e.to_mcp_error())?;

        // Validate all limit parameters
        let validated_decisions_limit =
            validate_limits(req.decisions_limit, 10, 100).map_err(|e| {
                e.with_parameter_path("decisions_limit".to_string())
                    .to_mcp_error()
            })?;
        let validated_entities_limit =
            validate_limits(req.entities_limit, 20, 200).map_err(|e| {
                e.with_parameter_path("entities_limit".to_string())
                    .to_mcp_error()
            })?;
        let validated_questions_limit =
            validate_limits(req.questions_limit, 5, 50).map_err(|e| {
                e.with_parameter_path("questions_limit".to_string())
                    .to_mcp_error()
            })?;
        let validated_concepts_limit =
            validate_limits(req.concepts_limit, 10, 50).map_err(|e| {
                e.with_parameter_path("concepts_limit".to_string())
                    .to_mcp_error()
            })?;

        // Determine which sections to include
        let include = req
            .include
            .as_ref()
            .map(|v| v.iter().map(|s| s.to_lowercase()).collect::<Vec<_>>());

        let include_all = include.is_none()
            || include
                .as_ref()
                .is_some_and(|v| v.contains(&"all".to_string()));
        let include_decisions = include_all
            || include
                .as_ref()
                .is_some_and(|v| v.contains(&"decisions".to_string()));
        let include_insights = include_all
            || include
                .as_ref()
                .is_some_and(|v| v.contains(&"insights".to_string()));
        let include_entities = include_all
            || include
                .as_ref()
                .is_some_and(|v| v.contains(&"entities".to_string()));

        let mut result_parts = Vec::new();

        // Get full summary if all or no specific sections
        if include_all {
            match crate::tools::mcp::get_structured_summary(
                req.session_id.clone(),
                Some(validated_decisions_limit),
                Some(validated_entities_limit),
                Some(validated_questions_limit),
                Some(validated_concepts_limit),
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
                    Some(validated_decisions_limit),
                )
                .await
                {
                    Ok(result) => result_parts.push(format!("## Insights\n{}", result.message)),
                    Err(e) => result_parts.push(format!("## Insights\nError: {}", e)),
                }
            }

            if include_entities {
                match crate::tools::mcp::get_entity_importance_analysis(
                    req.session_id.clone(),
                    req.entities_limit,
                    req.min_confidence,
                )
                .await
                {
                    Ok(result) => result_parts.push(format!("## Entities\n{}", result.message)),
                    Err(e) => result_parts.push(format!("## Entities\nError: {}", e)),
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(
            result_parts.join("\n\n"),
        )]))
    }

    // =========================================================================
    // Tool 5: query_conversation_context
    // =========================================================================
    #[tool(
        description = "Query session data. Types: find_related_entities, get_entity_context, search_updates, entity_importance, entity_network, etc.",
        input_schema = rmcp::handler::server::common::schema_for_type::<QueryConversationContextRequest>()
    )]
    async fn query_conversation_context(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        let req: QueryConversationContextRequest = coerce_and_validate(params.0)
            .map_err(|e| {
                if e.message.contains("session_id") {
                    e.clone()
                        .with_parameter_path("session_id".to_string())
                        .with_expected_type("UUID string (36 chars with hyphens)")
                        .with_hint("Provide the session UUID you want to query")
                        .to_mcp_error()
                } else if e.message.contains("query_type") {
                    e.clone()
                        .with_parameter_path("query_type".to_string())
                        .with_expected_type("query type string (e.g., entity_importance, entity_network, find_related_entities)")
                        .with_hint("Specify the type of query: entity_importance, entity_network, find_related_entities, get_entity_context, search_updates")
                        .to_mcp_error()
                } else if e.message.contains("parameters") {
                    e.clone()
                        .with_parameter_path("parameters".to_string())
                        .with_expected_type("object with string key-value pairs")
                        .with_hint("Query parameters vary by query_type. Provide as key-value pairs.")
                        .to_mcp_error()
                } else {
                    e.to_mcp_error()
                }
            })?;

        let uuid = validate_session_id(&req.session_id).map_err(|e| e.to_mcp_error())?;

        // Handle special query types that map to deprecated tools
        match req.query_type.to_lowercase().as_str() {
            "entity_importance" => {
                let limit = req
                    .parameters
                    .get("limit")
                    .and_then(|s: &String| s.parse().ok());
                let min_importance = req
                    .parameters
                    .get("min_importance")
                    .and_then(|s: &String| s.parse().ok());

                match crate::tools::mcp::get_entity_importance_analysis(
                    req.session_id.clone(),
                    limit,
                    min_importance,
                )
                .await
                {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "entity_network" => {
                let center_entity = req.parameters.get("center_entity").cloned();
                let max_entities = req
                    .parameters
                    .get("max_entities")
                    .and_then(|s: &String| s.parse().ok());
                let max_relationships = req
                    .parameters
                    .get("max_relationships")
                    .and_then(|s: &String| s.parse().ok());

                match crate::tools::mcp::get_entity_network_view(
                    req.session_id.clone(),
                    center_entity,
                    max_entities,
                    max_relationships,
                )
                .await
                {
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
    #[tool(
        description = "Manage workspaces. Actions: create, list, get, delete, add_session, remove_session",
        input_schema = rmcp::handler::server::common::schema_for_type::<ManageWorkspaceRequest>()
    )]
    async fn manage_workspace(
        &self,
        params: Parameters<serde_json::Value>,
    ) -> Result<CallToolResult, McpError> {
        let req: ManageWorkspaceRequest = coerce_and_validate(params.0)
            .map_err(|e| {
                if e.message.contains("action") {
                    e.clone()
                        .with_parameter_path("action".to_string())
                        .with_expected_type("one of: create, list, get, delete, add_session, remove_session")
                        .with_hint("Valid actions: create (workspace), list (all), get (workspace details), delete (workspace), add_session (to workspace), remove_session (from workspace)")
                        .to_mcp_error()
                } else {
                    e.to_mcp_error()
                }
            })?;

        validate_workspace_action(&req.action).map_err(|e| e.to_mcp_error())?;

        match req.action.to_lowercase().as_str() {
            "create" => {
                let name = req.name.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, "name required"),
                        None,
                    )
                    .with_parameter_path("name".to_string())
                    .with_expected_type("workspace name string")
                    .with_hint(
                        "For create action, provide 'name' and 'description' for the new workspace",
                    )
                    .to_mcp_error()
                })?;
                let description = req.description.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "description required",
                        ),
                        None,
                    )
                    .with_parameter_path("description".to_string())
                    .with_expected_type("workspace description string")
                    .with_hint(
                        "For create action, provide 'name' and 'description' for the new workspace",
                    )
                    .to_mcp_error()
                })?;

                match crate::tools::mcp::create_workspace(name.clone(), description.clone()).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "list" => match crate::tools::mcp::list_workspaces().await {
                Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                Err(e) => Err(McpError::internal_error(e.to_string(), None)),
            },
            "get" => {
                let workspace_id = req.workspace_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "workspace_id required",
                        ),
                        None,
                    )
                    .with_parameter_path("workspace_id".to_string())
                    .with_expected_type("workspace UUID string")
                    .with_hint(
                        "For get action, provide 'workspace_id' of the workspace to retrieve",
                    )
                    .to_mcp_error()
                })?;

                let uuid = validate_workspace_id(workspace_id).map_err(|e| e.to_mcp_error())?;

                match crate::tools::mcp::get_workspace(uuid).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "delete" => {
                let workspace_id = req.workspace_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "workspace_id required",
                        ),
                        None,
                    )
                    .with_parameter_path("workspace_id".to_string())
                    .with_expected_type("workspace UUID string")
                    .with_hint(
                        "For delete action, provide 'workspace_id' of the workspace to delete",
                    )
                    .to_mcp_error()
                })?;

                let uuid = validate_workspace_id(workspace_id).map_err(|e| e.to_mcp_error())?;

                match crate::tools::mcp::delete_workspace(uuid).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "add_session" => {
                let workspace_id = req.workspace_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "workspace_id required",
                        ),
                        None,
                    )
                    .with_parameter_path("workspace_id".to_string())
                    .with_expected_type("workspace UUID string")
                    .with_hint("For add_session action, provide 'workspace_id' and 'session_id'")
                    .to_mcp_error()
                })?;
                let session_id = req.session_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "session_id required",
                        ),
                        None,
                    )
                    .with_parameter_path("session_id".to_string())
                    .with_expected_type("session UUID string")
                    .with_hint("For add_session action, provide 'workspace_id' and 'session_id'")
                    .to_mcp_error()
                })?;
                let role = req.role.clone().unwrap_or_else(|| "related".to_string());

                let ws_uuid = validate_workspace_id(workspace_id).map_err(|e| e.to_mcp_error())?;
                let sess_uuid = validate_session_id(session_id).map_err(|e| e.to_mcp_error())?;

                validate_session_role(&role).map_err(|e| e.to_mcp_error())?;

                match crate::tools::mcp::add_session_to_workspace(ws_uuid, sess_uuid, role).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            "remove_session" => {
                let workspace_id = req.workspace_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "workspace_id required",
                        ),
                        None,
                    )
                    .with_parameter_path("workspace_id".to_string())
                    .with_expected_type("workspace UUID string")
                    .with_hint("For remove_session action, provide 'workspace_id' and 'session_id'")
                    .to_mcp_error()
                })?;
                let session_id = req.session_id.as_ref().ok_or_else(|| {
                    CoercionError::new(
                        "Missing required parameter",
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "session_id required",
                        ),
                        None,
                    )
                    .with_parameter_path("session_id".to_string())
                    .with_expected_type("session UUID string")
                    .with_hint("For remove_session action, provide 'workspace_id' and 'session_id'")
                    .to_mcp_error()
                })?;

                let ws_uuid = validate_workspace_id(workspace_id).map_err(|e| e.to_mcp_error())?;
                let sess_uuid = validate_session_id(session_id).map_err(|e| e.to_mcp_error())?;

                match crate::tools::mcp::remove_session_from_workspace(ws_uuid, sess_uuid).await {
                    Ok(result) => Ok(CallToolResult::success(vec![Content::text(result.message)])),
                    Err(e) => Err(McpError::internal_error(e.to_string(), None)),
                }
            }
            _ => Err(McpError::invalid_params(
                format!(
                    "Invalid action '{}'. Use: create | list | get | delete | add_session | remove_session",
                    req.action
                ),
                None,
            )),
        }
    }
}

// NOTE: We implement ServerHandler manually instead of using #[tool_handler] macro
// to strip $schema from tool input schemas for broader MCP client compatibility.
// Many clients (Cursor, Windsurf, Continue.dev) don't support JSON Schema draft/2020-12.
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

    // Custom list_tools that strips $schema from input schemas for client compatibility
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let tools = self.tool_router.list_all();

        // Strip $schema from each tool's input_schema for broader client compatibility
        // Clone and modify since input_schema is Arc<JsonObject>
        let tools: Vec<Tool> = tools
            .into_iter()
            .map(|mut tool| {
                let mut schema = (*tool.input_schema).clone();
                schema.remove("$schema");
                // Also strip from $defs if present
                if let Some(defs) = schema.get_mut("$defs") {
                    if let Some(defs_obj) = defs.as_object_mut() {
                        for (_, def) in defs_obj.iter_mut() {
                            if let Some(def_obj) = def.as_object_mut() {
                                def_obj.remove("$schema");
                            }
                        }
                    }
                }
                tool.input_schema = std::sync::Arc::new(schema);
                tool
            })
            .collect();

        Ok(ListToolsResult {
            tools,
            meta: None,
            next_cursor: None,
        })
    }

    // Route tool calls to the tool router
    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let tcc = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }
}
