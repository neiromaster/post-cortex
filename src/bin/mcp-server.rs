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

use clap::Parser;
use dashmap::DashMap;
use post_cortex::ConversationMemorySystem;
use post_cortex::tools::mcp::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

/// Post-Cortex MCP Server - Intelligent conversation memory system
#[derive(Parser)]
#[command(name = "post-cortex")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Production-grade intelligent conversation memory system for AI assistants", long_about = None)]
struct Cli {}

/// Universal error response generator - eliminates 24+ code duplications
fn create_error_response(id: Value, code: i32, message: String) -> Value {
    let id_clone = id.clone();
    let response = MCPResponse::<()> {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(MCPError { code, message }),
    };
    serde_json::to_value(response).unwrap_or_else(|e| {
        error!("Failed to serialize error response: {}", e);
        json!({
            "jsonrpc": "2.0",
            "id": id_clone,
            "result": null,
            "error": {
                "code": -32603,
                "message": "Internal serialization error"
            }
        })
    })
}

/// Universal success response generator
fn create_success_response<T: Serialize>(id: Value, result: T) -> Value {
    safe_to_value(MCPResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(result),
        error: None,
    })
}

/// Safe JSON serialization with error logging
fn safe_to_value<T: Serialize>(data: T) -> Value {
    serde_json::to_value(data).unwrap_or_else(|e| {
        error!("Failed to serialize data: {}", e);
        json!(null)
    })
}

/// Universal function to handle tool execution results - eliminates 20+ massive code duplications
fn handle_tool_result<T: Serialize>(
    id: Value,
    result: Result<T, anyhow::Error>,
    success_formatter: fn(T) -> ToolCallResult,
    error_code: i32,
    error_prefix: &str,
    tool_start: std::time::Instant,
    tool_name: &str,
) -> Value {
    match result {
        Ok(data) => {
            info!(
                "MCP-SERVER: {} completed in {:?}",
                tool_name,
                tool_start.elapsed()
            );
            let response_result = success_formatter(data);
            create_success_response(id, response_result)
        }
        Err(e) => {
            error!("MCP-SERVER: {} failed: {}", tool_name, e);
            create_error_response(id, error_code, format!("{}: {}", error_prefix, e))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CodeReferenceInput {
    file_path: Option<String>,
    start_line: Option<u64>,
    end_line: Option<u64>,
    code_snippet: Option<String>,
    commit_hash: Option<String>,
    branch: Option<String>,
    change_description: Option<String>,
}

#[derive(Serialize)]
struct ToolProperty {
    #[serde(rename = "type")]
    prop_type: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    items: Option<Box<ToolProperty>>,
}

#[derive(Serialize)]
struct InputSchema {
    #[serde(rename = "type")]
    schema_type: String,
    properties: std::collections::HashMap<String, ToolProperty>,
    required: Vec<String>,
}

#[derive(Serialize)]
struct Tool {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: InputSchema,
}

#[derive(Serialize)]
struct MCPResponse<T> {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<MCPError>,
}

#[derive(Serialize)]
struct MCPError {
    code: i32,
    message: String,
}

#[derive(Serialize)]
struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    protocol_version: String,
    capabilities: Capabilities,
    #[serde(rename = "serverInfo")]
    server_info: ServerInfo,
}

#[derive(Serialize)]
struct Capabilities {
    tools: std::collections::HashMap<String, Value>,
}

#[derive(Serialize)]
struct ServerInfo {
    name: String,
    version: String,
}

#[derive(Serialize)]
struct ToolsResult {
    tools: Vec<Value>,
}

#[derive(Serialize)]
struct ToolCallResult {
    content: Vec<ContentItem>,
}

#[derive(Serialize)]
struct ContentItem {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

struct MCPServer {
    sessions: DashMap<String, Uuid>,
    memory_system: Arc<ConversationMemorySystem>,
}

impl MCPServer {
    async fn new() -> anyhow::Result<Self> {
        let memory_system = get_memory_system().await?;
        eprintln!("MCP Server: Using RealRocksDBStorage for high performance");
        Ok(Self {
            sessions: DashMap::new(),
            memory_system,
        })
    }

    #[instrument(skip(self, request), fields(method = %request["method"].as_str().unwrap_or("unknown")))]
    async fn handle_request(&self, request: Value) -> Value {
        let method = request["method"].as_str().unwrap_or("");
        let params = &request["params"];
        let id = request["id"].clone();

        debug!("Processing MCP request: method={}", method);

        let start_time = std::time::Instant::now();
        let result = match method {
            "initialize" => {
                info!("MCP-SERVER: Handling initialize request");
                let result = InitializeResult {
                    protocol_version: "2024-11-05".to_string(),
                    capabilities: Capabilities {
                        tools: std::collections::HashMap::new(),
                    },
                    server_info: ServerInfo {
                        name: "post-cortex".to_string(),
                        version: "0.1.0".to_string(),
                    },
                };
                let id_clone = id.clone();
                serde_json::to_value(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(result),
                    error: None,
                })
                .unwrap_or_else(|e| {
                    error!("Failed to serialize initialize response: {}", e);
                    json!({
                        "jsonrpc": "2.0",
                        "id": id_clone,
                        "result": null,
                        "error": {
                            "code": -32603,
                            "message": "Internal serialization error"
                        }
                    })
                })
            }
            "tools/list" => {
                info!("MCP-SERVER: Handling tools/list request");
                let result = ToolsResult {
                    tools: Self::get_tool_definitions()
                        .into_iter()
                        .map(safe_to_value)
                        .collect::<Vec<_>>(),
                };
                let id_clone = id.clone();
                serde_json::to_value(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(result),
                    error: None,
                })
                .unwrap_or_else(|e| {
                    error!("Failed to serialize tools/list response: {}", e);
                    json!({
                        "jsonrpc": "2.0",
                        "id": id_clone,
                        "result": null,
                        "error": {
                            "code": -32603,
                            "message": "Internal serialization error"
                        }
                    })
                })
            }
            "tools/call" => {
                info!("MCP-SERVER: Handling tools/call request");
                let tool_name = params["name"].as_str().unwrap_or("");
                let arguments = &params["arguments"];

                info!("tools/call received - tool_name: {}", tool_name);
                let tool_start = std::time::Instant::now();

                let tool_result = match tool_name {
                    "create_session" => handle_tool_result(
                        id.clone(),
                        self.create_session(arguments).await,
                        |session_id| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: format!("Created new session: {}", session_id),
                            }],
                        },
                        -32603,
                        "Failed to create session",
                        tool_start,
                        "create_session",
                    ),
                    "update_conversation_context" => handle_tool_result(
                        id.clone(),
                        self.update_context(arguments).await,
                        |result| {
                            info!("tools/call: update_context completed successfully");
                            ToolCallResult {
                                content: vec![ContentItem {
                                    content_type: "text".to_string(),
                                    text: result,
                                }],
                            }
                        },
                        -32603,
                        "Failed to update context",
                        tool_start,
                        "update_conversation_context",
                    ),
                    "bulk_update_conversation_context" => handle_tool_result(
                        id.clone(),
                        self.bulk_update_context(arguments).await,
                        |result| {
                            info!("tools/call: bulk_update_context completed successfully");
                            ToolCallResult {
                                content: vec![ContentItem {
                                    content_type: "text".to_string(),
                                    text: result,
                                }],
                            }
                        },
                        -32603,
                        "Failed to bulk update context",
                        tool_start,
                        "bulk_update_conversation_context",
                    ),
                    "query_conversation_context" => handle_tool_result(
                        id.clone(),
                        self.query_context(arguments).await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: result,
                            }],
                        },
                        -32603,
                        "Failed to query context",
                        tool_start,
                        "query_conversation_context",
                    ),
                    "create_session_checkpoint" => handle_tool_result(
                        id.clone(),
                        self.create_checkpoint(arguments).await,
                        |checkpoint_id| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: format!("Created checkpoint: {}", checkpoint_id),
                            }],
                        },
                        -32603,
                        "Failed to create checkpoint",
                        tool_start,
                        "create_session_checkpoint",
                    ),
                    "list_sessions" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::list_sessions().await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32000,
                        "Error listing sessions",
                        tool_start,
                        "list_sessions",
                    ),

                    "search_sessions" => {
                        let query = arguments["query"].as_str().unwrap_or("").to_string();
                        handle_tool_result(
                            id.clone(),
                            post_cortex::tools::mcp::search_sessions(query.clone()).await,
                            |result| ToolCallResult {
                                content: vec![ContentItem {
                                    content_type: "text".to_string(),
                                    text: serde_json::to_string_pretty(&result)
                                        .unwrap_or_default()
                                        .to_string(),
                                }],
                            },
                            -32000,
                            "Error searching sessions",
                            tool_start,
                            "search_sessions",
                        )
                    }

                    "load_session" => {
                        let session_id_str = arguments["session_id"].as_str().unwrap_or("");
                        match Uuid::parse_str(session_id_str) {
                            Ok(session_uuid) => {
                                let system = &self.memory_system;
                                match post_cortex::tools::mcp::load_session_with_system(
                                    session_uuid,
                                    system,
                                )
                                .await
                                {
                                    Ok(result) => {
                                        // Add the loaded session to the sessions HashMap
                                        self.sessions
                                            .insert(session_id_str.to_string(), session_uuid);
                                        info!(
                                            "MCP Server: Loaded session {} -> {} and added to sessions HashMap",
                                            session_id_str, session_uuid
                                        );
                                        let response_result = ToolCallResult {
                                            content: vec![ContentItem {
                                                content_type: "text".to_string(),
                                                text: serde_json::to_string_pretty(&result)
                                                    .unwrap_or_default()
                                                    .to_string(),
                                            }],
                                        };
                                        create_success_response(id.clone(), response_result)
                                    }
                                    Err(e) => create_error_response(
                                        id.clone(),
                                        -32603,
                                        format!("Failed to load session: {}", e),
                                    ),
                                }
                            }
                            Err(e) => create_error_response(
                                id.clone(),
                                -32602,
                                format!("Invalid session ID: {}", e),
                            ),
                        }
                    }

                    "update_session_metadata" => {
                        let session_id_str = arguments["session_id"].as_str().unwrap_or("");
                        let name = arguments["name"].as_str().map(|s| s.to_string());
                        let description = arguments["description"].as_str().map(|s| s.to_string());

                        match Uuid::parse_str(session_id_str) {
                            Ok(session_uuid) => handle_tool_result(
                                id.clone(),
                                post_cortex::tools::mcp::update_session_metadata(
                                    session_uuid,
                                    name,
                                    description,
                                )
                                .await,
                                |result| ToolCallResult {
                                    content: vec![ContentItem {
                                        content_type: "text".to_string(),
                                        text: serde_json::to_string_pretty(&result)
                                            .unwrap_or_default()
                                            .to_string(),
                                    }],
                                },
                                -32603,
                                "Failed to update session metadata",
                                tool_start,
                                "update_session_metadata",
                            ),
                            Err(e) => create_error_response(
                                id.clone(),
                                -32602,
                                format!("Invalid session ID: {}", e),
                            ),
                        }
                    }

                    "get_structured_summary" => {
                        let session_id = arguments["session_id"].as_str().unwrap_or("").to_string();
                        let decisions_limit = arguments
                            .get("decisions_limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize);
                        let entities_limit = arguments
                            .get("entities_limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize);
                        let questions_limit = arguments
                            .get("questions_limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize);
                        let concepts_limit = arguments
                            .get("concepts_limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize);
                        let min_confidence = arguments
                            .get("min_confidence")
                            .and_then(|v| v.as_f64())
                            .map(|v| v as f32);
                        let compact = arguments.get("compact").and_then(|v| v.as_bool());

                        handle_tool_result(
                            id.clone(),
                            post_cortex::tools::mcp::get_structured_summary(
                                session_id,
                                decisions_limit,
                                entities_limit,
                                questions_limit,
                                concepts_limit,
                                min_confidence,
                                compact,
                            )
                            .await,
                            |result| ToolCallResult {
                                content: vec![ContentItem {
                                    content_type: "text".to_string(),
                                    text: serde_json::to_string_pretty(&result)
                                        .unwrap_or_default()
                                        .to_string(),
                                }],
                            },
                            -32603,
                            "Failed to get structured summary",
                            tool_start,
                            "get_structured_summary",
                        )
                    }

                    "get_key_decisions" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_key_decisions(
                            arguments["session_id"].as_str().unwrap_or("").to_string(),
                        )
                        .await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get key decisions",
                        tool_start,
                        "get_key_decisions",
                    ),

                    "get_key_insights" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_key_insights(
                            arguments["session_id"].as_str().unwrap_or("").to_string(),
                            arguments["limit"].as_u64().map(|l| l as usize),
                        )
                        .await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get key insights",
                        tool_start,
                        "get_key_insights",
                    ),

                    "get_entity_importance_analysis" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_entity_importance_analysis(
                            arguments["session_id"].as_str().unwrap_or("").to_string(),
                            arguments
                                .get("limit")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                            arguments
                                .get("min_importance")
                                .and_then(|v| v.as_f64())
                                .map(|v| v as f32),
                        )
                        .await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get entity importance analysis",
                        tool_start,
                        "get_entity_importance_analysis",
                    ),

                    "get_entity_network_view" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_entity_network_view(
                            arguments["session_id"].as_str().unwrap_or("").to_string(),
                            arguments["center_entity"].as_str().map(|s| s.to_string()),
                            arguments
                                .get("max_entities")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                            arguments
                                .get("max_relationships")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as usize),
                        )
                        .await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get entity network view",
                        tool_start,
                        "get_entity_network_view",
                    ),

                    "get_session_statistics" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_session_statistics(
                            arguments["session_id"].as_str().unwrap_or("").to_string(),
                        )
                        .await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get session statistics",
                        tool_start,
                        "get_session_statistics",
                    ),

                    "semantic_search_global" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::semantic_search_global(
                            arguments["query"].as_str().unwrap_or("").to_string(),
                            arguments
                                .get("limit")
                                .and_then(|v| v.as_u64())
                                .map(|l| l as usize),
                            arguments
                                .get("date_from")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            arguments
                                .get("date_to")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string()),
                            arguments
                                .get("interaction_type")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str().map(String::from))
                                        .collect()
                                }),
                        )
                        .await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to perform semantic search global",
                        tool_start,
                        "semantic_search_global",
                    ),

                    "semantic_search_session" => {
                        info!("=== semantic_search_session arguments: {:?}", arguments);
                        let session_id = arguments["session_id"].as_str().unwrap_or("").to_string();
                        let query = arguments["query"].as_str().unwrap_or("").to_string();
                        let limit = arguments
                            .get("limit")
                            .and_then(|v| v.as_u64())
                            .map(|l| l as usize);
                        let date_from = arguments
                            .get("date_from")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let date_to = arguments
                            .get("date_to")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let interaction_type = arguments
                            .get("interaction_type")
                            .and_then(|v| {
                                info!("Found interaction_type in arguments: {:?}", v);
                                v.as_array()
                            })
                            .map(|arr| {
                                let result: Vec<String> = arr
                                    .iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect();
                                info!("Parsed interaction_type array: {:?}", result);
                                result
                            });

                        info!("Final interaction_type value: {:?}", interaction_type);

                        match uuid::Uuid::parse_str(&session_id) {
                            Ok(uuid) => handle_tool_result(
                                id.clone(),
                                post_cortex::tools::mcp::semantic_search_session(
                                    uuid,
                                    query,
                                    limit,
                                    date_from,
                                    date_to,
                                    interaction_type,
                                )
                                .await,
                                |result| ToolCallResult {
                                    content: vec![ContentItem {
                                        content_type: "text".to_string(),
                                        text: serde_json::to_string_pretty(&result)
                                            .unwrap_or_default()
                                            .to_string(),
                                    }],
                                },
                                -32603,
                                "Failed to perform semantic search session",
                                tool_start,
                                "semantic_search_session",
                            ),
                            Err(e) => create_error_response(
                                id.clone(),
                                -32602,
                                format!("Invalid session_id format: {}", e),
                            ),
                        }
                    }

                    "find_related_content" => {
                        let session_id = arguments["session_id"].as_str().unwrap_or("").to_string();
                        let topic = arguments["topic"].as_str().unwrap_or("").to_string();
                        let limit = arguments
                            .get("limit")
                            .and_then(|v| v.as_u64())
                            .map(|l| l as usize);

                        match uuid::Uuid::parse_str(&session_id) {
                            Ok(uuid) => {
                                match post_cortex::tools::mcp::find_related_content(
                                    uuid, topic, limit,
                                )
                                .await
                                {
                                    Ok(result) => {
                                        let response_result = ToolCallResult {
                                            content: vec![ContentItem {
                                                content_type: "text".to_string(),
                                                text: serde_json::to_string_pretty(&result)
                                                    .unwrap_or_default()
                                                    .to_string(),
                                            }],
                                        };
                                        create_success_response(id.clone(), response_result)
                                    }
                                    Err(e) => create_error_response(
                                        id.clone(),
                                        -32603,
                                        format!("Find related content failed: {}", e),
                                    ),
                                }
                            }
                            Err(e) => create_error_response(
                                id.clone(),
                                -32602,
                                format!("Invalid session_id format: {}", e),
                            ),
                        }
                    }

                    "vectorize_session" => {
                        let session_id = arguments["session_id"].as_str().unwrap_or("").to_string();

                        match uuid::Uuid::parse_str(&session_id) {
                            Ok(uuid) => handle_tool_result(
                                id.clone(),
                                post_cortex::tools::mcp::vectorize_session(uuid).await,
                                |result| ToolCallResult {
                                    content: vec![ContentItem {
                                        content_type: "text".to_string(),
                                        text: serde_json::to_string_pretty(&result)
                                            .unwrap_or_default()
                                            .to_string(),
                                    }],
                                },
                                -32603,
                                "Failed to vectorize session",
                                tool_start,
                                "vectorize_session",
                            ),
                            Err(e) => create_error_response(
                                id.clone(),
                                -32602,
                                format!("Invalid session_id format: {}", e),
                            ),
                        }
                    }

                    "get_vectorization_stats" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_vectorization_stats().await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get vectorization stats",
                        tool_start,
                        "get_vectorization_stats",
                    ),

                    "get_tool_catalog" => handle_tool_result(
                        id.clone(),
                        post_cortex::tools::mcp::get_tool_catalog().await,
                        |result| ToolCallResult {
                            content: vec![ContentItem {
                                content_type: "text".to_string(),
                                text: serde_json::to_string_pretty(&result)
                                    .unwrap_or_default()
                                    .to_string(),
                            }],
                        },
                        -32603,
                        "Failed to get tool catalog",
                        tool_start,
                        "get_tool_catalog",
                    ),

                    _ => {
                        error!("MCP-SERVER: Unknown tool: {tool_name}");
                        create_error_response(
                            id.clone(),
                            -32601,
                            format!("Unknown tool: {tool_name}"),
                        )
                    }
                };
                info!(
                    "MCP-SERVER: Tool '{}' completed in {:?}",
                    tool_name,
                    tool_start.elapsed()
                );
                tool_result
            }
            _ => {
                let id_clone = id.clone();
                let response = MCPResponse::<()> {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: None,
                    error: Some(MCPError {
                        code: -32601,
                        message: format!("Method not found: {method}"),
                    }),
                };
                serde_json::to_value(response).unwrap_or_else(|e| {
                    error!("Failed to serialize method not found response: {}", e);
                    json!({
                        "jsonrpc": "2.0",
                        "id": id_clone,
                        "result": null,
                        "error": {
                            "code": -32603,
                            "message": "Internal serialization error"
                        }
                    })
                })
            }
        };

        info!(
            "MCP-SERVER: Request '{}' completed in {:?}",
            method,
            start_time.elapsed()
        );
        result
    }

    fn get_tool_definitions() -> Vec<Tool> {
        use std::collections::HashMap;

        vec![
            Tool {
                name: "create_session".to_string(),
                description: "Create a new conversation session with optional name and description. Sessions store conversation history, entity relationships, and context across multiple interactions.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("name".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional human-readable name for the session (e.g., 'Project Alpha Discussion')".to_string(),
                            items: None,
                        });
                        props.insert("description".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional detailed description of the session's purpose and content".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec![],
                }
            },
            Tool {
                name: "update_conversation_context".to_string(),
                description: "Add new interaction context to a session. Automatically extracts entities, relationships, and structured information. Supports Q&A pairs, code changes, decisions, and problem resolutions.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the target session".to_string(),
                            items: None,
                        });
                        props.insert("interaction_type".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Type of interaction: 'qa' (question-answer), 'code_change' (file modification), 'problem_solved' (issue resolution), 'decision_made' (architectural choice)".to_string(),
                            items: None,
                        });
                        props.insert("content".to_string(), ToolProperty {
                            prop_type: "object".to_string(),
                            description: "Content object with keys depending on interaction_type. For 'qa': {question, answer}. For 'decision_made': {decision, rationale}. For 'problem_solved': {problem, solution}".to_string(),
                            items: None,
                        });
                        props.insert("code_reference".to_string(), ToolProperty {
                            prop_type: "object".to_string(),
                            description: "Optional code reference with file_path, start_line, end_line, code_snippet, and change_description".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string(), "interaction_type".to_string(), "content".to_string()],
                }
            },
            Tool {
                name: "bulk_update_conversation_context".to_string(),
                description: "Add multiple interaction contexts to a session in a single batch operation. More efficient than multiple single updates. Returns statistics on success/failure counts.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the target session".to_string(),
                            items: None,
                        });
                        props.insert("updates".to_string(), ToolProperty {
                            prop_type: "array".to_string(),
                            description: "Array of context update objects, each containing: interaction_type, content, and optional code_reference".to_string(),
                            items: Some(Box::new(ToolProperty {
                                prop_type: "object".to_string(),
                                description: "Single context update with interaction_type, content object, and optional code_reference".to_string(),
                                items: None,
                            })),
                        });
                        props
                    },
                    required: vec!["session_id".to_string(), "updates".to_string()],
                }
            },
            Tool {
                name: "query_conversation_context".to_string(),
                description: "Query session data using various search types. Supports entity searches, relationship tracing, content searches, and structured summaries. Returns relevant context and relationships.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to query".to_string(),
                            items: None,
                        });
                        props.insert("query_type".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Query type: 'find_related_entities' (find connected concepts), 'get_entity_context' (get entity details), 'search_updates' (text search), 'get_most_important_entities' (top entities)".to_string(),
                            items: None,
                        });
                        props.insert("parameters".to_string(), ToolProperty {
                            prop_type: "object".to_string(),
                            description: "Query-specific parameters. For entity queries: {entity_name}. For searches: {keywords, query}. For relationship tracing: {from_entity, max_depth}".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string(), "query_type".to_string()],
                }
            },
            Tool {
                name: "create_session_checkpoint".to_string(),
                description: "Create a persistent snapshot of the current session state including all context, entities, and relationships. Useful for creating restore points or archiving session state.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to checkpoint".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "list_sessions".to_string(),
                description: "List all conversation sessions with their metadata including names, descriptions, creation dates, entity counts, and update counts. Shows both active and archived sessions.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::new(),
                    required: vec![],
                }
            },
            Tool {
                name: "load_session".to_string(),
                description: "Load an archived or inactive session into active memory for querying and updates. Required before using other session-specific operations on archived sessions.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to load into active memory".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "search_sessions".to_string(),
                description: "Search for sessions by name or description using text matching. Returns sessions that contain the search terms in their name or description fields.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("query".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Search text to match against session names and descriptions (case-insensitive partial matching)".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["query".to_string()],
                }
            },
            Tool {
                name: "update_session_metadata".to_string(),
                description: "Update the name and/or description of an existing session. Supports partial updates - provide only the fields you want to change. Existing values are preserved for omitted fields.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to update".to_string(),
                            items: None,
                        });
                        props.insert("name".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "New name for the session (optional - omit to keep current name)".to_string(),
                            items: None,
                        });
                        props.insert("description".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "New description for the session (optional - omit to keep current description)".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_structured_summary".to_string(),
                description: "Get intelligent summary of a session with decisions, questions, concepts, and entities. Supports pagination via limit parameters. For large sessions (>100 updates), use compact:true to prevent token overflow errors. Auto-compact activates if response exceeds 25K tokens. All limits are optional with sensible defaults.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to summarize".to_string(),
                            items: None,
                        });
                        props.insert("decisions_limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Max decisions to return (default: unlimited, sorted by confidence desc). Use to prevent token overflow. Example: 5 for top decisions only.".to_string(),
                            items: None,
                        });
                        props.insert("entities_limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Max entities to return (default: 20, sorted by importance). Example: 10 for focused list. Compact mode uses 10.".to_string(),
                            items: None,
                        });
                        props.insert("questions_limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Max questions to return (default: unlimited, newest first). Example: 10 for recent questions only.".to_string(),
                            items: None,
                        });
                        props.insert("concepts_limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Max concepts to return (default: unlimited, newest first). Example: 5 for latest topics only.".to_string(),
                            items: None,
                        });
                        props.insert("min_confidence".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Filter decisions by confidence (0.0-1.0, default: no filter). Example: 0.7 for Medium/High only. Compact uses 0.6.".to_string(),
                            items: None,
                        });
                        props.insert("compact".to_string(), ToolProperty {
                            prop_type: "boolean".to_string(),
                            description: "Quick overview mode: limits output to 5 decisions, 10 entities, 5 questions, 5 concepts (min confidence 0.6). RECOMMENDED for large sessions to prevent token overflow. Overrides other limit parameters. Auto-enables if response > 25K tokens.".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_key_decisions".to_string(),
                description: "Get timeline of key decisions made in a session with confidence levels and context.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_key_insights".to_string(),
                description: "Extract key insights from session data including high-confidence decisions, primary focus areas, and discussion patterns.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session".to_string(),
                            items: None,
                        });
                        props.insert("limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of insights to return (default: 5)".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_entity_importance_analysis".to_string(),
                description: "Get detailed analysis of entity importance with pagination support. Returns top entities by importance score, mention counts, and relationships.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session".to_string(),
                            items: None,
                        });
                        props.insert("limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of entities to return (default: 100)".to_string(),
                            items: None,
                        });
                        props.insert("min_importance".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Minimum importance score threshold for filtering entities (optional)".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_entity_network_view".to_string(),
                description: "Get network view of entity relationships with pagination support. Returns entities and relationships, optionally centered on a specific entity.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session".to_string(),
                            items: None,
                        });
                        props.insert("center_entity".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional entity name to center the network view on".to_string(),
                            items: None,
                        });
                        props.insert("max_entities".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of entities to return (default: 50)".to_string(),
                            items: None,
                        });
                        props.insert("max_relationships".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of relationships to return (default: 100)".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_session_statistics".to_string(),
                description: "Get comprehensive session statistics including update counts, entity counts, activity level, and duration analysis.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "semantic_search_global".to_string(),
                description: "Search semantically across all conversation sessions using AI embeddings to find contextually similar content. Supports optional date range filtering. Requires embeddings feature to be enabled.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("query".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "The search query - can be a question, topic, or concept to find semantically similar content".to_string(),
                            items: None,
                        });
                        props.insert("limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of results to return (default: 20)".to_string(),
                            items: None,
                        });
                        props.insert("date_from".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional start date for filtering results (RFC3339 format, e.g., '2025-11-15T00:00:00Z'). Must be used with date_to.".to_string(),
                            items: None,
                        });
                        props.insert("date_to".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional end date for filtering results (RFC3339 format, e.g., '2025-11-15T23:59:59Z'). Must be used with date_from.".to_string(),
                            items: None,
                        });
                        props.insert("interaction_type".to_string(), ToolProperty {
                            prop_type: "array".to_string(),
                            description: "Optional filter by interaction types. Array of strings: 'qa', 'code_change', 'decision_made', 'problem_solved'. Example: ['decision_made', 'qa']".to_string(),
                            items: Some(Box::new(ToolProperty {
                                prop_type: "string".to_string(),
                                description: "Interaction type".to_string(),
                                items: None,
                            })),
                        });
                        props
                    },
                    required: vec!["query".to_string()],
                }
            },
            Tool {
                name: "semantic_search_session".to_string(),
                description: "Search semantically within a specific conversation session using AI embeddings. Supports optional date range filtering. Finds contextually related content within the session scope.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to search within".to_string(),
                            items: None,
                        });
                        props.insert("query".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "The search query - can be a question, topic, or concept to find semantically similar content".to_string(),
                            items: None,
                        });
                        props.insert("limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of results to return (default: 20)".to_string(),
                            items: None,
                        });
                        props.insert("date_from".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional start date for filtering results (RFC3339 format, e.g., '2025-11-15T00:00:00Z'). Must be used with date_to.".to_string(),
                            items: None,
                        });
                        props.insert("date_to".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Optional end date for filtering results (RFC3339 format, e.g., '2025-11-15T23:59:59Z'). Must be used with date_from.".to_string(),
                            items: None,
                        });
                        props.insert("interaction_type".to_string(), ToolProperty {
                            prop_type: "array".to_string(),
                            description: "Optional filter by interaction types. Array of strings: 'qa', 'code_change', 'decision_made', 'problem_solved'. Example: ['decision_made', 'qa']".to_string(),
                            items: Some(Box::new(ToolProperty {
                                prop_type: "string".to_string(),
                                description: "Interaction type".to_string(),
                                items: None,
                            })),
                        });
                        props
                    },
                    required: vec!["session_id".to_string(), "query".to_string()],
                }
            },
            Tool {
                name: "find_related_content".to_string(),
                description: "Find related content across different sessions based on a topic or concept. Discovers connections between conversations using semantic similarity.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the current session (content from this session will be excluded from results)".to_string(),
                            items: None,
                        });
                        props.insert("topic".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "Topic or concept to find related content for".to_string(),
                            items: None,
                        });
                        props.insert("limit".to_string(), ToolProperty {
                            prop_type: "number".to_string(),
                            description: "Maximum number of results to return (default: 10)".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string(), "topic".to_string()],
                }
            },
            Tool {
                name: "vectorize_session".to_string(),
                description: "Generate embeddings for all content in a session to enable semantic search. This is automatically done when auto_vectorize_on_update is enabled.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: {
                        let mut props = HashMap::new();
                        props.insert("session_id".to_string(), ToolProperty {
                            prop_type: "string".to_string(),
                            description: "UUID of the session to vectorize".to_string(),
                            items: None,
                        });
                        props
                    },
                    required: vec!["session_id".to_string()],
                }
            },
            Tool {
                name: "get_vectorization_stats".to_string(),
                description: "Get statistics about the vectorization system including total embeddings, model information, and performance metrics.".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::new(),
                    required: vec![],
                }
            },
            Tool {
                name: "get_tool_catalog".to_string(),
                description: "📖 TOOL DISCOVERY - START HERE! Comprehensive catalog of all 20 post-cortex tools organized by category with usage guidance.

**When to use:**
- ⭐ FIRST TIME using post-cortex? Call this to see what's available!
- Need to find the right tool for your task
- Want to understand tool categories and workflows
- Exploring post-cortex capabilities

**What you get:**
- All 20 tools across 4 categories (Session Management, Context Operations, Semantic Search, Analysis & Insights)
- Each tool shows: name, description, when to use it, important notes
- Getting started workflow (4 steps)
- Most-used tools list (top 5)
- Practical tips (auto-vectorization, similarity scores, etc.)

**Recommended workflow:**
1. ⭐ Call get_tool_catalog FIRST to see all available tools
2. Review categories and find tools for your needs
3. Follow getting_started workflow for common tasks
4. Use specific tools from the catalog

**Example:**
Call mcp__post-cortex__get_tool_catalog() → get organized view of all 20 tools → pick the ones you need!".to_string(),
                input_schema: InputSchema {
                    schema_type: "object".to_string(),
                    properties: HashMap::new(),
                    required: vec![],
                }
            },
        ]
    }

    async fn create_session(&self, arguments: &Value) -> anyhow::Result<String> {
        info!(
            "MCP-SERVER: create_session() called with args: {:?}",
            arguments
        );

        let name = arguments["name"].as_str().map(|s| s.to_string());
        let description = arguments["description"].as_str().map(|s| s.to_string());
        info!(
            "MCP-SERVER: Parsed name: {:?}, description: {:?}",
            name, description
        );

        info!("MCP-SERVER: Getting memory system reference");
        let system = &self.memory_system;

        info!("MCP-SERVER: About to call system.create_session()");
        let session_uuid = system
            .create_session(name, description)
            .await
            .map_err(|e| {
                error!("MCP-SERVER: Failed to create session: {}", e);
                anyhow::Error::msg(e)
            })?;
        let session_id = session_uuid.to_string();

        info!("MCP-SERVER: Session created successfully: {}", session_id);

        info!("MCP-SERVER: Storing session in local map");
        self.sessions.insert(session_id.clone(), session_uuid);
        info!("MCP-SERVER: Session stored, returning ID");
        Ok(session_id)
    }

    #[instrument(skip(self, arguments), fields(session_id = %arguments["session_id"].as_str().unwrap_or("unknown")))]
    async fn update_context(&self, arguments: &Value) -> anyhow::Result<String> {
        let session_id_str = arguments["session_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing session_id"))?;

        let session_uuid = uuid::Uuid::parse_str(session_id_str)
            .map_err(|e| anyhow::anyhow!("Invalid session_id format: {}", e))?;

        let interaction_type = arguments["interaction_type"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing interaction_type"))?;

        let content_value = &arguments["content"];
        let mut content = HashMap::new();

        // Convert JSON object to HashMap
        if let Some(obj) = content_value.as_object() {
            for (key, value) in obj {
                if let Some(str_val) = value.as_str() {
                    content.insert(key.clone(), str_val.to_string());
                }
            }
        }
        let code_reference = if arguments["code_reference"].is_object() {
            match serde_json::from_value::<CodeReferenceInput>(arguments["code_reference"].clone())
            {
                Ok(code_ref_input) => {
                    info!("Serde deserialization successful");
                    Some(post_cortex::core::context_update::CodeReference {
                        file_path: code_ref_input.file_path.unwrap_or_default(),
                        start_line: code_ref_input.start_line.unwrap_or(0) as u32,
                        end_line: code_ref_input.end_line.unwrap_or(0) as u32,
                        code_snippet: code_ref_input.code_snippet.unwrap_or_default(),
                        commit_hash: code_ref_input.commit_hash,
                        branch: code_ref_input.branch,
                        change_description: code_ref_input.change_description.unwrap_or_default(),
                    })
                }
                Err(_) => None,
            }
        } else {
            None
        };
        info!("Code reference processed: {}", code_reference.is_some());

        let system = &self.memory_system;

        let result = update_conversation_context_with_system(
            interaction_type.to_string(),
            content,
            code_reference,
            session_uuid,
            system,
        )
        .await?;

        info!("update_conversation_context_with_system completed successfully");

        Ok(result.message)
    }

    async fn bulk_update_context(&self, arguments: &Value) -> anyhow::Result<String> {
        let session_id_str = arguments["session_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing session_id"))?;

        let session_uuid = uuid::Uuid::parse_str(session_id_str)
            .map_err(|e| anyhow::anyhow!("Invalid session_id format: {}", e))?;

        let updates_array = arguments["updates"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid updates array"))?;

        let mut updates: Vec<post_cortex::tools::mcp::ContextUpdateItem> = Vec::new();

        for update_value in updates_array {
            let interaction_type = update_value["interaction_type"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing interaction_type in update"))?
                .to_string();

            let content_value = &update_value["content"];
            let mut content = HashMap::new();

            if let Some(obj) = content_value.as_object() {
                for (key, value) in obj {
                    if let Some(str_val) = value.as_str() {
                        content.insert(key.clone(), str_val.to_string());
                    }
                }
            }

            let code_reference = if update_value["code_reference"].is_object() {
                match serde_json::from_value::<CodeReferenceInput>(
                    update_value["code_reference"].clone(),
                ) {
                    Ok(code_ref_input) => Some(post_cortex::core::context_update::CodeReference {
                        file_path: code_ref_input.file_path.unwrap_or_default(),
                        start_line: code_ref_input.start_line.unwrap_or(0) as u32,
                        end_line: code_ref_input.end_line.unwrap_or(0) as u32,
                        code_snippet: code_ref_input.code_snippet.unwrap_or_default(),
                        commit_hash: code_ref_input.commit_hash,
                        branch: code_ref_input.branch,
                        change_description: code_ref_input.change_description.unwrap_or_default(),
                    }),
                    Err(_) => None,
                }
            } else {
                None
            };

            updates.push(post_cortex::tools::mcp::ContextUpdateItem {
                interaction_type,
                content,
                code_reference,
            });
        }

        let result =
            post_cortex::tools::mcp::bulk_update_conversation_context(updates, session_uuid)
                .await?;

        info!("bulk_update_conversation_context completed successfully");

        Ok(serde_json::to_string_pretty(&result)?)
    }

    async fn query_context(&self, arguments: &Value) -> anyhow::Result<String> {
        let session_id_str = arguments["session_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing session_id"))?;

        // Parse UUID directly instead of checking self.sessions HashMap
        // This allows querying sessions that are loaded from RocksDB storage
        let session_uuid = uuid::Uuid::parse_str(session_id_str)
            .map_err(|e| anyhow::anyhow!("Invalid session_id format: {}", e))?;

        let query_type = arguments["query_type"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing query_type"))?;

        let mut parameters = HashMap::new();
        if let Some(params_obj) = arguments["parameters"].as_object() {
            for (key, value) in params_obj {
                if let Some(str_val) = value.as_str() {
                    parameters.insert(key.clone(), str_val.to_string());
                } else if let Some(num_val) = value.as_number() {
                    parameters.insert(key.clone(), num_val.to_string());
                }
            }
        }

        let system = &self.memory_system;
        let result = query_conversation_context_with_system(
            query_type.to_string(),
            parameters,
            session_uuid,
            system,
        )
        .await?;

        if let Some(data) = result.data {
            Ok(format!(
                "{}: {}",
                result.message,
                serde_json::to_string_pretty(&data)?
            ))
        } else {
            Ok(result.message)
        }
    }

    async fn create_checkpoint(&self, arguments: &Value) -> anyhow::Result<String> {
        let session_id_str = arguments["session_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing session_id"))?;

        // Parse UUID directly instead of checking self.sessions HashMap
        // This allows creating checkpoints for sessions loaded from RocksDB storage
        let session_uuid = uuid::Uuid::parse_str(session_id_str)
            .map_err(|e| anyhow::anyhow!("Invalid session_id format: {}", e))?;

        let system = &self.memory_system;
        let result = create_session_checkpoint_with_system(session_uuid, system).await?;
        Ok(format!("Created checkpoint: {}", result.message))
    }
}

fn init_logging() -> anyhow::Result<()> {
    use tracing_appender::rolling::{RollingFileAppender, Rotation};
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    // Create file appender for guaranteed persistence
    let file_appender = RollingFileAppender::new(Rotation::DAILY, "./logs", "mcp-server.log");

    // Create subscriber with both console and file output
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(true)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(
            fmt::layer()
                .with_writer(file_appender)
                .with_ansi(false)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .json(),
        )
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")))
        .init();

    info!("Logs being written to ./logs/mcp-server.log");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();

    // Initialize simple sync logging system
    if let Err(e) = init_logging() {
        eprintln!("Failed to initialize logging: {e}");
        return Err(e);
    }

    info!("Post-Cortex MCP Server starting with env_logger");

    let server = MCPServer::new().await?;
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;

        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<Value>(&line) {
            Ok(request) => {
                let response = server.handle_request(request).await;
                let response_str = serde_json::to_string(&response)?;
                writeln!(stdout, "{response_str}")?;
                stdout.flush()?;
            }
            Err(e) => {
                eprintln!("Failed to parse JSON: {e}");
                let error_response = json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                });
                let response_str = serde_json::to_string(&error_response)?;
                writeln!(stdout, "{response_str}")?;
                stdout.flush()?;
            }
        }
    }

    Ok(())
}
