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

//! Lock-free HTTP daemon server
//!
//! Provides HTTP/JSON-RPC endpoint for MCP protocol with zero blocking operations.

use crate::daemon::sse::LockFreeSSEBroadcaster;
use crate::ConversationMemorySystem;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info};
use uuid::Uuid;

/// Configuration for daemon server
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub host: String,
    pub port: u16,
    pub data_directory: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3737,
            data_directory: dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".post-cortex/data")
                .to_str()
                .unwrap()
                .to_string(),
        }
    }
}

/// Connection information (lock-free)
#[allow(dead_code)] // Will be used for connection tracking in future
struct ConnectionInfo {
    id: Uuid,
    connected_at: SystemTime,
    last_request: Arc<AtomicU64>,
    request_count: Arc<AtomicU64>,
}

/// Lock-free daemon server
pub struct LockFreeDaemonServer {
    /// Core memory system (already lock-free)
    memory_system: Arc<ConversationMemorySystem>,

    /// Lock-free connection tracking
    active_connections: Arc<DashMap<Uuid, ConnectionInfo>>,

    /// Lock-free SSE broadcaster
    sse_broadcaster: Arc<LockFreeSSEBroadcaster>,

    /// Atomic metrics
    connection_counter: Arc<AtomicU64>,
    total_requests: Arc<AtomicU64>,
    config: DaemonConfig,
}

impl LockFreeDaemonServer {
    pub async fn new(config: DaemonConfig) -> Result<Self, String> {
        info!("Initializing lock-free daemon server on {}:{}", config.host, config.port);

        // Create memory system
        let system_config = crate::SystemConfig {
            data_directory: config.data_directory.clone(),
            ..Default::default()
        };

        let memory_system = Arc::new(
            ConversationMemorySystem::new(system_config)
                .await
                .map_err(|e| format!("Failed to initialize memory system: {}", e))?,
        );

        info!("Memory system initialized successfully");

        Ok(Self {
            memory_system,
            active_connections: Arc::new(DashMap::new()),
            sse_broadcaster: Arc::new(LockFreeSSEBroadcaster::new()),
            connection_counter: Arc::new(AtomicU64::new(0)),
            total_requests: Arc::new(AtomicU64::new(0)),
            config,
        })
    }

    /// Start HTTP server
    pub async fn start(self) -> Result<(), String> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let server = Arc::new(self);

        // Build router
        let app = Router::new()
            .route("/health", get(health_check))
            .route("/mcp", post(handle_mcp_request))
            .route("/stats", get(get_stats))
            .layer(CorsLayer::permissive())
            .with_state(server.clone());

        info!("Starting HTTP server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| format!("Failed to bind to {}: {}", addr, e))?;

        info!("HTTP server listening on {}", addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| format!("Server error: {}", e))?;

        Ok(())
    }

    /// Get server statistics
    fn get_statistics(&self) -> ServerStats {
        ServerStats {
            active_connections: self.active_connections.len() as u64,
            total_connections: self.connection_counter.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            active_sse_clients: self.sse_broadcaster.active_clients(),
            total_sse_events: self.sse_broadcaster.total_events(),
            workspace_count: self.memory_system.workspace_manager.total_workspaces(),
        }
    }
}

/// MCP JSON-RPC request
#[derive(Debug, Deserialize)]
struct MCPRequest {
    #[allow(dead_code)] // Used for validation
    jsonrpc: String,
    id: serde_json::Value,
    method: String,
    #[allow(dead_code)] // Will be used for tool calls
    params: Option<serde_json::Value>,
}

/// MCP JSON-RPC response
#[derive(Debug, Serialize)]
struct MCPResponse {
    jsonrpc: String,
    id: serde_json::Value,
    result: Option<serde_json::Value>,
    error: Option<MCPError>,
}

#[derive(Debug, Serialize)]
struct MCPError {
    code: i32,
    message: String,
}

/// Server statistics
#[derive(Debug, Serialize)]
struct ServerStats {
    active_connections: u64,
    total_connections: u64,
    total_requests: u64,
    active_sse_clients: u64,
    total_sse_events: u64,
    workspace_count: u64,
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "post-cortex-daemon"
    }))
}

/// Statistics endpoint
async fn get_stats(State(server): State<Arc<LockFreeDaemonServer>>) -> impl IntoResponse {
    Json(server.get_statistics())
}

/// MCP request handler
async fn handle_mcp_request(
    State(server): State<Arc<LockFreeDaemonServer>>,
    Json(request): Json<MCPRequest>,
) -> Result<Json<MCPResponse>, AppError> {
    debug!("Handling MCP request: {}", request.method);

    server.total_requests.fetch_add(1, Ordering::Relaxed);

    // Route to appropriate handler based on method
    let result = match request.method.as_str() {
        "initialize" => handle_initialize(),
        "tools/list" => handle_tools_list(&server),
        "tools/call" => {
            handle_tool_call(&server, &request).await
        }
        _ => Err(format!("Unknown method: {}", request.method)),
    };

    match result {
        Ok(result_data) => Ok(Json(MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(result_data),
            error: None,
        })),
        Err(error_msg) => Ok(Json(MCPResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(MCPError {
                code: -32603,
                message: error_msg,
            }),
        })),
    }
}

fn handle_initialize() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "post-cortex-daemon",
            "version": env!("CARGO_PKG_VERSION")
        }
    }))
}

fn handle_tools_list(_server: &Arc<LockFreeDaemonServer>) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "tools": [
            {
                "name": "create_session",
                "description": "Create a new conversation session with optional name and description",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Optional name for the session"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description for the session"
                        }
                    }
                }
            }
        ]
    }))
}

async fn handle_tool_call(
    server: &Arc<LockFreeDaemonServer>,
    request: &MCPRequest,
) -> Result<serde_json::Value, String> {
    let params = request.params.as_ref()
        .ok_or_else(|| "Missing params in tool call".to_string())?;

    let tool_name = params["name"].as_str()
        .ok_or_else(|| "Missing tool name".to_string())?;

    let arguments = &params["arguments"];

    debug!("Tool call: {} with args: {:?}", tool_name, arguments);

    match tool_name {
        "create_session" => handle_create_session(server, arguments).await,
        _ => Err(format!("Unknown tool: {}", tool_name)),
    }
}

async fn handle_create_session(
    server: &Arc<LockFreeDaemonServer>,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let name = arguments["name"].as_str().map(|s| s.to_string());
    let description = arguments["description"].as_str().map(|s| s.to_string());

    let session_id = server
        .memory_system
        .create_session(name, description)
        .await
        .map_err(|e| format!("Failed to create session: {}", e))?;

    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": format!("Created new session: {}", session_id)
        }]
    }))
}

/// Error type for HTTP handlers
struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("HTTP error: {}", self.0);
        (StatusCode::INTERNAL_SERVER_ERROR, self.0).into_response()
    }
}

impl<E> From<E> for AppError
where
    E: std::error::Error,
{
    fn from(err: E) -> Self {
        AppError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_daemon_server_creation() {
        let config = DaemonConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // Random port
            data_directory: tempfile::tempdir().unwrap().path().to_str().unwrap().to_string(),
        };

        let server = LockFreeDaemonServer::new(config).await;
        assert!(server.is_ok());

        let server = server.unwrap();
        assert_eq!(server.active_connections.len(), 0);
        assert_eq!(server.total_requests.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_server_statistics() {
        let config = DaemonConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            data_directory: tempfile::tempdir().unwrap().path().to_str().unwrap().to_string(),
        };

        let server = LockFreeDaemonServer::new(config).await.unwrap();

        let stats = server.get_statistics();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.workspace_count, 0);
    }
}
