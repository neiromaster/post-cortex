// Copyright (c) 2025 Julius ML
// MIT License

//! RMCP-based SSE Server for Post-Cortex daemon
//!
//! Provides SSE transport using rmcp library following official shuttle patterns.

use crate::ConversationMemorySystem;
use crate::daemon::{DaemonConfig, PostCortexService};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
};
use rmcp::transport::sse_server::{SseServer, SseServerConfig};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

/// Start RMCP-based SSE server
pub async fn start_rmcp_daemon(config: DaemonConfig) -> Result<(), String> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| format!("Invalid address: {}", e))?;

    info!("Initializing Post-Cortex daemon with RMCP SSE transport");
    info!("  Host: {}", config.host);
    info!("  Port: {}", config.port);
    info!("  Data Directory: {}", config.data_directory);

    // Create memory system with embeddings enabled
    let mut system_config = crate::SystemConfig {
        data_directory: config.data_directory.clone(),
        ..Default::default()
    };

    #[cfg(feature = "embeddings")]
    {
        system_config.enable_embeddings = true;
        system_config.embeddings_model_type = "MultilingualMiniLM".to_string();
        system_config.auto_vectorize_on_update = true;
        system_config.cross_session_search_enabled = true;
        info!("Embeddings enabled in daemon config");
    }

    // Configure storage backend if surrealdb-storage feature is enabled
    #[cfg(feature = "surrealdb-storage")]
    {
        use crate::storage::traits::StorageBackendType;

        system_config.storage_backend = match config.storage_backend.as_str() {
            "surrealdb" => StorageBackendType::SurrealDB,
            _ => StorageBackendType::RocksDB,
        };
        system_config.surrealdb_endpoint = config.surrealdb_endpoint.clone();
        system_config.surrealdb_username = config.surrealdb_username.clone();
        system_config.surrealdb_password = config.surrealdb_password.clone();

        if system_config.storage_backend == StorageBackendType::SurrealDB {
            info!(
                "Using SurrealDB storage backend: {}",
                system_config.surrealdb_endpoint.as_deref().unwrap_or("not configured")
            );
        } else {
            info!("Using RocksDB storage backend");
        }
    }

    let memory_system = Arc::new(
        ConversationMemorySystem::new(system_config)
            .await
            .map_err(|e| format!("Failed to initialize memory system: {}", e))?,
    );

    info!("Memory system initialized successfully");

    // Inject memory system into MCP tools so they use shared instance
    crate::tools::mcp::inject_memory_system(memory_system.clone());
    info!("Memory system injected into MCP tools");

    // Clear query cache to prevent stale vector IDs from previous runs
    if let Err(e) = memory_system.clear_query_cache().await {
        error!("Failed to clear query cache on startup: {}", e);
    } else {
        info!("Query cache cleared successfully on daemon startup");
    }

    // Create SSE server configuration
    let sse_config = SseServerConfig {
        bind: addr,
        sse_path: "/sse".to_string(),
        post_path: "/message".to_string(),
        ct: CancellationToken::new(),
        sse_keep_alive: Some(std::time::Duration::from_secs(30)),
    };

    info!("SSE endpoints configured:");
    info!("  SSE stream: http://{}/sse", addr);
    info!("  POST messages: http://{}/message", addr);

    // Create SSE server
    let (sse_server, sse_router) = SseServer::new(sse_config);

    // Create API state
    let api_state = Arc::new(ApiState {
        memory_system: memory_system.clone(),
    });

    // Create API router with its own state
    let api_router = Router::new()
        .route("/health", get(api_health))
        .route("/api/sessions", get(api_list_sessions).post(api_create_session))
        .route("/api/sessions/{id}", delete(api_delete_session))
        .route("/api/workspaces", get(api_list_workspaces).post(api_create_workspace))
        .route("/api/workspaces/{id}", delete(api_delete_workspace))
        .route("/api/workspaces/{workspace_id}/sessions/{session_id}", post(api_attach_session))
        .with_state(api_state);

    // Merge SSE router with API router
    let router = sse_router.merge(api_router);

    // Create TCP listener
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("Failed to bind to {}: {}", addr, e))?;

    info!("TCP listener bound to {}", addr);

    // Setup graceful shutdown
    let ct = sse_server.config.ct.child_token();
    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        ct.cancelled().await;
        info!("SSE server shutting down gracefully");
    });

    // Start HTTP server in background
    tokio::spawn(async move {
        if let Err(e) = server.await {
            error!("SSE server error: {}", e);
        }
    });

    // Register Post-Cortex service with SSE server
    let ct = sse_server.with_service(move || PostCortexService::new(memory_system.clone()));

    info!("Post-Cortex MCP service registered with SSE server");
    info!("Daemon is ready to accept connections");
    info!("Press Ctrl+C to shutdown");

    // Wait for shutdown signal
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, initiating shutdown");
        }
        _ = ct.cancelled() => {
            info!("Service cancelled");
        }
    }

    ct.cancel();
    info!("Post-Cortex daemon stopped");
    Ok(())
}

// ============================================================================
// REST API for CLI
// ============================================================================

/// Shared state for API handlers
struct ApiState {
    memory_system: Arc<ConversationMemorySystem>,
}

#[derive(Serialize)]
struct SessionInfo {
    id: String,
    name: String,
    workspace: Option<String>,
}

#[derive(Serialize)]
struct WorkspaceInfo {
    id: String,
    name: String,
    description: String,
    session_count: usize,
}

#[derive(Deserialize)]
struct CreateSessionRequest {
    name: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize)]
struct CreateWorkspaceRequest {
    name: String,
    description: Option<String>,
}

#[derive(Deserialize)]
struct AttachSessionRequest {
    role: Option<String>,
}

async fn api_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "post-cortex"
    }))
}

async fn api_list_sessions(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<Vec<SessionInfo>>, (StatusCode, String)> {
    let ids = state
        .memory_system
        .list_sessions()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let workspaces = state.memory_system.workspace_manager.list_workspaces();
    let mut session_workspace_map = std::collections::HashMap::new();
    for ws in workspaces {
        for (session_id, _role) in ws.get_all_sessions() {
            session_workspace_map.insert(session_id, ws.name.clone());
        }
    }

    let mut sessions = Vec::new();
    for id in ids {
        let name = match state.memory_system.get_session(id).await {
            Ok(session_arc) => {
                let session = session_arc.load();
                session.name().unwrap_or_else(|| "Unnamed".to_string())
            }
            Err(_) => "Error loading".to_string(),
        };

        sessions.push(SessionInfo {
            id: id.to_string(),
            name,
            workspace: session_workspace_map.get(&id).cloned(),
        });
    }

    Ok(Json(sessions))
}

async fn api_create_session(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<SessionInfo>, (StatusCode, String)> {
    let id = state
        .memory_system
        .create_session(req.name.clone(), req.description)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(SessionInfo {
        id: id.to_string(),
        name: req.name.unwrap_or_else(|| "Unnamed".to_string()),
        workspace: None,
    }))
}

async fn api_delete_session(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {}", e)))?;

    state
        .memory_system
        .get_storage()
        .delete_session(uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(StatusCode::NO_CONTENT)
}

async fn api_list_workspaces(
    State(state): State<Arc<ApiState>>,
) -> Result<Json<Vec<WorkspaceInfo>>, (StatusCode, String)> {
    let workspaces = state
        .memory_system
        .get_storage()
        .list_all_workspaces()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // Get list of existing sessions to filter out deleted ones
    let existing_sessions: std::collections::HashSet<_> = state
        .memory_system
        .list_sessions()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?
        .into_iter()
        .collect();

    let result: Vec<WorkspaceInfo> = workspaces
        .into_iter()
        .map(|ws| {
            // Count only sessions that still exist
            let actual_count = ws.sessions.iter()
                .filter(|(id, _)| existing_sessions.contains(id))
                .count();
            WorkspaceInfo {
                id: ws.id.to_string(),
                name: ws.name,
                description: ws.description,
                session_count: actual_count,
            }
        })
        .collect();

    Ok(Json(result))
}

async fn api_create_workspace(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<CreateWorkspaceRequest>,
) -> Result<Json<WorkspaceInfo>, (StatusCode, String)> {
    let id = Uuid::new_v4();
    let description = req.description.unwrap_or_default();

    state
        .memory_system
        .get_storage()
        .save_workspace_metadata(id, &req.name, &description, &[])
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(WorkspaceInfo {
        id: id.to_string(),
        name: req.name,
        description,
        session_count: 0,
    }))
}

async fn api_delete_workspace(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {}", e)))?;

    state
        .memory_system
        .get_storage()
        .delete_workspace(uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(StatusCode::NO_CONTENT)
}

async fn api_attach_session(
    State(state): State<Arc<ApiState>>,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(req): Json<AttachSessionRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let ws_id = Uuid::parse_str(&workspace_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid workspace UUID: {}", e)))?;
    let sess_id = Uuid::parse_str(&session_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid session UUID: {}", e)))?;

    let role = match req.role.as_deref().unwrap_or("related") {
        "primary" => crate::workspace::SessionRole::Primary,
        "related" => crate::workspace::SessionRole::Related,
        "dependency" => crate::workspace::SessionRole::Dependency,
        "shared" => crate::workspace::SessionRole::Shared,
        other => return Err((StatusCode::BAD_REQUEST, format!("Invalid role: {}", other))),
    };

    state
        .memory_system
        .get_storage()
        .add_session_to_workspace(ws_id, sess_id, role)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(StatusCode::OK)
}
