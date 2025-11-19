// Copyright (c) 2025 Julius ML
// MIT License

//! RMCP-based SSE Server for Post-Cortex daemon
//!
//! Provides SSE transport using rmcp library following official shuttle patterns.

use crate::ConversationMemorySystem;
use crate::daemon::{DaemonConfig, PostCortexService};
use rmcp::transport::sse_server::{SseServer, SseServerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

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
        system_config.embeddings_model_type = "StaticSimilarityMRL".to_string();
        system_config.auto_vectorize_on_update = true;
        system_config.cross_session_search_enabled = true;
        info!("Embeddings enabled in daemon config");
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
    let (sse_server, router) = SseServer::new(sse_config);

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
