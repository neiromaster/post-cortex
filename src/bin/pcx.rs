// Copyright (c) 2025 Julius ML
// MIT License

//! pcx - Post-Cortex unified MCP server
//!
//! A single binary supporting both stdio and SSE transports with auto-daemon management.
//!
//! Usage:
//!   pcx                    - Start in stdio mode (auto-starts daemon if needed)
//!   pcx start              - Start daemon in foreground (SSE mode)
//!   pcx start --daemon     - Start daemon in background
//!   pcx status             - Check daemon status
//!   pcx stop               - Stop running daemon
//!   pcx workspace          - Manage workspaces
//!   pcx session            - Manage sessions

const VERSION: &str = env!("CARGO_PKG_VERSION");

use clap::{Parser, Subcommand};
use post_cortex::daemon::{run_stdio_proxy, start_rmcp_daemon, DaemonConfig};
use post_cortex::workspace::SessionRole;
use post_cortex::{LockFreeConversationMemorySystem, SystemConfig};
use std::env;
use tracing::{error, info};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "pcx")]
#[command(version = VERSION)]
#[command(about = "Post-Cortex - Intelligent conversation memory system")]
#[command(long_about = "Post-Cortex unified MCP server supporting stdio and SSE transports.\n\n\
    When run without arguments, starts in stdio mode (for MCP clients).\n\
    The daemon is auto-started in background if not already running.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP server (daemon mode)
    Start {
        /// Run as background daemon (used internally for auto-start)
        #[arg(long)]
        daemon: bool,

        /// Port to listen on (default: 3737)
        #[arg(long, short, default_value = "3737")]
        port: u16,

        /// Host to bind to (default: 127.0.0.1)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Check if daemon is running
    Status,

    /// Stop running daemon
    Stop,

    /// Initialize configuration file
    Init,

    /// Vectorize all sessions
    VectorizeAll,

    /// Manage workspaces
    Workspace {
        #[command(subcommand)]
        action: WorkspaceAction,
    },

    /// Manage sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
}

#[derive(Subcommand)]
enum WorkspaceAction {
    /// Create a new workspace
    Create {
        /// Workspace name
        name: String,
        /// Workspace description
        #[arg(default_value = "")]
        description: String,
    },
    /// Delete a workspace
    Delete {
        /// Workspace ID (UUID)
        id: String,
    },
    /// List all workspaces
    List,
    /// Attach a session to a workspace
    Attach {
        /// Workspace ID (UUID)
        workspace_id: String,
        /// Session ID (UUID)
        session_id: String,
        /// Role: primary, related, dependency, shared
        #[arg(default_value = "related")]
        role: String,
    },
}

#[derive(Subcommand)]
enum SessionAction {
    /// Create a new session
    Create {
        /// Session name
        name: Option<String>,
        /// Session description
        description: Option<String>,
    },
    /// Delete a session
    Delete {
        /// Session ID (UUID)
        id: String,
    },
    /// List all sessions
    List,
}

fn init_logging(to_file: bool) {
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".post-cortex/logs");
    std::fs::create_dir_all(&log_dir).ok();

    if to_file {
        let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, "pcx.log");
        tracing_subscriber::registry()
            .with(fmt::layer().with_writer(file_appender))
            .with(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(fmt::layer().with_writer(std::io::stderr))
            .with(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
            .init();
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let cli = Cli::parse();

    match cli.command {
        // No command = stdio proxy mode (default for MCP clients)
        None => {
            // Minimal logging for stdio mode (to stderr only)
            init_logging(false);

            let config = DaemonConfig::load();
            run_stdio_proxy(config).await
        }

        Some(Commands::Start { daemon, port, host }) => {
            let mut config = DaemonConfig::load();
            config.port = port;
            config.host = host;

            if daemon {
                // Background daemon mode - log to file only
                init_logging(true);
                info!("Starting pcx daemon in background mode");
            } else {
                // Foreground mode - log to both
                init_logging(false);
                println!("Starting Post-Cortex daemon...");
                println!("Version: {}", VERSION);
                println!();
            }

            start_rmcp_daemon(config).await
        }

        Some(Commands::Status) => {
            let config = DaemonConfig::load();
            check_status(&config).await
        }

        Some(Commands::Stop) => {
            let config = DaemonConfig::load();
            stop_daemon(&config)
        }

        Some(Commands::Init) => {
            init_config()
        }

        Some(Commands::VectorizeAll) => {
            init_logging(false);
            vectorize_all().await
        }

        Some(Commands::Workspace { action }) => {
            init_logging(false);
            handle_workspace_action(action).await
        }

        Some(Commands::Session { action }) => {
            init_logging(false);
            handle_session_action(action).await
        }
    }
}

async fn check_status(config: &DaemonConfig) -> Result<(), String> {
    use tokio::net::TcpStream;
    use tokio::time::{timeout, Duration};

    println!("Checking daemon status at {}:{}...", config.host, config.port);

    let addr = format!("{}:{}", config.host, config.port);

    match timeout(Duration::from_secs(2), TcpStream::connect(&addr)).await {
        Ok(Ok(_)) => {
            println!("Daemon is running");
            println!("  SSE endpoint: http://{}:{}/sse", config.host, config.port);
            println!("  POST endpoint: http://{}:{}/message", config.host, config.port);
            Ok(())
        }
        Ok(Err(e)) => {
            println!("Daemon is not running");
            println!("  Error: {}", e);
            Err("Daemon not running".to_string())
        }
        Err(_) => {
            println!("Daemon is not running");
            println!("  Error: Connection timeout");
            Err("Daemon not running".to_string())
        }
    }
}

fn stop_daemon(config: &DaemonConfig) -> Result<(), String> {
    println!("Stopping daemon at {}:{}...", config.host, config.port);
    println!();
    println!("To stop the daemon:");
    println!("1. Press Ctrl+C in the terminal where daemon is running");
    println!("2. Or use: kill $(lsof -t -i:{})", config.port);
    println!("3. Or use: pkill -f 'pcx start'");
    Ok(())
}

fn init_config() -> Result<(), String> {
    println!("Creating example config file...");

    match DaemonConfig::create_example_config() {
        Ok(path) => {
            println!("Config file created at: {:?}", path);
            println!();
            println!("To start the daemon:");
            println!("  pcx start");
            Ok(())
        }
        Err(e) => {
            eprintln!("Failed to create config file: {}", e);
            Err(e)
        }
    }
}

async fn init_admin_system() -> Result<LockFreeConversationMemorySystem, String> {
    let daemon_config = DaemonConfig::load();
    let config = SystemConfig {
        enable_embeddings: false,
        data_directory: daemon_config.data_directory,
        ..SystemConfig::default()
    };
    LockFreeConversationMemorySystem::new(config).await
}

async fn handle_workspace_action(action: WorkspaceAction) -> Result<(), String> {
    match action {
        WorkspaceAction::Create { name, description } => {
            let system = init_admin_system().await?;
            let id = Uuid::new_v4();

            system
                .get_storage()
                .save_workspace_metadata(id, &name, &description, &[])
                .await?;

            println!("Workspace created:");
            println!("  ID:          {}", id);
            println!("  Name:        {}", name);
            println!("  Description: {}", description);
            Ok(())
        }

        WorkspaceAction::Delete { id } => {
            let id = Uuid::parse_str(&id).map_err(|e| format!("Invalid UUID: {}", e))?;
            let system = init_admin_system().await?;
            system.get_storage().delete_workspace(id).await?;
            println!("Workspace {} deleted", id);
            Ok(())
        }

        WorkspaceAction::List => {
            let system = init_admin_system().await?;
            let workspaces = system.get_storage().list_all_workspaces().await?;

            println!("Workspaces ({})", workspaces.len());
            println!("{:<38} {:<20} {}", "ID", "Name", "Sessions");
            println!("{:-<38} {:-<20} {:-<10}", "", "", "");

            for ws in workspaces {
                println!("{:<38} {:<20} {}", ws.id, ws.name, ws.sessions.len());
            }
            Ok(())
        }

        WorkspaceAction::Attach {
            workspace_id,
            session_id,
            role,
        } => {
            let ws_id = Uuid::parse_str(&workspace_id)
                .map_err(|e| format!("Invalid workspace UUID: {}", e))?;
            let sess_id =
                Uuid::parse_str(&session_id).map_err(|e| format!("Invalid session UUID: {}", e))?;

            let role = match role.to_lowercase().as_str() {
                "primary" => SessionRole::Primary,
                "related" => SessionRole::Related,
                "dependency" => SessionRole::Dependency,
                "shared" => SessionRole::Shared,
                _ => return Err(format!("Invalid role: {}. Use: primary, related, dependency, shared", role)),
            };

            let system = init_admin_system().await?;
            system
                .get_storage()
                .add_session_to_workspace(ws_id, sess_id, role)
                .await?;

            println!("Session {} attached to workspace {} as {:?}", sess_id, ws_id, role);
            Ok(())
        }
    }
}

async fn handle_session_action(action: SessionAction) -> Result<(), String> {
    match action {
        SessionAction::Create { name, description } => {
            let system = init_admin_system().await?;
            let id = system.create_session(name.clone(), description.clone()).await?;

            println!("Session created:");
            println!("  ID:          {}", id);
            if let Some(n) = name {
                println!("  Name:        {}", n);
            }
            if let Some(d) = description {
                println!("  Description: {}", d);
            }
            Ok(())
        }

        SessionAction::Delete { id } => {
            let id = Uuid::parse_str(&id).map_err(|e| format!("Invalid UUID: {}", e))?;
            let system = init_admin_system().await?;
            system.get_storage().delete_session(id).await?;
            println!("Session {} deleted", id);
            Ok(())
        }

        SessionAction::List => {
            let system = init_admin_system().await?;
            let ids = system.list_sessions().await?;

            let workspaces = system.workspace_manager.list_workspaces();
            let mut session_workspace_map = std::collections::HashMap::new();

            for ws in workspaces {
                for (session_id, _role) in ws.get_all_sessions() {
                    session_workspace_map.insert(session_id, ws.name.clone());
                }
            }

            println!("Sessions ({})", ids.len());
            println!("{:<38} {:<20} {}", "ID", "Workspace", "Name");
            println!("{:-<38} {:-<20} {:-<30}", "", "", "");

            for id in ids {
                let workspace_name = session_workspace_map
                    .get(&id)
                    .map(|s| s.as_str())
                    .unwrap_or("-");

                let session_name = match system.get_session(id).await {
                    Ok(session_arc) => {
                        let session = session_arc.load();
                        session.name().unwrap_or("Unnamed Session".to_string())
                    }
                    Err(_) => "Error loading".to_string(),
                };

                println!("{:<38} {:<20} {}", id, workspace_name, session_name);
            }
            Ok(())
        }
    }
}

#[cfg(feature = "embeddings")]
async fn vectorize_all() -> Result<(), String> {
    println!("Starting vectorization of all sessions...");

    let daemon_config = DaemonConfig::load();
    let config = SystemConfig {
        enable_embeddings: true,
        auto_vectorize_on_update: false,
        data_directory: daemon_config.data_directory,
        ..SystemConfig::default()
    };

    let system = LockFreeConversationMemorySystem::new(config)
        .await
        .map_err(|e| format!("Failed to initialize: {}", e))?;

    match system.vectorize_all_sessions().await {
        Ok((total, successful, failed)) => {
            println!("Vectorization complete!");
            println!("  Total items:  {}", total);
            println!("  Successful:   {}", successful);
            println!("  Failed:       {}", failed);
            Ok(())
        }
        Err(e) => {
            error!("Vectorization failed: {}", e);
            Err(e)
        }
    }
}

#[cfg(not(feature = "embeddings"))]
async fn vectorize_all() -> Result<(), String> {
    eprintln!("Vectorization requires the 'embeddings' feature");
    eprintln!("Rebuild with: cargo build --release --features embeddings");
    Err("Embeddings feature not enabled".to_string())
}
