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

//! Post-Cortex daemon binary for manual HTTP server management and admin tasks
//!
//! Usage:
//!   post-cortex-daemon init     - Create example config file
//!   post-cortex-daemon start    - Start daemon server (foreground)
//!   post-cortex-daemon status   - Check if daemon is running
//!   post-cortex-daemon stop     - Stop running daemon
//!   post-cortex-daemon workspace - Manage workspaces
//!   post-cortex-daemon session   - Manage sessions

const VERSION: &str = "0.1.5";
const BUILD_DATE: &str = match option_env!("BUILD_DATE") {
    Some(date) => date,
    None => "dev-build",
};

use post_cortex::daemon::{DaemonConfig, start_rmcp_daemon};
use post_cortex::workspace::SessionRole;
use post_cortex::{LockFreeConversationMemorySystem, SystemConfig};
use std::env;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), String> {
    // Create logs directory in ~/.post-cortex/logs
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".post-cortex/logs");
    std::fs::create_dir_all(&log_dir).ok();

    // Initialize tracing with file appender
    let file_appender = RollingFileAppender::new(Rotation::DAILY, log_dir, "mcp-server.log");

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(file_appender))
        .with(fmt::layer()) // Also log to stdout
        .with(EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    match args[1].as_str() {
        "start" => start_daemon().await,
        "status" => check_status().await,
        "stop" => stop_daemon().await,
        "init" => init_config(),
        "vectorize-all" => vectorize_all().await,
        "workspace" => handle_workspace_command(&args[2..]).await,
        "session" => handle_session_command(&args[2..]).await,
        "version" | "--version" | "-v" => {
            print_version();
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        unknown => {
            eprintln!("Unknown command: {}", unknown);
            print_usage();
            Err(format!("Unknown command: {}", unknown))
        }
    }
}

fn print_version() {
    println!("Post-Cortex Daemon v{}", VERSION);
    println!("Build date: {}", BUILD_DATE);
}

fn print_usage() {
    println!("Post-Cortex Daemon - Lock-free HTTP MCP server");
    println!("Version: {}", VERSION);
    println!();
    println!("USAGE:");
    println!("    post-cortex-daemon <COMMAND>");
    println!();
    println!("COMMANDS:");
    println!("    start          Start daemon server in foreground");
    println!("    status         Check if daemon is running");
    println!("    stop           Stop running daemon (sends SIGTERM to port 3737)");
    println!("    init           Create example config file at ~/.post-cortex/daemon.toml");
    println!(
        "    vectorize-all  Vectorize all sessions in the system (requires embeddings feature)"
    );
    println!("    workspace      Manage workspaces (create, delete, list, attach)");
    println!("    session        Manage sessions (create, delete, list)");
    println!("    version        Print version information");
    println!("    help           Print this help message");
    println!();
    println!("CONFIGURATION:");
    println!("    Config File:    ~/.post-cortex/daemon.toml (optional)");
    println!("    Defaults:");
    println!("      Host:           127.0.0.1 (localhost only)");
    println!("      Port:           3737");
    println!("      Data Directory: ~/.post-cortex/data");
}

async fn start_daemon() -> Result<(), String> {
    println!("Starting Post-Cortex daemon with RMCP SSE transport...");
    println!("Version: {} (built: {})", VERSION, BUILD_DATE);
    println!();

    // Load configuration (priority: env vars > config file > defaults)
    let config = DaemonConfig::load();

    // Validate configuration
    config.validate()?;

    // Start RMCP-based SSE daemon (blocks until shutdown)
    start_rmcp_daemon(config).await
}

async fn check_status() -> Result<(), String> {
    let config = DaemonConfig::load();

    println!(
        "Checking daemon status at {}:{}...",
        config.host, config.port
    );

    // Check if port is listening by attempting a TCP connection
    use tokio::net::TcpStream;
    use tokio::time::{Duration, timeout};

    let addr = format!("{}:{}", config.host, config.port);

    match timeout(Duration::from_secs(2), TcpStream::connect(&addr)).await {
        Ok(Ok(_)) => {
            println!("✓ Daemon is running");
            println!("  Status: OK");
            println!("  SSE endpoint: http://{}:{}/sse", config.host, config.port);
            println!(
                "  POST endpoint: http://{}:{}/message",
                config.host, config.port
            );
            Ok(())
        }
        Ok(Err(e)) => {
            println!("✗ Daemon is not running");
            println!("  Error: Cannot connect to port {} ({})", config.port, e);
            Err("Daemon not running".to_string())
        }
        Err(_) => {
            println!("✗ Daemon is not running");
            println!("  Error: Connection timeout");
            Err("Daemon not running".to_string())
        }
    }
}

async fn stop_daemon() -> Result<(), String> {
    let config = DaemonConfig::load();

    println!("Stopping daemon at {}:{}...", config.host, config.port);
    println!();
    println!("Note: This will send SIGTERM signal");
    println!(
        "If daemon doesn't stop, use: kill $(lsof -t -i:{}))",
        config.port
    );

    // Try graceful shutdown via /shutdown endpoint (if we add it)
    // For now, just provide instructions
    println!();
    println!("To stop the daemon:");
    println!("1. Press Ctrl+C in the terminal where daemon is running");
    println!("2. Or use: kill $(lsof -t -i:{})", config.port);
    println!("3. Or use: pkill -f post-cortex-daemon");

    Ok(())
}

fn init_config() -> Result<(), String> {
    println!("Creating example config file...");
    println!();

    match DaemonConfig::create_example_config() {
        Ok(path) => {
            println!("Success! Config file created at:");
            println!("  {:?}", path);
            println!();
            println!("You can now edit this file to customize daemon settings.");
            println!("Environment variables will override config file values.");
            println!();
            println!("To start the daemon:");
            println!("  post-cortex-daemon start");
            Ok(())
        }
        Err(e) => {
            eprintln!("Failed to create config file: {}", e);
            Err(e)
        }
    }
}

// --- Admin Commands ---

async fn init_admin_system() -> Result<LockFreeConversationMemorySystem, String> {
    let daemon_config = DaemonConfig::load();
    let config = SystemConfig {
        enable_embeddings: false, // Admin tasks don't need embeddings
        data_directory: daemon_config.data_directory,
        ..SystemConfig::default()
    };
    LockFreeConversationMemorySystem::new(config).await
}

async fn handle_workspace_command(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_workspace_usage();
        return Err("Missing workspace subcommand".to_string());
    }

    match args[0].as_str() {
        "create" => {
            if args.len() < 2 {
                println!("Usage: workspace create <name> [description]");
                return Err("Missing workspace name".to_string());
            }
            let name = args[1].clone();
            let description = args.get(2).cloned().unwrap_or_default();

            let system = init_admin_system().await?;
            let id = Uuid::new_v4();

            // Use storage actor directly for persistence
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
        "delete" => {
            if args.len() < 2 {
                println!("Usage: workspace delete <id>");
                return Err("Missing workspace ID".to_string());
            }
            let id = Uuid::parse_str(&args[1]).map_err(|e| format!("Invalid UUID: {}", e))?;

            let system = init_admin_system().await?;
            system.get_storage().delete_workspace(id).await?;
            println!("Workspace {} deleted", id);
            Ok(())
        }
        "list" => {
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
        "attach" => {
            if args.len() < 3 {
                println!("Usage: workspace attach <workspace_id> <session_id> [role]");
                println!("Roles: primary, related, dependency, shared (default: related)");
                return Err("Missing arguments".to_string());
            }
            let ws_id =
                Uuid::parse_str(&args[1]).map_err(|e| format!("Invalid workspace UUID: {}", e))?;
            let session_id =
                Uuid::parse_str(&args[2]).map_err(|e| format!("Invalid session UUID: {}", e))?;

            let role_str = args.get(3).map(|s| s.as_str()).unwrap_or("related");
            let role = match role_str.to_lowercase().as_str() {
                "primary" => SessionRole::Primary,
                "related" => SessionRole::Related,
                "dependency" => SessionRole::Dependency,
                "shared" => SessionRole::Shared,
                _ => {
                    return Err(format!(
                        "Invalid role: {}. Allowed: primary, related, dependency, shared",
                        role_str
                    ));
                }
            };

            let system = init_admin_system().await?;
            system
                .get_storage()
                .add_session_to_workspace(ws_id, session_id, role)
                .await?;

            println!(
                "Session {} attached to workspace {} as {:?}",
                session_id, ws_id, role
            );
            Ok(())
        }
        "help" => {
            print_workspace_usage();
            Ok(())
        }
        _ => {
            print_workspace_usage();
            Err(format!("Unknown workspace command: {}", args[0]))
        }
    }
}

async fn handle_session_command(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_session_usage();
        return Err("Missing session subcommand".to_string());
    }

    match args[0].as_str() {
        "create" => {
            let name = args.get(1).cloned();
            let description = args.get(2).cloned();

            let system = init_admin_system().await?;
            let id = system
                .create_session(name.clone(), description.clone())
                .await?;

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
        "delete" => {
            if args.len() < 2 {
                println!("Usage: session delete <id>");
                return Err("Missing session ID".to_string());
            }
            let id = Uuid::parse_str(&args[1]).map_err(|e| format!("Invalid UUID: {}", e))?;

            let system = init_admin_system().await?;
            // LockFreeConversationMemorySystem doesn't have delete_session, use storage directly
            system.get_storage().delete_session(id).await?;
            println!("Session {} deleted", id);
            Ok(())
        }
        "list" => {
            let system = init_admin_system().await?;
            let ids = system.list_sessions().await?;

            println!("Sessions ({})", ids.len());
            for id in ids {
                // Try to get details if possible, but list_sessions only returns IDs.
                // To show names we'd need to load each session which is slow.
                // For CLI list, just IDs is standard unless we implement list_sessions_with_details
                println!("{}", id);
            }
            Ok(())
        }
        "help" => {
            print_session_usage();
            Ok(())
        }
        _ => {
            print_session_usage();
            Err(format!("Unknown session command: {}", args[0]))
        }
    }
}

fn print_workspace_usage() {
    println!("Usage: post-cortex-daemon workspace <COMMAND>");
    println!();
    println!("Commands:");
    println!("  create <name> [desc]               Create a new workspace");
    println!("  delete <id>                        Delete a workspace");
    println!("  list                               List all workspaces");
    println!("  attach <ws_id> <sess_id> [role]    Attach session to workspace");
    println!("                                     Roles: primary, related, dependency, shared");
}

fn print_session_usage() {
    println!("Usage: post-cortex-daemon session <COMMAND>");
    println!();
    println!("Commands:");
    println!("  create [name] [desc]    Create a new session");
    println!("  delete <id>             Delete a session");
    println!("  list                    List all sessions");
}

#[cfg(feature = "embeddings")]
async fn vectorize_all() -> Result<(), String> {
    use post_cortex::daemon::DaemonConfig;

    println!("Starting vectorization of all sessions...");
    println!();

    // Load daemon config to get data directory
    let daemon_config = DaemonConfig::load();
    println!("Data directory: {}", daemon_config.data_directory);
    println!();

    // Create memory system config with embeddings enabled
    let config = SystemConfig {
        enable_embeddings: true,
        auto_vectorize_on_update: false, // Disable auto-vectorize during bulk operation
        data_directory: daemon_config.data_directory,
        ..SystemConfig::default()
    };

    // Initialize memory system
    println!("Initializing memory system...");
    let system = match LockFreeConversationMemorySystem::new(config).await {
        Ok(sys) => sys,
        Err(e) => {
            eprintln!("Failed to initialize memory system: {}", e);
            return Err(e);
        }
    };
    println!("Memory system initialized successfully");
    println!();

    // Run vectorization
    println!("Starting bulk vectorization...");
    println!("This may take a while depending on the number of sessions and their size");
    println!();

    match system.vectorize_all_sessions().await {
        Ok((total_items, successful, failed)) => {
            println!();
            println!("Vectorization Complete!");
            println!("======================");
            println!("Total items vectorized: {}", total_items);
            println!("Successful sessions:    {}", successful);
            println!("Failed sessions:        {}", failed);
            println!();

            if failed > 0 {
                println!("Some sessions failed to vectorize. Check logs for details.");
                println!();
            }

            Ok(())
        }
        Err(e) => {
            eprintln!();
            eprintln!("Vectorization failed: {}", e);
            eprintln!();
            Err(e)
        }
    }
}

#[cfg(not(feature = "embeddings"))]
async fn vectorize_all() -> Result<(), String> {
    eprintln!("Error: Vectorization requires the 'embeddings' feature");
    eprintln!();
    eprintln!("Please rebuild with:");
    eprintln!("  cargo build --release --features embeddings");
    eprintln!();
    Err("Embeddings feature not enabled".to_string())
}
