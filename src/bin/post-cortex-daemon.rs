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

//! Post-Cortex daemon binary for manual HTTP server management
//!
//! Usage:
//!   post-cortex-daemon init     - Create example config file
//!   post-cortex-daemon start    - Start daemon server (foreground)
//!   post-cortex-daemon status   - Check if daemon is running
//!   post-cortex-daemon stop     - Stop running daemon

const VERSION: &str = "0.1.4";
const BUILD_DATE: &str = match option_env!("BUILD_DATE") {
    Some(date) => date,
    None => "dev-build",
};

use post_cortex::daemon::{DaemonConfig, start_rmcp_daemon};
use std::env;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

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
    println!("    version        Print version information");
    println!("    help           Print this help message");
    println!();
    println!("CONFIGURATION:");
    println!("    Config File:    ~/.post-cortex/daemon.toml (optional)");
    println!("    Defaults:");
    println!("      Host:           127.0.0.1 (localhost only)");
    println!("      Port:           3737");
    println!("      Data Directory: ~/.post-cortex/data");
    println!();
    println!("    Priority: Environment variables > Config file > Defaults");
    println!();
    println!("ENVIRONMENT VARIABLES:");
    println!("    RUST_LOG        Set logging level (e.g., RUST_LOG=debug)");
    println!("    PC_HOST         Override host");
    println!("    PC_PORT         Override port");
    println!("    PC_DATA_DIR     Override data directory");
    println!();
    println!("EXAMPLES:");
    println!("    # Create config file");
    println!("    post-cortex-daemon init");
    println!();
    println!("    # Start daemon with debug logging");
    println!("    RUST_LOG=debug post-cortex-daemon start");
    println!();
    println!("    # Check daemon status");
    println!("    post-cortex-daemon status");
    println!();
    println!("    # Vectorize all sessions");
    println!("    post-cortex-daemon vectorize-all");
    println!();
    println!("    # Start daemon with environment override");
    println!("    PC_PORT=8080 post-cortex-daemon start");
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

#[cfg(feature = "embeddings")]
async fn vectorize_all() -> Result<(), String> {
    use post_cortex::SystemConfig;
    use post_cortex::core::lockfree_memory_system::LockFreeConversationMemorySystem;
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
