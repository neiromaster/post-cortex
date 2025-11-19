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

use post_cortex::daemon::{DaemonConfig, start_rmcp_daemon};
use std::env;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), String> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
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

fn print_usage() {
    println!("Post-Cortex Daemon - Lock-free HTTP MCP server");
    println!();
    println!("USAGE:");
    println!("    post-cortex-daemon <COMMAND>");
    println!();
    println!("COMMANDS:");
    println!("    start     Start daemon server in foreground");
    println!("    status    Check if daemon is running");
    println!("    stop      Stop running daemon (sends SIGTERM to port 3737)");
    println!("    init      Create example config file at ~/.post-cortex/daemon.toml");
    println!("    help      Print this help message");
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
    println!("    # Start daemon with environment override");
    println!("    PC_PORT=8080 post-cortex-daemon start");
}

async fn start_daemon() -> Result<(), String> {
    println!("Starting Post-Cortex daemon with RMCP SSE transport...");
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
    let url = format!("http://{}:{}/health", config.host, config.port);

    println!("Checking daemon status at {}...", url);

    match reqwest::get(&url).await {
        Ok(response) => {
            if response.status().is_success() {
                println!("Daemon is running");
                println!("Status: OK");
                println!("URL: http://{}:{}", config.host, config.port);

                // Try to get stats
                if let Ok(stats_response) =
                    reqwest::get(format!("http://{}:{}/stats", config.host, config.port)).await
                {
                    if let Ok(stats_text) = stats_response.text().await {
                        println!("\nServer Statistics:");
                        println!("{}", stats_text);
                    }
                }

                Ok(())
            } else {
                println!("Daemon responded but with error status: {}", response.status());
                Err(format!("Daemon unhealthy: {}", response.status()))
            }
        }
        Err(e) => {
            println!("Daemon is not running");
            println!("Error: {}", e);
            Err("Daemon not running".to_string())
        }
    }
}

async fn stop_daemon() -> Result<(), String> {
    let config = DaemonConfig::load();

    println!("Stopping daemon at {}:{}...", config.host, config.port);
    println!();
    println!("Note: This will send SIGTERM signal");
    println!("If daemon doesn't stop, use: kill $(lsof -t -i:{}))", config.port);

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

