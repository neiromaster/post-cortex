// Copyright (c) 2025 Julius ML
// MIT License

//! stdio-to-daemon proxy with auto-start capability
//!
//! This module provides a stdio interface that proxies MCP requests to a running daemon.
//! If no daemon is running, it automatically starts one in the background.

use crate::daemon::DaemonConfig;
use std::io::{self, BufRead};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};
use tokio::time::sleep;
use tracing::{error, info};

const MAX_STARTUP_WAIT_SECS: u64 = 30;
const STARTUP_CHECK_INTERVAL_MS: u64 = 100;

/// Check if daemon is running by attempting TCP connection
pub async fn is_daemon_running(host: &str, port: u16) -> bool {
    let addr = format!("{}:{}", host, port);
    TcpStream::connect(&addr).await.is_ok()
}

/// Start daemon in background process
pub fn start_daemon_background(config: &DaemonConfig) -> Result<(), String> {
    let current_exe =
        std::env::current_exe().map_err(|e| format!("Failed to get current executable: {}", e))?;

    info!("Starting daemon in background: {:?}", current_exe);

    let child = Command::new(&current_exe)
        .args(["start", "--daemon"])
        .env("PCX_PORT", config.port.to_string())
        .env("PCX_HOST", &config.host)
        .env("PCX_DATA_DIR", &config.data_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn daemon: {}", e))?;

    info!("Daemon process spawned with PID: {}", child.id());
    Ok(())
}

/// Wait for daemon to become ready
pub async fn wait_for_daemon(host: &str, port: u16) -> Result<(), String> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(MAX_STARTUP_WAIT_SECS);

    while start.elapsed() < timeout {
        if is_daemon_running(host, port).await {
            info!("Daemon is ready on {}:{}", host, port);
            return Ok(());
        }
        sleep(Duration::from_millis(STARTUP_CHECK_INTERVAL_MS)).await;
    }

    Err(format!(
        "Daemon failed to start within {} seconds",
        MAX_STARTUP_WAIT_SECS
    ))
}

/// Ensure daemon is running, starting it if necessary
pub async fn ensure_daemon_running(config: &DaemonConfig) -> Result<(), String> {
    if is_daemon_running(&config.host, config.port).await {
        info!("Daemon already running on {}:{}", config.host, config.port);
        return Ok(());
    }

    info!("Daemon not running, starting in background...");
    start_daemon_background(config)?;
    wait_for_daemon(&config.host, config.port).await
}

/// Run stdio proxy mode - bridges stdin/stdout to daemon via HTTP SSE
pub async fn run_stdio_proxy(config: DaemonConfig) -> Result<(), String> {
    // Ensure daemon is running
    ensure_daemon_running(&config).await?;

    let base_url = format!("http://{}:{}", config.host, config.port);
    let client = reqwest::Client::new();

    let sse_url = format!("{}/sse", base_url);

    info!("Connecting to daemon at {}", sse_url);

    // Channels for communication (all lock-free)
    let (stdout_tx, mut stdout_rx) = mpsc::unbounded_channel::<String>();
    let (session_tx, session_rx) = oneshot::channel::<String>();

    // SSE reader task - receives responses and session ID
    let sse_client = client.clone();
    let sse_url_clone = sse_url.clone();
    let mut session_tx = Some(session_tx);

    let _sse_handle = tokio::spawn(async move {
        match sse_client.get(&sse_url_clone).send().await {
            Ok(response) => {
                use futures::StreamExt;
                let mut stream = response.bytes_stream();
                let mut buffer = String::new();

                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            buffer.push_str(&text);

                            // Parse SSE events
                            while let Some(pos) = buffer.find("\n\n") {
                                let event = buffer[..pos].to_string();
                                buffer = buffer[pos + 2..].to_string();

                                // Extract session ID from endpoint event
                                if event.contains("event: endpoint") {
                                    if let Some(data_line) =
                                        event.lines().find(|l| l.starts_with("data: "))
                                    {
                                        let endpoint = data_line.trim_start_matches("data: ");
                                        if let Some(sid) = endpoint.split("sessionId=").nth(1) {
                                            let sid = sid.trim().to_string();
                                            info!("Got session ID: {}", sid);

                                            // Notify stdin reader (lock-free oneshot channel)
                                            if let Some(tx) = session_tx.take() {
                                                let _ = tx.send(sid);
                                            }
                                        }
                                    }
                                }
                                // Forward message events to stdout
                                else if event.contains("event: message") {
                                    if let Some(data_line) =
                                        event.lines().find(|l| l.starts_with("data: "))
                                    {
                                        let data = data_line.trim_start_matches("data: ");
                                        if stdout_tx.send(data.to_string()).is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("SSE stream error: {}", e);
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to connect to SSE: {}", e);
            }
        }
    });

    // Stdout writer task
    let _stdout_handle = tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        let mut stdout = tokio::io::stdout();
        while let Some(msg) = stdout_rx.recv().await {
            let line = format!("{}\n", msg);
            if stdout.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            if stdout.flush().await.is_err() {
                break;
            }
        }
    });

    // Wait for session ID before processing stdin
    let session_id_str = session_rx
        .await
        .map_err(|_| "Failed to get session ID from SSE".to_string())?;

    let message_url = format!("{}/message?sessionId={}", base_url, session_id_str);
    info!("Ready to proxy requests to {}", message_url);

    // Stdin reader - send requests to daemon
    let stdin_handle = tokio::task::spawn_blocking(move || {
        let stdin = io::stdin();
        let reader = stdin.lock();
        let rt = tokio::runtime::Handle::current();

        for line in reader.lines() {
            match line {
                Ok(line) if !line.trim().is_empty() => {
                    let client = client.clone();
                    let url = message_url.clone();

                    rt.block_on(async {
                        match client
                            .post(&url)
                            .header("Content-Type", "application/json")
                            .body(line)
                            .send()
                            .await
                        {
                            Ok(resp) => {
                                if !resp.status().is_success() {
                                    error!("Request failed with status: {}", resp.status());
                                }
                            }
                            Err(e) => {
                                error!("Failed to send request: {}", e);
                            }
                        }
                    });
                }
                Ok(_) => {} // Empty line, skip
                Err(e) => {
                    error!("Error reading stdin: {}", e);
                    break;
                }
            }
        }
    });

    // Wait for stdin to close (client disconnected)
    let _ = stdin_handle.await;

    Ok(())
}
