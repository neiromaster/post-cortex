// Integration tests for daemon HTTP server with real clients
use post_cortex::daemon::{DaemonConfig, LockFreeDaemonServer};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::timeout;

async fn start_test_daemon() -> (u16, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();

    // Find free port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = DaemonConfig {
        host: "127.0.0.1".to_string(),
        port,
        data_directory: temp_dir.path().to_str().unwrap().to_string(),
    };

    let server = LockFreeDaemonServer::new(config).await.unwrap();

    // Start server in background
    tokio::spawn(async move {
        server.start().await.unwrap();
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    (port, temp_dir)
}

#[tokio::test]
async fn test_daemon_health_check() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    let response = client
        .get(format!("http://127.0.0.1:{}/health", port))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "post-cortex-daemon");
}

#[tokio::test]
async fn test_daemon_stats_endpoint() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    let response = client
        .get(format!("http://127.0.0.1:{}/stats", port))
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["active_connections"].is_number());
    assert!(body["total_requests"].is_number());
    assert!(body["workspace_count"].is_number());
}

#[tokio::test]
async fn test_daemon_mcp_initialize() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["id"], 1);
    assert!(body["result"].is_object());
    assert!(body["result"]["serverInfo"].is_object());
}

#[tokio::test]
async fn test_multiple_concurrent_clients() {
    let (port, _temp_dir) = start_test_daemon().await;

    // Spawn 10 concurrent HTTP clients
    let tasks: Vec<_> = (0..10)
        .map(|i| {
            let port = port;
            tokio::spawn(async move {
                let client = Client::new();

                // Make health check request
                let health_response = client
                    .get(format!("http://127.0.0.1:{}/health", port))
                    .send()
                    .await
                    .unwrap();
                assert!(health_response.status().is_success());

                // Make MCP initialize request
                let mcp_request = json!({
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "initialize",
                    "params": {}
                });

                let mcp_response = client
                    .post(format!("http://127.0.0.1:{}/mcp", port))
                    .json(&mcp_request)
                    .send()
                    .await
                    .unwrap();

                assert!(mcp_response.status().is_success());
                let body: serde_json::Value = mcp_response.json().await.unwrap();
                assert_eq!(body["id"], i);
            })
        })
        .collect();

    // Wait for all clients - if deadlock, this will timeout
    for task in tasks {
        timeout(Duration::from_secs(5), task)
            .await
            .expect("Task timed out - possible deadlock")
            .unwrap();
    }

    // Verify stats increased
    let client = Client::new();
    let response = client
        .get(format!("http://127.0.0.1:{}/stats", port))
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["total_requests"].as_u64().unwrap() >= 10);
}

#[tokio::test]
async fn test_stress_concurrent_requests() {
    let (port, _temp_dir) = start_test_daemon().await;

    // Spawn 50 concurrent clients making multiple requests each
    let tasks: Vec<_> = (0..50)
        .map(|i| {
            let port = port;
            tokio::spawn(async move {
                let client = Client::new();

                // Each client makes 5 requests
                for j in 0..5 {
                    let request = json!({
                        "jsonrpc": "2.0",
                        "id": format!("{}-{}", i, j),
                        "method": "initialize",
                        "params": {}
                    });

                    let response = client
                        .post(format!("http://127.0.0.1:{}/mcp", port))
                        .json(&request)
                        .send()
                        .await
                        .unwrap();

                    assert!(response.status().is_success());
                }
            })
        })
        .collect();

    // Wait for all - 250 total requests
    for task in tasks {
        timeout(Duration::from_secs(10), task)
            .await
            .expect("Stress test timed out - possible deadlock")
            .unwrap();
    }

    // Verify total requests
    let client = Client::new();
    let response = client
        .get(format!("http://127.0.0.1:{}/stats", port))
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["total_requests"].as_u64().unwrap() >= 250);
}

#[tokio::test]
async fn test_daemon_shares_rocksdb() {
    let (port, temp_dir) = start_test_daemon().await;
    let client = Client::new();

    // Verify server is running
    let response = client
        .get(format!("http://127.0.0.1:{}/health", port))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());

    // Try to create second ConversationMemorySystem instance with same data dir
    // This should FAIL because daemon already has RocksDB open
    let config = post_cortex::SystemConfig {
        data_directory: temp_dir.path().to_str().unwrap().to_string(),
        ..Default::default()
    };

    let result = post_cortex::ConversationMemorySystem::new(config).await;

    // Should fail with lock error
    assert!(result.is_err());
    if let Err(error_msg) = result {
        let error_lower = error_msg.to_lowercase();
        assert!(
            error_lower.contains("lock") || error_lower.contains("io error"),
            "Expected lock error, got: {}",
            error_msg
        );
    }
}
