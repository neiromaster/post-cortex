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
async fn test_create_session_tool() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    // Call create_session tool
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "create_session",
            "arguments": {
                "name": "Test Session",
                "description": "Integration test session"
            }
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    let body: serde_json::Value = response.json().await.unwrap();

    // Verify response structure
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["id"], 1);
    assert!(body["result"].is_object());
    assert!(body["result"]["content"].is_array());

    // Verify session was created
    let text = body["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Created new session"));
    assert!(text.contains("-")); // UUID contains dashes
}

#[tokio::test]
async fn test_tools_list_includes_create_session() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
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
    assert!(body["result"]["tools"].is_array());

    let tools = body["result"]["tools"].as_array().unwrap();
    assert!(!tools.is_empty());

    let create_session_tool = tools.iter().find(|t| t["name"] == "create_session");

    assert!(create_session_tool.is_some());
    let tool = create_session_tool.unwrap();
    assert!(tool["description"].is_string());
    assert!(tool["inputSchema"].is_object());
}

#[tokio::test]
async fn test_concurrent_create_sessions() {
    let (port, _temp_dir) = start_test_daemon().await;

    // Create 10 sessions concurrently
    let tasks: Vec<_> = (0..10)
        .map(|i| {
            let port = port;
            tokio::spawn(async move {
                let client = Client::new();

                let request = json!({
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {
                        "name": "create_session",
                        "arguments": {
                            "name": format!("Session {}", i),
                            "description": format!("Concurrent test {}", i)
                        }
                    }
                });

                let response = client
                    .post(format!("http://127.0.0.1:{}/mcp", port))
                    .json(&request)
                    .send()
                    .await
                    .unwrap();

                assert!(response.status().is_success());

                let body: serde_json::Value = response.json().await.unwrap();
                assert!(body["result"].is_object());

                body["result"]["content"][0]["text"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
        })
        .collect();

    // Wait for all sessions to be created
    let results = futures::future::join_all(tasks).await;

    // Verify all succeeded
    assert_eq!(results.len(), 10);
    for result in results {
        let text = result.unwrap();
        assert!(text.contains("Created new session"));
    }
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
#[tokio::test]
async fn test_update_conversation_context_tool() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    // First create a session
    let create_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "create_session",
            "arguments": {
                "name": "Test Session",
                "description": "Testing update_conversation_context"
            }
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&create_request)
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    let session_text = body["result"]["content"][0]["text"].as_str().unwrap();
    let session_id = session_text
        .split("Created new session: ")
        .nth(1)
        .unwrap()
        .trim();

    // Now update context
    let update_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "update_conversation_context",
            "arguments": {
                "session_id": session_id,
                "interaction_type": "qa",
                "content": {
                    "question": "How does daemon mode work?",
                    "answer": "Daemon mode allows multiple Claude instances to share RocksDB"
                }
            }
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&update_request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["result"].is_object());
}

#[tokio::test]
async fn test_semantic_search_session_tool() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    // Create session and add context
    let create_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "create_session",
            "arguments": {}
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&create_request)
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    let session_text = body["result"]["content"][0]["text"].as_str().unwrap();
    let session_id = session_text
        .split("Created new session: ")
        .nth(1)
        .unwrap()
        .trim();

    // Search in session
    let search_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "semantic_search_session",
            "arguments": {
                "session_id": session_id,
                "query": "daemon mode",
                "limit": 10
            }
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&search_request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["result"].is_object());
}

#[tokio::test]
async fn test_list_sessions_tool() {
    let (port, _temp_dir) = start_test_daemon().await;
    let client = Client::new();

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "list_sessions",
            "arguments": {}
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["result"].is_object());
}
#[tokio::test]
async fn test_list_sessions_debug() {
    use post_cortex::daemon::{DaemonConfig, LockFreeDaemonServer};
    use reqwest::Client;
    use serde_json::json;
    use std::time::Duration;

    let temp_dir = tempfile::tempdir().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let config = DaemonConfig {
        host: "127.0.0.1".to_string(),
        port,
        data_directory: temp_dir.path().to_str().unwrap().to_string(),
    };

    let server = LockFreeDaemonServer::new(config).await.unwrap();
    tokio::spawn(async move {
        server.start().await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = Client::new();
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "list_sessions",
            "arguments": {}
        }
    });

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .json(&request)
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = response.json().await.unwrap();
    println!(
        "Response body: {}",
        serde_json::to_string_pretty(&body).unwrap()
    );
}
