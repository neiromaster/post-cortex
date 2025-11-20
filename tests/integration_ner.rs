// Integration tests for NER engine
#![cfg(feature = "embeddings")]

use post_cortex::session::active_session::{ActiveSession, preload_ner_engine};
use uuid::Uuid;

#[tokio::test]
#[ignore] // Requires model download (~250MB), run manually with --ignored
async fn test_ner_integration_with_session() {
    // Pre-load NER engine
    println!("\n=== Pre-loading NER engine ===");
    let loaded = preload_ner_engine().await;
    assert!(loaded, "NER engine should load successfully");
    println!(" NER engine loaded successfully!\n");

    // Create a test session
    let session = ActiveSession::new(
        Uuid::new_v4(),
        Some("NER Test Session".to_string()),
        Some("Testing NER integration".to_string()),
    );

    // Test text with real-world examples relevant to post-cortex system
    let test_cases = vec![
        // Technical discussions
        "Claude Code uses RocksDB for persistent storage in San Francisco.",
        "Julius implemented the lock-free memory system using DashMap and Arc.",
        "The NER engine from Anthropic uses DistilBERT for entity extraction.",
        // Codebase references
        "ActiveSession stores data in PostgreSQL and Redis.",
        "John Smith from Google contributed to the Rust tokio runtime.",
        // Architecture discussions
        "Microsoft Azure and Amazon AWS both use distributed systems.",
        "Marie Curie discovered radium, similar to how we discovered the HNSW algorithm.",
        // Real project scenarios
        "The daemon runs on Linux servers in London and New York.",
        "Sarah Johnson from Netflix optimized the vector search in Los Angeles.",
    ];

    for test_text in test_cases {
        println!("--- Testing: {} ---", test_text);
        let entities = session.extract_entities_from_text(test_text);
        println!("Extracted {} entities: {:?}", entities.len(), entities);

        // Verify we got some entities (NER should find at least person/org/location)
        assert!(
            !entities.is_empty(),
            "NER should extract entities from: {}",
            test_text
        );
    }

    println!("\n NER integration test passed!");
}

#[tokio::test]
async fn test_ner_preload_function() {
    // Test that preload function works
    println!("Testing NER preload...");
    let result = preload_ner_engine().await;

    // Should either succeed (model loaded) or fail gracefully
    // Don't assert true because model might not be downloadable in CI
    println!("NER preload result: {}", result);

    if result {
        println!("NER engine preloaded successfully");
    } else {
        println!("NER engine failed to load (expected in CI without model files)");
    }
}
