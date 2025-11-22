use anyhow::Result;
use post_cortex::core::lockfree_embeddings::{EmbeddingConfig, LockFreeLocalEmbeddingEngine};

#[tokio::test]
async fn test_identical_text_similarity() -> Result<()> {
    // Test that identical texts produce highly similar embeddings (>95%)

    let config = EmbeddingConfig::default();
    let engine = LockFreeLocalEmbeddingEngine::new(config).await?;

    let text = "How does vectorization work with BERT embeddings?";

    // Generate embedding twice for the same text
    let embedding1 = engine.encode_text(text).await?;
    let embedding2 = engine.encode_text(text).await?;

    // Calculate cosine similarity manually
    let dot_product: f32 = embedding1
        .iter()
        .zip(embedding2.iter())
        .map(|(x, y)| x * y)
        .sum();
    let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    };

    println!("Embedding 1 norm: {}", norm1);
    println!("Embedding 2 norm: {}", norm2);
    println!("Dot product: {}", dot_product);
    println!("Cosine similarity: {}", similarity);
    println!("First 10 dims of embedding1: {:?}", &embedding1[0..10]);
    println!("First 10 dims of embedding2: {:?}", &embedding2[0..10]);

    // For identical text, similarity should be very high (>0.99)
    // If L2 normalized properly, norms should be ~1.0
    assert!(
        norm1 > 0.99 && norm1 < 1.01,
        "Embedding 1 should be L2 normalized (norm ≈ 1.0), got {}",
        norm1
    );
    assert!(
        norm2 > 0.99 && norm2 < 1.01,
        "Embedding 2 should be L2 normalized (norm ≈ 1.0), got {}",
        norm2
    );
    assert!(
        similarity > 0.95,
        "Identical text should have >95% similarity, got {}",
        similarity
    );

    Ok(())
}

#[tokio::test]
async fn test_similar_text_similarity() -> Result<()> {
    // Test that similar texts produce reasonable similarity scores

    let config = EmbeddingConfig::default();
    let engine = LockFreeLocalEmbeddingEngine::new(config).await?;

    let text1 = "How does vectorization work with BERT embeddings?";
    let text2 = "How does vectorization work with BERT embeddings? Post-Cortex uses local models.";

    let embedding1 = engine.encode_text(text1).await?;
    let embedding2 = engine.encode_text(text2).await?;

    // Calculate cosine similarity
    let dot_product: f32 = embedding1
        .iter()
        .zip(embedding2.iter())
        .map(|(x, y)| x * y)
        .sum();
    let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    };

    println!("Text 1: {}", text1);
    println!("Text 2: {}", text2);
    println!("Embedding 1 norm: {}", norm1);
    println!("Embedding 2 norm: {}", norm2);
    println!("Cosine similarity: {}", similarity);

    // Similar texts should have decent similarity (>0.7 for overlapping content)
    // NOT 5.3% as reported in the bug!
    assert!(
        similarity > 0.7,
        "Similar text should have >70% similarity, got {}. Bug reproduced!",
        similarity
    );

    Ok(())
}
#[tokio::test]
async fn test_problematic_text_similarity() -> Result<()> {
    // Test the problematic text that gives 11% similarity in daemon

    let config = EmbeddingConfig::default();
    let engine = LockFreeLocalEmbeddingEngine::new(config).await?;

    let query = "How does the lock-free conversation memory system work in Post-Cortex?";
    let answer = "The lock-free conversation memory system in Post-Cortex is built using Rust's advanced concurrency primitives to achieve zero-deadlock guarantees. It uses DashMap for concurrent hash maps, AtomicU64 for counters, ArcSwap for atomic pointer swapping, and an actor pattern for asynchronous RocksDB storage operations. The system implements a three-tier memory hierarchy with hot cache for 50 most recent items, warm cache for 200 items, and cold storage in RocksDB for everything else. Entity relationships are tracked using Petgraph, and semantic search is powered by local BERT models generating 384-dimensional L2-normalized embeddings stored in an HNSW index for fast nearest neighbor search. The entire architecture is designed to be production-grade with high throughput and low latency even under heavy concurrent load.";

    let embedding_query = engine.encode_text(query).await?;
    let embedding_answer = engine.encode_text(answer).await?;

    // Calculate cosine similarity
    let dot_product: f32 = embedding_query
        .iter()
        .zip(embedding_answer.iter())
        .map(|(x, y)| x * y)
        .sum();
    let norm_query: f32 = embedding_query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_answer: f32 = embedding_answer.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = if norm_query == 0.0 || norm_answer == 0.0 {
        0.0
    } else {
        dot_product / (norm_query * norm_answer)
    };

    println!("Query: {}", query);
    println!("Answer: {}...", &answer[..100]);
    println!("Query embedding norm: {}", norm_query);
    println!("Answer embedding norm: {}", norm_answer);
    println!("Cosine similarity: {}", similarity);
    println!("Similarity percentage: {:.2}%", similarity * 100.0);

    Ok(())
}

#[tokio::test]
async fn test_end_to_end_semantic_search_pipeline() -> Result<()> {
    // End-to-end test: Create session, add QA update, search semantically
    // This tests the FULL pipeline including text extraction and vectorization

    use post_cortex::core::lockfree_memory_system::{
        LockFreeConversationMemorySystem, SystemConfig,
    };
    use std::sync::Arc;
    use tempfile::tempdir;

    // Create temporary directory for test data
    let temp_dir = tempdir()?;

    let mut config = SystemConfig::default();
    config.data_directory = temp_dir.path().to_str().unwrap().to_string();

    // Initialize memory system
    let system = Arc::new(
        LockFreeConversationMemorySystem::new(config)
            .await
            .map_err(|e| anyhow::anyhow!(e))?,
    );

    // Create test session
    let session_id = system
        .create_session(
            Some("test-session".to_string()),
            Some("End-to-end test session".to_string()),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Add QA update with question and answer combined in description
    let question = "How does the lock-free conversation memory system work in Post-Cortex?";
    let answer = "The lock-free conversation memory system in Post-Cortex is built using Rust's advanced concurrency primitives to achieve zero-deadlock guarantees. It uses DashMap for concurrent hash maps, AtomicU64 for counters, ArcSwap for atomic pointer swapping, and an actor pattern for asynchronous RocksDB storage operations.";

    // Combine question and answer for the update
    let full_text = format!("Q: {}\nA: {}", question, answer);

    system
        .add_incremental_update(session_id, full_text, None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Wait for vectorization to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Search with the question
    let results = system
        .semantic_search_session(session_id, question, Some(5), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    println!("\nEnd-to-end test results:");
    println!("Query: {}", question);
    println!("Found {} results", results.len());

    if !results.is_empty() {
        let top_result = &results[0];
        println!(
            "Top result similarity: {:.2}%",
            top_result.similarity_score * 100.0
        );
        println!(
            "Top result combined score: {:.2}%",
            top_result.combined_score * 100.0
        );
        println!("Top result content type: {:?}", top_result.content_type);

        // CRITICAL: Similarity should be HIGH (>70%) when searching with the exact question
        // This was the bug - before fix it would be ~11% because only answer was vectorized
        assert!(
            top_result.similarity_score > 0.70,
            "Semantic search should return high similarity (>70%) when searching with question that matches QA update. Got {:.2}%. This indicates the text extraction bug is not fixed!",
            top_result.similarity_score * 100.0
        );

        println!(
            "End-to-end test PASSED - similarity is {:.2}%",
            top_result.similarity_score * 100.0
        );
    } else {
        panic!("No results found! Vectorization or search failed.");
    }

    Ok(())
}
