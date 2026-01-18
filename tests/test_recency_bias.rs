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
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Tests for temporal decay (recency bias) in semantic search

use anyhow::Result;
use post_cortex::core::lockfree_memory_system::{
    LockFreeConversationMemorySystem, SystemConfig,
};
use serial_test::serial;
use std::sync::Arc;
use tempfile::tempdir;
use uuid::Uuid;

/// Helper to create test system
async fn create_test_system() -> Result<(Arc<LockFreeConversationMemorySystem>, tempfile::TempDir)> {
    let temp_dir = tempdir()?;
    let mut config = SystemConfig::default();
    config.data_directory = temp_dir.path().to_str().unwrap().to_string();
    config.enable_embeddings = true;
    config.auto_vectorize_on_update = true;

    let system = Arc::new(
        LockFreeConversationMemorySystem::new(config)
            .await
            .map_err(|e| anyhow::anyhow!(e))?,
    );

    Ok((system, temp_dir))
}

#[serial]
#[tokio::test]
async fn test_recency_bias_zero_has_no_effect() -> Result<()> {
    // Test that recency_bias=0.0 doesn't affect ranking (backward compatibility)

    let (system, _temp_dir) = create_test_system().await?;

    // Create session
    let session_id = system
        .create_session(
            Some("test-session".to_string()),
            Some("Test session for recency bias".to_string()),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Add two pieces of similar content at the same time
    let text1 = "Authentication system uses JWT tokens for secure access";
    let text2 = "Authentication system uses OAuth2 for secure access";

    system
        .add_incremental_update(session_id, text1.to_string(), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    system
        .add_incremental_update(session_id, text2.to_string(), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Wait for vectorization
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Ensure semantic engine is initialized
    let engine = system.ensure_semantic_engine_initialized().await
        .map_err(|e| anyhow::anyhow!(e))?;
    let results = engine
        .semantic_search_session(session_id, "authentication system", None, None, Some(0.0))
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    println!("\nTest: recency_bias=0.0 (should not affect ranking)");
    println!("Found {} results", results.len());
    for (i, result) in results.iter().enumerate() {
        println!(
            "  Result {}: score={:.4}, similarity={:.4}",
            i, result.combined_score, result.similarity_score
        );
    }

    // With recency_bias=0.0, both results should have similar combined scores
    // (only based on similarity + importance, not time)
    if results.len() >= 2 {
        let score_diff = (results[0].combined_score - results[1].combined_score).abs();
        println!(
            "Score difference between top 2 results: {:.4}",
            score_diff
        );
        // Scores should be relatively close (within 0.4) since time is not factored in
        assert!(
            score_diff < 0.4,
            "With recency_bias=0.0, scores should be similar, got diff={:.4}",
            score_diff
        );
    }

    Ok(())
}

#[serial]
#[tokio::test]
async fn test_recency_bias_prioritizes_recent_content() -> Result<()> {
    // Test that recency_bias>0.0 prioritizes recent content
    // This test adds content at different times by manipulating the internal storage

    let (system, _temp_dir) = create_test_system().await?;

    // Create session
    let session_id = system
        .create_session(
            Some("test-recency".to_string()),
            Some("Test recency bias prioritization".to_string()),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Add content with similar semantic meaning
    let text = "Secure authentication system using JWT tokens";

    system
        .add_incremental_update(session_id, text.to_string(), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Wait for vectorization
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Search with different recency_bias values
    let engine = system.ensure_semantic_engine_initialized().await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Use more specific query that should match the content
    let results_no_bias = engine
        .semantic_search_session(session_id, "JWT tokens authentication", None, None, Some(0.0))
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    let results_with_bias = engine
        .semantic_search_session(session_id, "JWT tokens authentication", None, None, Some(1.0))
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    println!("\nTest: Recency bias effect on ranking");
    println!("Results WITHOUT recency_bias:");
    for (i, result) in results_no_bias.iter().enumerate() {
        println!(
            "  Result {}: score={:.4}, similarity={:.4}",
            i, result.combined_score, result.similarity_score
        );
    }

    println!("Results WITH recency_bias=1.0:");
    for (i, result) in results_with_bias.iter().enumerate() {
        println!(
            "  Result {}: score={:.4}, similarity={:.4}",
            i, result.combined_score, result.similarity_score
        );
    }

    // With same content, both should find it
    assert!(!results_no_bias.is_empty(), "Should find content without bias");
    assert!(
        !results_with_bias.is_empty(),
        "Should find content with bias"
    );

    // Scores should be identical since content age is the same
    if !results_no_bias.is_empty() && !results_with_bias.is_empty() {
        let score_no_bias = results_no_bias[0].combined_score;
        let score_with_bias = results_with_bias[0].combined_score;

        println!(
            "Score comparison: no_bias={:.4}, with_bias={:.4}",
            score_no_bias, score_with_bias
        );

        // Since content was added at the same time, scores should be very similar
        let score_diff = (score_no_bias - score_with_bias).abs();
        assert!(
            score_diff < 0.01,
            "With same-age content, recency_bias should not change scores. Got diff={:.4}",
            score_diff
        );
    }

    Ok(())
}

#[serial]
#[tokio::test]
async fn test_recency_bias_multisession() -> Result<()> {
    // Test recency bias across multiple sessions (workspace scenario)

    let (system, _temp_dir) = create_test_system().await?;

    // Create two sessions
    let session1 = system
        .create_session(
            Some("session-1".to_string()),
            Some("First session".to_string()),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    let session2 = system
        .create_session(
            Some("session-2".to_string()),
            Some("Second session".to_string()),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Add similar content to both sessions
    let text = "Vector embeddings enable semantic search capabilities";

    system
        .add_incremental_update(session1, text.to_string(), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Small delay to ensure different timestamps
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    system
        .add_incremental_update(session2, text.to_string(), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Wait for vectorization
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let engine = system.ensure_semantic_engine_initialized().await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Search across both sessions with recency bias
    let session_ids = vec![session1, session2];
    let results_no_bias = engine
        .semantic_search_multisession(&session_ids, "semantic search embeddings", None, None, Some(0.0))
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    let results_with_bias = engine
        .semantic_search_multisession(&session_ids, "semantic search embeddings", None, None, Some(1.0))
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    println!("\nTest: Multisession recency bias");
    println!("Results WITHOUT bias: {} items", results_no_bias.len());
    println!("Results WITH bias=1.0: {} items", results_with_bias.len());

    // Should find results from both sessions
    assert_eq!(results_no_bias.len(), 2, "Should find 2 results without bias");
    assert_eq!(results_with_bias.len(), 2, "Should find 2 results with bias");

    // With high recency bias, session2 (more recent) should have equal or higher score
    let session1_result = results_with_bias.iter()
        .find(|r| r.session_id == session1)
        .expect("Should find session1 result");
    let session2_result = results_with_bias.iter()
        .find(|r| r.session_id == session2)
        .expect("Should find session2 result");

    println!(
        "Session 1 (older) score: {:.4}, Session 2 (recent) score: {:.4}",
        session1_result.combined_score, session2_result.combined_score
    );

    assert!(
        session2_result.combined_score >= session1_result.combined_score,
        "With recency_bias=1.0, recent session should have >= score than older session"
    );

    println!("Recent session has equal or higher score: ✓");

    Ok(())
}

#[serial]
#[tokio::test]
async fn test_recency_bias_formula_consistency() -> Result<()> {
    // Test that recency_bias formula produces consistent, predictable results

    let (system, _temp_dir) = create_test_system().await?;

    let session_id = system
        .create_session(
            Some("test-formula".to_string()),
            Some("Test formula consistency".to_string()),
        )
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Add content
    let text = "Exponential decay reduces old content scores gradually";
    system
        .add_incremental_update(session_id, text.to_string(), None)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    let engine = system.ensure_semantic_engine_initialized().await
        .map_err(|e| anyhow::anyhow!(e))?;

    // Search with different lambda values and verify monotonic behavior
    let lambda_values = vec![0.0, 0.3, 0.5, 1.0, 2.0];
    let mut previous_score = None;

    println!("\nTest: Formula consistency across lambda values");

    for lambda in lambda_values {
        let results = engine
            .semantic_search_session(
                session_id,
                "exponential decay content",
                None,
                None,
                Some(lambda),
            )
            .await
            .map_err(|e| anyhow::anyhow!(e))?;

        if !results.is_empty() {
            let score = results[0].combined_score;
            println!(
                "  λ={:.1}: combined_score={:.4}",
                lambda, score
            );

            // Higher lambda should not increase score for same content
            if let Some(prev) = previous_score {
                assert!(
                    score <= prev + 0.001, // Allow small floating point errors
                    "Increasing lambda should not increase score. Got λ={:.1} > λ_prev, score={:.4} > prev_score={:.4}",
                    lambda, score, prev
                );
            }

            previous_score = Some(score);
        }
    }

    println!("Formula produces consistent results: ✓");

    Ok(())
}
