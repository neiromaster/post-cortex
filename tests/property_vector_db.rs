// Property-based tests for VectorDB using proptest
//
// These tests verify key invariants of the lock-free vector database:
// 1. Dimension consistency - vectors must match configured dimension
// 2. Search bounds - search returns at most k results
// 3. Similarity range - cosine similarity is always in [-1.0, 1.0]
// 4. Idempotency - adding vectors doesn't corrupt state
// 5. Remove consistency - removed vectors don't appear in searches
// 6. Stats accuracy - statistics match actual operations
// 7. Concurrent safety - operations are thread-safe

use post_cortex::core::vector_db::{FastVectorDB, VectorDbConfig, VectorMetadata};
use proptest::prelude::*;

// Strategy to generate valid vector dimensions
fn dimension_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![Just(128), Just(256), Just(384), Just(512), Just(768)]
}

// Strategy to generate normalized vectors of given dimension
fn vector_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0f32, dim).prop_map(|v| {
        // Normalize vector to unit length
        let magnitude = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            v.iter().map(|x| x / magnitude).collect()
        } else {
            v
        }
    })
}

// Strategy to generate vector metadata
fn metadata_strategy() -> impl Strategy<Value = VectorMetadata> {
    (
        prop::string::string_regex("[a-z]{5,10}").unwrap(),
        prop::string::string_regex("[a-z ]{10,50}").unwrap(),
        prop::string::string_regex("session-[0-9]{1,5}").unwrap(),
        prop::string::string_regex("(qa|decision|problem|code)").unwrap(),
    )
        .prop_map(|(id, text, source, content_type)| {
            VectorMetadata::new(id, text, source, content_type)
        })
}

// Strategy to generate search limit k
fn search_k_strategy() -> impl Strategy<Value = usize> {
    1usize..=20
}

proptest! {
    /// Property: Vectors must match configured dimension
    #[test]
    fn prop_dimension_validation(
        (dim, wrong_dim) in (dimension_strategy(), dimension_strategy())
            .prop_filter("Dimensions must be different", |(d1, d2)| d1 != d2)
    ) {
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Correct dimension should succeed
        let correct_vector = vec![0.5f32; dim];
        let metadata = VectorMetadata::new(
            "test-1".to_string(),
            "test text".to_string(),
            "test-source".to_string(),
            "qa".to_string(),
        );
        let result = db.add_vector(correct_vector, metadata);
        prop_assert!(result.is_ok());

        // Wrong dimension should fail
        let wrong_vector = vec![0.5f32; wrong_dim];
        let metadata2 = VectorMetadata::new(
            "test-2".to_string(),
            "test text 2".to_string(),
            "test-source".to_string(),
            "qa".to_string(),
        );
        let result = db.add_vector(wrong_vector, metadata2);
        prop_assert!(result.is_err());
    }

    /// Property: Search returns at most k results
    #[test]
    fn prop_search_bounds(
        k in search_k_strategy(),
        num_vectors in 1usize..=50,
    ) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors
        for i in 0..num_vectors {
            let vector = vec![(i as f32) / num_vectors as f32; dim];
            let metadata = VectorMetadata::new(
                format!("vec-{}", i),
                format!("text {}", i),
                "test-source".to_string(),
                "qa".to_string(),
            );
            db.add_vector(vector, metadata).unwrap();
        }

        // Search with limit k
        let query = vec![0.5f32; dim];
        let results = db.search(&query, k).unwrap();

        // Results should not exceed k
        prop_assert!(results.len() <= k);

        // If we have fewer than k vectors, results should match vector count
        if num_vectors < k {
            prop_assert_eq!(results.len(), num_vectors);
        }
    }

    /// Property: Cosine similarity is always in valid range
    #[test]
    fn prop_similarity_range(
        dim in dimension_strategy(),
        vectors in prop::collection::vec(vector_strategy(384), 1..=20),
    ) {
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors (only if dimension matches)
        if dim == 384 {
            for (i, vector) in vectors.iter().enumerate() {
                let metadata = VectorMetadata::new(
                    format!("vec-{}", i),
                    format!("text {}", i),
                    "test-source".to_string(),
                    "qa".to_string(),
                );
                db.add_vector(vector.clone(), metadata).unwrap();
            }

            // Search with arbitrary query
            let query = vec![0.5f32; dim];
            let results = db.search(&query, 10).unwrap();

            // All similarities must be in [-1.0, 1.0]
            for result in results {
                prop_assert!(result.similarity >= -1.0);
                prop_assert!(result.similarity <= 1.0);
            }
        }
    }

    /// Property: Adding vectors is idempotent (can add multiple times)
    #[test]
    fn prop_add_idempotency(
        num_vectors in 1usize..=20,
        metadata in metadata_strategy(),
    ) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors multiple times
        for _ in 0..num_vectors {
            let vector = vec![0.5f32; dim];
            let result = db.add_vector(vector, metadata.clone());
            prop_assert!(result.is_ok());
        }

        // Stats should reflect total additions
        let stats = db.get_stats();
        prop_assert_eq!(stats.total_vectors, num_vectors);
    }

    /// Property: Removed vectors don't appear in search results
    #[test]
    fn prop_remove_consistency(
        num_vectors in 2usize..=20,
        remove_idx in 0usize..=10,
    ) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors and track IDs
        let mut vector_ids = Vec::new();
        for i in 0..num_vectors {
            let vector = vec![(i as f32) / num_vectors as f32; dim];
            let metadata = VectorMetadata::new(
                format!("vec-{}", i),
                format!("text {}", i),
                "test-source".to_string(),
                "qa".to_string(),
            );
            let id = db.add_vector(vector, metadata).unwrap();
            vector_ids.push(id);
        }

        // Remove a vector if index is valid
        let remove_idx = remove_idx % num_vectors;
        let removed_id = vector_ids[remove_idx];
        let removed = db.remove_vector(removed_id).unwrap();
        prop_assert!(removed);

        // Search should not return removed vector
        let query = vec![0.5f32; dim];
        let results = db.search(&query, num_vectors).unwrap();

        for result in results {
            prop_assert_ne!(result.vector_id, removed_id);
        }

        // Stats should reflect removal
        let stats = db.get_stats();
        prop_assert_eq!(stats.total_vectors, num_vectors - 1);
    }

    /// Property: Statistics are accurate
    #[test]
    fn prop_stats_accuracy(
        num_add in 0usize..=30,
        num_remove in 0usize..=15,
    ) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors
        let mut vector_ids = Vec::new();
        for i in 0..num_add {
            let vector = vec![0.5f32; dim];
            let metadata = VectorMetadata::new(
                format!("vec-{}", i),
                format!("text {}", i),
                "test-source".to_string(),
                "qa".to_string(),
            );
            let id = db.add_vector(vector, metadata).unwrap();
            vector_ids.push(id);
        }

        // Remove some vectors
        let actual_removes = num_remove.min(num_add);
        for i in 0..actual_removes {
            db.remove_vector(vector_ids[i]).unwrap();
        }

        // Check stats
        let stats = db.get_stats();
        let expected_count = num_add - actual_removes;
        prop_assert_eq!(stats.total_vectors, expected_count);
    }

    /// Property: Clear removes all vectors
    #[test]
    fn prop_clear_completeness(num_vectors in 1usize..=30) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors
        for i in 0..num_vectors {
            let vector = vec![0.5f32; dim];
            let metadata = VectorMetadata::new(
                format!("vec-{}", i),
                format!("text {}", i),
                "test-source".to_string(),
                "qa".to_string(),
            );
            db.add_vector(vector, metadata).unwrap();
        }

        // Clear database
        db.clear().unwrap();

        // Stats should show zero vectors
        let stats = db.get_stats();
        prop_assert_eq!(stats.total_vectors, 0);

        // Search should return no results
        let query = vec![0.5f32; dim];
        let results = db.search(&query, 10).unwrap();
        prop_assert_eq!(results.len(), 0);
    }

    /// Property: Search returns results in descending similarity order
    #[test]
    fn prop_search_ordering(num_vectors in 2usize..=20) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vectors with different similarities to query
        for i in 0..num_vectors {
            let mut vector = vec![0.0f32; dim];
            // Create vectors with varying dot products with [1, 0, 0, ...]
            vector[0] = (i as f32) / (num_vectors as f32);
            vector[1] = 1.0 - vector[0];

            let metadata = VectorMetadata::new(
                format!("vec-{}", i),
                format!("text {}", i),
                "test-source".to_string(),
                "qa".to_string(),
            );
            db.add_vector(vector, metadata).unwrap();
        }

        // Search
        let query = vec![1.0f32, 0.0f32].into_iter().chain(vec![0.0f32; dim - 2]).collect::<Vec<_>>();
        let results = db.search(&query, num_vectors).unwrap();

        // Verify descending order
        for i in 0..results.len().saturating_sub(1) {
            prop_assert!(results[i].similarity >= results[i + 1].similarity);
        }
    }

    /// Property: Identical vectors have similarity close to 1.0
    #[test]
    fn prop_identical_vector_similarity(vector in vector_strategy(384)) {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = FastVectorDB::new(config).unwrap();

        // Add vector
        let metadata = VectorMetadata::new(
            "test".to_string(),
            "test text".to_string(),
            "test-source".to_string(),
            "qa".to_string(),
        );
        db.add_vector(vector.clone(), metadata).unwrap();

        // Search with same vector
        let results = db.search(&vector, 1).unwrap();

        // Should return exactly 1 result with similarity ~1.0
        prop_assert_eq!(results.len(), 1);
        prop_assert!((results[0].similarity - 1.0).abs() < 0.01);
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_concurrent_adds() {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = Arc::new(FastVectorDB::new(config).unwrap());

        let num_threads = 10;
        let vectors_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let db_clone = Arc::clone(&db);
                thread::spawn(move || {
                    for i in 0..vectors_per_thread {
                        let vector = vec![((thread_id * 10 + i) as f32) / 100.0; dim];
                        let metadata = VectorMetadata::new(
                            format!("thread-{}-vec-{}", thread_id, i),
                            format!("text {} {}", thread_id, i),
                            "test-source".to_string(),
                            "qa".to_string(),
                        );
                        db_clone.add_vector(vector, metadata).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all vectors were added
        let stats = db.get_stats();
        assert_eq!(stats.total_vectors, num_threads * vectors_per_thread);
    }

    #[test]
    fn test_concurrent_search() {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = Arc::new(FastVectorDB::new(config).unwrap());

        // Add some vectors first
        for i in 0..20 {
            let vector = vec![(i as f32) / 20.0; dim];
            let metadata = VectorMetadata::new(
                format!("vec-{}", i),
                format!("text {}", i),
                "test-source".to_string(),
                "qa".to_string(),
            );
            db.add_vector(vector, metadata).unwrap();
        }

        // Concurrent searches
        let num_threads = 20;
        let searches_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let db_clone = Arc::clone(&db);
                thread::spawn(move || {
                    for _ in 0..searches_per_thread {
                        let query = vec![(thread_id as f32) / num_threads as f32; dim];
                        let results = db_clone.search(&query, 5).unwrap();
                        assert!(results.len() <= 5);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify search operations completed successfully (avg_search_time_us > 0 indicates searches ran)
        let stats = db.get_stats();
        assert!(stats.avg_search_time_us > 0.0);
    }

    #[test]
    fn test_concurrent_add_remove() {
        let dim = 384;
        let config = VectorDbConfig {
            dimension: dim,
            enable_quantization: false,
            enable_hnsw_index: false,
            ..Default::default()
        };
        let db = Arc::new(FastVectorDB::new(config).unwrap());

        let num_threads = 10;
        let operations_per_thread = 10;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let db_clone = Arc::clone(&db);
                thread::spawn(move || {
                    let mut ids = Vec::new();

                    // Add vectors
                    for i in 0..operations_per_thread {
                        let vector = vec![((thread_id * 10 + i) as f32) / 100.0; dim];
                        let metadata = VectorMetadata::new(
                            format!("thread-{}-vec-{}", thread_id, i),
                            format!("text {} {}", thread_id, i),
                            "test-source".to_string(),
                            "qa".to_string(),
                        );
                        let id = db_clone.add_vector(vector, metadata).unwrap();
                        ids.push(id);
                    }

                    // Remove half
                    for i in 0..operations_per_thread / 2 {
                        db_clone.remove_vector(ids[i]).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final count
        let stats = db.get_stats();
        let expected = num_threads * operations_per_thread - num_threads * (operations_per_thread / 2);
        assert_eq!(stats.total_vectors, expected);
    }
}
