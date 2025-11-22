use post_cortex::{ConversationMemorySystem, SystemConfig};
use std::sync::Arc;
use uuid::Uuid;

async fn create_test_system() -> (Arc<ConversationMemorySystem>, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = SystemConfig {
        data_directory: temp_dir.path().to_str().unwrap().to_string(),
        enable_embeddings: true,
        auto_vectorize_on_update: true, // We need vectorization for search
        semantic_search_threshold: 0.1, // Low threshold for testing with basic embeddings
        ..Default::default()
    };
    let system = ConversationMemorySystem::new(config).await.unwrap();
    
    // Initialize embeddings engine (lazy loading)
    // We need to trigger it or ensure it's ready.
    // In tests, auto_vectorize might be slow or async.
    
    (Arc::new(system), temp_dir)
}

#[tokio::test]
async fn test_unified_search_scopes() {
    let (system_arc, _temp_dir) = create_test_system().await;
    let system = system_arc.as_ref();
    
    // Ensure semantic engine is initialized
    system.ensure_semantic_engine_initialized().await.unwrap();
    
    // 1. Create Workspaces
    let ws_alpha = system.workspace_manager.create_workspace("Alpha".to_string(), "".to_string());
    let ws_beta = system.workspace_manager.create_workspace("Beta".to_string(), "".to_string());
    
    // 2. Create Sessions
    let sess_a1 = system.create_session(None, None).await.unwrap(); // "Apple" in Alpha
    let sess_a2 = system.create_session(None, None).await.unwrap(); // "Banana" in Alpha
    let sess_b1 = system.create_session(None, None).await.unwrap(); // "Carrot" in Beta
    let sess_b2 = system.create_session(None, None).await.unwrap(); // "Apple" in Beta (conflict!)
    
    // 3. Assign to Workspaces
    system.workspace_manager.add_session_to_workspace(&ws_alpha, sess_a1, post_cortex::workspace::SessionRole::Primary).unwrap();
    system.workspace_manager.add_session_to_workspace(&ws_alpha, sess_a2, post_cortex::workspace::SessionRole::Related).unwrap();
    system.workspace_manager.add_session_to_workspace(&ws_beta, sess_b1, post_cortex::workspace::SessionRole::Primary).unwrap();
    system.workspace_manager.add_session_to_workspace(&ws_beta, sess_b2, post_cortex::workspace::SessionRole::Related).unwrap();
    
    // 4. Add Content (and wait for vectorization)
    // use post_cortex::core::context_update::{ContextUpdate, UpdateContent, UpdateType};
    
    async fn add_update(sys: &ConversationMemorySystem, sess: Uuid, text: &str) {
        let mut content = std::collections::HashMap::new();
        content.insert("text".to_string(), text.to_string());
        
        // Use the system-aware helper
        post_cortex::tools::mcp::update_conversation_context_with_system(
            "qa".to_string(), 
            content, 
            None, 
            sess,
            sys
        ).await.unwrap();
    }
    
    add_update(system, sess_a1, "The Apple is a red fruit.").await;
    add_update(system, sess_a2, "The Banana is a yellow fruit.").await;
    add_update(system, sess_b1, "The Carrot is an orange vegetable.").await;
    add_update(system, sess_b2, "Apple Inc. makes computers.").await; // "Apple" again
    
    // 5. Wait for vectorization (it's async in background usually)
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    
    // 6. Perform Searches
    
    // Helper for search
    let search = |query: &str, scope_type: &str, id: Option<Uuid>| {
        let sys = system_arc.clone();
        let q = query.to_string();
        let s_type = scope_type.to_string();
        async move {
            let engine = sys.semantic_query_engine.get().unwrap();
            if s_type == "workspace" {
                let ws = sys.workspace_manager.get_workspace(&id.unwrap()).unwrap();
                let s_ids: Vec<Uuid> = ws.get_all_sessions().into_iter().map(|(i, _)| i).collect();
                engine.semantic_search_multisession(&s_ids, &q, None, None).await.unwrap()
            } else if s_type == "session" {
                engine.semantic_search_session(id.unwrap(), &q, None, None).await.unwrap()
            } else {
                engine.semantic_search_global(&q, None, None).await.unwrap()
            }
        }
    };
    
    // Search "Apple" in Alpha -> Expect sess_a1 (Fruit)
    let res_alpha = search("Apple", "workspace", Some(ws_alpha)).await;
    println!("Alpha results: {:?}", res_alpha);
    assert!(res_alpha.iter().any(|r| r.session_id == sess_a1));
    assert!(!res_alpha.iter().any(|r| r.session_id == sess_b2)); // Should NOT find Beta's Apple
    
    // Search "Apple" in Beta -> Expect sess_b2 (Computer)
    let res_beta = search("Apple", "workspace", Some(ws_beta)).await;
    println!("Beta results: {:?}", res_beta);
    assert!(res_beta.iter().any(|r| r.session_id == sess_b2));
    assert!(!res_beta.iter().any(|r| r.session_id == sess_a1)); // Should NOT find Alpha's Apple
    
    // Search "Apple" Global -> Expect Both
    let res_global = search("Apple", "global", None).await;
    println!("Global results: {:?}", res_global);
    assert!(res_global.iter().any(|r| r.session_id == sess_a1));
    assert!(res_global.iter().any(|r| r.session_id == sess_b2));
    
    // Search "Banana" in Alpha -> Found
    let res_banana = search("Banana", "workspace", Some(ws_alpha)).await;
    assert!(!res_banana.is_empty());
    assert!(res_banana.iter().any(|r| r.session_id == sess_a2));
    
    // Search "Banana" in Beta -> Should NOT find the Banana from Alpha
    // It might find other things from Beta due to low threshold, but NOT sess_a2
    let res_banana_beta = search("Banana", "workspace", Some(ws_beta)).await;
    println!("Banana in Beta results: {:?}", res_banana_beta);
    assert!(!res_banana_beta.iter().any(|r| r.session_id == sess_a2));
}
