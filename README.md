# Post-Cortex

**Production-Grade Intelligent Memory System for AI Assistants**

Post-Cortex is a high-performance MCP (Model Context Protocol) server that transforms ephemeral AI conversations into persistent, searchable knowledge with zero-deadlock guarantees. Built in Rust with a sophisticated lock-free concurrent architecture, it enables AI assistants to maintain perfect memory across sessions, understand complex semantic relationships, and retrieve contextually relevant information through local AI-powered search—all while preserving complete privacy through on-device processing.

Unlike traditional conversation systems that forget everything when the session ends, Post-Cortex provides durable memory infrastructure with automatic knowledge graph construction, intelligent entity extraction, and semantic search powered by local transformer models. Every conversation becomes part of a growing, interconnected knowledge base that your AI assistant can query instantly using natural language.

## Core Features

- **Persistent Memory System**: Maintains conversation history across sessions with intelligent hierarchical storage
- **Semantic Search Engine**: Powered by local transformer models that enable context-aware retrieval based on meaning
- **Automatic Knowledge Graph**: Dynamically maps relationships between entities, decisions, and concepts without manual intervention
- **Privacy-First Architecture**: All processing occurs locally with zero external API dependencies
- **Lock-Free Concurrency**: Zero-deadlock guarantee through atomic operations and advanced concurrent data structures
- **Production-Ready Scaling**: Extensively tested with hundreds of conversations, thousands of entities, and tens of thousands of embeddings

## Applications

### For End Users
- **Cross-Session Memory**: AI assistants maintain context and recall information from previous conversations
- **Natural Language Queries**: Ask questions in natural language and receive relevant answers regardless of exact terminology used
- **Automatic Context Tracking**: Important concepts, decisions, and code changes are captured without manual intervention
- **IDE Integration**: Seamless integration with popular development environments including Claude Desktop and Zed editor

### For Developers
- **Robust Memory Infrastructure**: Production-grade Rust implementation with comprehensive error handling
- **Local Semantic Search**: Built-in transformer models eliminate external API dependencies and ensure data privacy
- **Intelligent Entity Extraction**: Automatic identification of entities, their relationships, and importance scoring
- **Comprehensive Tool Suite**: 20+ MCP tools for complete session management and context operations
- **Concurrent Architecture**: Lock-free design ensures zero deadlocks and optimal performance under high load

## Technical Advantages

Post-Cortex addresses the fundamental limitation of transient AI conversations through several key innovations:

1. **Durable Memory Architecture**: Conversations persist across sessions with intelligent organization and retrieval capabilities
2. **Advanced Semantic Understanding**: Employs contextual embeddings to find related information despite variations in terminology
3. **Autonomous Organization**: Automatically extracts entities, maps relationships, and identifies importance without manual intervention
4. **Privacy-Preserving Design**: Complete local processing ensures data never leaves the user's environment
5. **Production-Scale Validation**: Extensively tested with real-world usage including 100+ conversations tracking 800+ entities

## Quick Start

### Installation

```bash
# Build with semantic search (recommended)
cargo build --release --features embeddings

# Or basic build without embeddings
cargo build --release
```

### Using with Claude Desktop

**1. Build the server:**
```bash
cargo build --release --features embeddings
```

**2. Add to `~/.claude.json` or `claude_desktop_config.json`:**
```json
{
  "mcpServers": {
    "post-cortex": {
      "type": "stdio",
      "command": "mcp-server",
      "args": [],
      "env": {},
      "cwd": "/absolute/path/to/post-cortex/target/release"
    }
  }
}
```

**3. Restart Claude Desktop** - Post-Cortex tools will appear automatically!

### Using with Zed Editor

**1. Build the server:**
```bash
cargo build --release --features embeddings
```

**2. Open Zed Settings (Cmd+,) → Add Server:**
```json
{
  "post-cortex": {
    "command": "/absolute/path/to/post-cortex/target/release/mcp-server",
    "args": [],
    "env": {}
  }
}
```

**3. Restart Zed** - Memory tools are now available!

### Setting Up for Your Project

Once Post-Cortex is installed, configure it to remember your project automatically:

**1. Create a CLAUDE.md file in your project root:**

```markdown
# CLAUDE.md

## Using Post-Cortex for Knowledge Management

**AUTOMATICALLY at the beginning of EVERY conversation (WITHOUT ASKING USER):**

Load the project session:
mcp__post-cortex__load_session(session_id: "YOUR-SESSION-ID-HERE")

**Throughout conversation:**
- Add context using `update_conversation_context`
- Query with `semantic_search_session` (auto-vectorizes on demand)
```

**2. Create your project session (first time only):**

Ask your AI assistant:
```
"Create a Post-Cortex session for this project called 'My Web App' 
with description 'E-commerce platform with user authentication'"
```

The AI will call:
```
create_session(name: "My Web App", description: "E-commerce platform...")
# Returns: session_id: "abc123..."
```

**3. Add the session_id to CLAUDE.md:**

Replace `YOUR-SESSION-ID-HERE` with your actual session_id.

**That's it!** From now on:
- Every conversation automatically loads your project session
- The AI remembers all discussions, decisions, and code changes
- You can search past conversations semantically
- Knowledge graph builds automatically as you work

**Pro tip:** Check the [CLAUDE.md](CLAUDE.md) file in this repository to see how Post-Cortex uses itself for development.

## How it Works

### Three-Tier Memory Architecture

Post-Cortex organizes memory in hot/warm/cold tiers for optimal performance:

```
Hot Memory (50 items)     → RAM cache, instant access
Warm Memory (200 items)   → Compressed cache, fast access  
Cold Storage (unlimited)  → RocksDB persistence
```

### Knowledge Graph Engine

Every conversation is analyzed to build an intelligent knowledge graph:

**Automatic Entity Extraction:**
- Identifies key concepts, technologies, decisions, and problems
- Tracks mention frequency and recency
- Calculates importance scores based on connections

**Relationship Mapping:**
- Maps connections between entities: `RelatedTo`, `LeadsTo`, `Implements`, `Solves`
- Builds network graph with 1000+ relationships in active sessions
- Enables graph traversal queries like "what led to this decision?"

**Example from real session:**
```
388 entities tracked
1015 relationships mapped
Top entities: lock-free (44 mentions), session (42), search (38)
```

### Semantic Search with Local AI

Post-Cortex uses **local transformer models** for privacy-first semantic search:

**Embedding Models Available:**
- **MiniLM** (384-dim, default) - Balanced performance, uses `sentence-transformers/all-MiniLM-L6-v2`
- **StaticSimilarityMRL** (1024-dim) - Mac M-series optimized, fastest inference
- **TinyBERT** (312-dim) - Smallest BERT variant for low-memory environments
- **BGESmall** (384-dim) - High-quality embeddings for maximum accuracy

All models run completely locally using the Candle ML framework - no external API calls, guaranteed privacy.

**How Semantic Search Works:**
1. Text converted to 384-dimensional vector using local transformer model
2. Long texts (>500 tokens) intelligently summarized with priority-based extraction
3. Vector stored in HNSW index (Hierarchical Navigable Small World) for fast nearest-neighbor search
4. Cosine similarity computed between query vector and stored vectors
5. Results ranked by combined score: `(similarity × 0.7) + (importance × 0.3)`
6. Quality filtering applied (default threshold: similarity > 0.30)

**Quality Scoring (Combined Score Algorithm):**

Results ranked by: `combined_score = (similarity × 0.7) + (importance × 0.3)`

This balances semantic relevance with entity importance for optimal results.

**Quality Levels:**
- **0.85-1.00** Excellent - Exact semantic matches, high relevance
- **0.70-0.84** Very Good - Direct semantic matches
- **0.55-0.69** Good - Related concepts and ideas
- **0.40-0.54** Moderate - Tangentially related
- **0.30-0.39** Fair - Weak matches
- **< 0.30** Filtered out automatically

**Similarity Score Interpretation:**
- **0.75-1.00**: Exact matches with high relevance
- **0.65-0.75**: Direct semantic matches
- **0.45-0.55**: Good semantic matches
- **0.35-0.45**: Related concepts
- **< 0.30**: Weak matches (filtered by default)

**Two Search Modes:**
- **Keyword Search** - Fast exact matching (< 10ms), uses entity graph
- **Semantic Search** - AI-powered meaning search (50-200ms), uses embeddings

**Privacy-First:** All models run locally, no data sent to external APIs

## Available Tools (MCP)

Post-Cortex provides 20+ tools through the Model Context Protocol:

**Session Management:**
- `create_session` - Start new conversation memory
- `load_session` - Resume previous session
- `list_sessions` - View all saved sessions

**Adding Context:**
- `update_conversation_context` - Add Q&A, decisions, problems, code changes
- `bulk_update_conversation_context` - Batch updates

**Context Types (interaction_type):**
- **`qa`** - Questions and answers about architecture, implementation, or behavior
- **`decision_made`** - Architectural choices with rationale and trade-offs (includes confidence level)
- **`problem_solved`** - Bugs, issues, and their solutions with impact analysis
- **`code_change`** - Significant refactoring, feature additions, or API changes with file references

**Searching:**
- `semantic_search_session` - AI-powered meaning search
- `semantic_search_global` - Search across all sessions
- `query_conversation_context` - Fast keyword search

**Analysis:**
- `get_structured_summary` - Full session overview
- `get_key_decisions` - Timeline of important choices
- `get_entity_network_view` - Visualize concept relationships

[See full tool list](docs/MCP_TOOLS.md)

## See It In Action: A Quick Example

Let's say you're working on a web project and want your AI assistant to remember everything about it.

**Day 1 - First conversation:**

```
You: "I'm building a user authentication system with JWT tokens"
AI: "Got it, let me create a session for this project."

# Behind the scenes, Post-Cortex:
- Creates session: "Web App Authentication" 
- Stores entities: "authentication", "JWT tokens"
- Maps relationship: authentication → implements → JWT tokens
```

**Day 3 - Ask a question:**

```
You: "What approach did we decide for user login?"
AI searches Post-Cortex: semantic_search("user login approach")

# Finds your Day 1 conversation even though you said "authentication" not "login"
# Returns: "JWT tokens for authentication"
```

**Week 2 - Multiple projects:**

```
# You now have 3 sessions:
- "Web App Authentication" (your original project)
- "Mobile App Project" 
- "API Gateway Design"

You: "Show me what we've discussed about security"
AI: get_entity_network_view(session: "Web App Authentication", center: "security")

# Returns graph showing connections within this session:
security → relates to → authentication
security → relates to → password hashing
authentication → implements → JWT tokens
JWT tokens → leads to → refresh token strategy
```

**Month 2 - Search across ALL projects:**

```
You: "Did we discuss rate limiting anywhere?"
AI: semantic_search_global(query: "rate limiting")

# Searches across all 3 sessions and finds:
1. [API Gateway Design, Score: 0.78] "Implemented token bucket for rate limiting"
2. [Web App Authentication, Score: 0.52] "Considered rate limits for login attempts"

# Found relevant info from different projects!
```

**The magic:** You never manually tagged anything. Post-Cortex automatically:
- Extracted 20+ entities per session (authentication, JWT, security, passwords, API, database...)
- Built 50+ relationships between concepts
- Created 75 semantic embeddings for searching by meaning
- Remembered every decision with context ("Why did we choose JWT? Because...")
- **Isolated sessions** - Each project has its own knowledge graph
- **Cross-session search** - Find related info across all your projects

**Real-world growth:**

Post-Cortex is developed using itself. Current production stats from active development session:
- 122 conversation updates across multiple topics
- 898 technical concepts and entities tracked
- 1,015 relationships mapped in knowledge graph
- 10,000+ semantic embeddings indexed
- Ask "How did we optimize performance?" → Instantly finds answers with 0.85+ relevance scores
- Search "lock-free implementation" → Returns exact architectural decisions from weeks ago

**Start empty, grow organically.** No setup, no configuration, no manual organization needed.

## Key Features

### Semantic Search
- **Local AI models** - No external API calls, complete privacy
- **HNSW indexing** - Sub-10ms search across 10k+ items
- **Auto-vectorization** - Embeddings generated automatically on first search
- **Smart summarization** - Intelligent text condensation with priority-based extraction:
  - Preserves titles, descriptions, and code snippets
  - Extracts key technical terms (O(), Performance, Algorithm, etc.)
  - Retains structured content (bullet points, numbered lists)
  - Maintains context while reducing token usage
- **Parallel processing** - Rayon-based parallel vectorization for large sessions (2-3x speedup)
- **Query caching** - Deduplicates identical searches (~30% hit rate)

### Memory Architecture
- **Lock-free design** - Zero deadlocks, high concurrency
- **Three-tier hierarchy** - Hot/Warm/Cold for optimal speed
- **Persistent storage** - RocksDB with ACID guarantees
- **LRU eviction** - Automatic memory management

### Entity Tracking
- **Automatic extraction** - No manual tagging needed
- **Relationship graphs** - Petgraph-based connections
- **Importance scoring** - Identifies key concepts
- **Network visualization** - See how ideas connect

## Configuration

```rust
SystemConfig {
    // Memory limits
    max_hot_context_size: 50,      // Most recent items
    max_warm_context_size: 200,    // Frequently accessed

    // Semantic search
    enable_embeddings: true,        // Use AI models
    auto_vectorize_on_update: true, // Auto-generate embeddings
    semantic_search_threshold: 0.7, // Similarity cutoff

    // Storage
    data_directory: "./post_cortex_data",
    cache_capacity: 100,

    // Performance
    enable_metrics: true,
}
```

## Performance

**Real-world metrics (from production usage):**
- Session creation: 1000+ ops/sec
- Context updates: 500+ ops/sec with parallel vectorization
- Keyword search: < 10ms (entity graph queries)
- Semantic search: 50-200ms (with auto-vectorization)
- Embedding generation: 17-20 texts/sec (optimized for Mac M-series)
- Parallel vectorization: 2-3x speedup for large sessions (500+ items)
- Cache hit rate: ~50% (hot/warm tiers)
- Query cache: ~30% hit rate (semantic search deduplication)

**Scalability (verified in active development session):**
- 122+ conversation updates tracked
- 898+ entities extracted and mapped
- 1,015+ relationships in knowledge graph
- 10k+ semantic embeddings indexed
- Zero deadlocks across thousands of concurrent operations
- Linear scaling with CPU core count

## Architecture Highlights

**Lock-Free Concurrency:**
- Uses `DashMap`, `ArcSwap`, and atomic operations throughout core system
- Hybrid approach: hot path operations are lock-free (session management, metadata, caching), cold path operations (HNSW index rebuild) may use locks for correctness
- Actor pattern for async storage operations with message-passing concurrency
- Scales linearly with CPU cores for concurrent operations
- Zero deadlocks in production usage (verified with extensive concurrent testing)

**Why Lock-Free?**
Traditional locks can cause:
- Deadlocks when locks acquired in wrong order
- Performance bottlenecks under high concurrency
- Unpredictable latency spikes
- Priority inversion and convoy effects

Lock-free design eliminates these issues in hot paths through atomic operations (AtomicU32, AtomicU64, AtomicBool) and concurrent data structures (DashMap for hashmaps, ArcSwap for atomic pointer updates). The hybrid approach allows us to maintain correctness guarantees for complex operations like HNSW index building while ensuring zero deadlocks for user-facing operations.

## Development

```bash
# Run tests
cargo test --features embeddings

# Run with debug logging
RUST_LOG=post_cortex=debug cargo run --bin mcp-server --features embeddings

# Benchmarks
cargo bench
```

**Project Structure:**
```
src/
├── core/          # Lock-free memory system, vector DB, caching
├── session/       # Session management and entity extraction
├── storage/       # RocksDB persistence with actor pattern
├── graph/         # Entity relationship graphs (Petgraph)
├── summary/       # Analysis and summarization
└── tools/mcp/     # MCP protocol implementation (20+ tools)
```

## Troubleshooting

**Semantic search returns no results:**
- Auto-vectorization runs on first search - give it a moment
- Try broader search terms
- Check embeddings are enabled: `cargo build --features embeddings`

**High memory usage:**
- Adjust `max_hot_context_size` and `max_warm_context_size`
- Monitor with `get_session_statistics` tool

**Slow performance:**
- Verify HNSW indexing is active
- Check query cache is enabled (default: on)
- Use `get_vectorization_stats` for diagnostics

## System Requirements

- **Rust Toolchain:** Version 1.70+ with 2024 edition support
- **Supported Platforms:** Linux, macOS, and Windows
- **Storage Requirements:** Approximately 100MB for embeddings models plus conversation data
- **Memory Usage:** Configurable with default hot memory allocation of ~50MB

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions from the community! To maintain code quality and architectural integrity, please ensure that:

- **Lock-Free Principles**: All contributions must adhere to the lock-free architecture patterns established in the project
- **Comprehensive Testing**: Include appropriate tests for any new features or modifications
- **Documentation**: Update relevant documentation when making API changes or adding new functionality
- **Rust Best Practices**: Follow established Rust idioms and conventions for clean, efficient code

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines for detailed information on our development process and contribution standards.

## Acknowledgments

Post-Cortex is built upon exceptional open-source Rust libraries that form the foundation of our advanced memory system:

- **DashMap**: Provides the lock-free concurrent hashmap that enables our zero-deadlock architecture
- **ArcSwap**: Supplies atomic pointer swapping mechanisms for efficient concurrent data structure updates
- **RocksDB**: Delivers high-performance persistent storage with ACID transaction guarantees
- **Candle**: Powers our local machine learning inference for privacy-first semantic embeddings
- **Petgraph**: Enables sophisticated relationship mapping and graph traversal algorithms
- **Tokio**: Offers the asynchronous runtime that supports our high-concurrency operations

We are grateful to these projects and their maintainers for their invaluable contributions to the Rust ecosystem.
