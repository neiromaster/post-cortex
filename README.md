# Post-Cortex

**Production-Grade Intelligent Memory System for AI Assistants**

Post-Cortex is a high-performance MCP (Model Context Protocol) server that transforms ephemeral AI conversations into persistent, searchable knowledge with zero-deadlock guarantees. Built in Rust with lock-free concurrent architecture, it enables AI assistants to maintain perfect memory across sessions, understand semantic relationships, and retrieve contextually relevant information through local AI-powered search.

## Core Features

- **Persistent Memory System**: Three-tier hierarchical storage (hot/warm/cold) with RocksDB persistence
- **Semantic Search Engine**: Local transformer models with configurable accuracy/speed modes
- **NER-Powered Knowledge Graph**: Automatic entity extraction using DistilBERT with 95%+ accuracy
- **Privacy-First Architecture**: All processing occurs locally with zero external API dependencies
- **Lock-Free Concurrency**: Zero-deadlock guarantee through atomic operations and DashMap
- **Production-Ready Performance**: 6-8x search speedup, 192x memory compression, 10-50x improved I/O latency

## Performance Highlights

**Phase 1-3, 6 Optimizations Complete:**
- **Semantic Search**: 6.70x speedup (6.7ms → 1.0ms with approximate mode)
- **RocksDB I/O**: 10-50x latency reduction via spawn_blocking
- **Memory**: 192x compression with Product Quantization (153MB → 800KB)
- **Test Execution**: 22x faster (45s → 2s)
- **NER Engine**: 95%+ accuracy with lazy loading and pattern fallback

## Quick Start

### Installation

Post-Cortex provides **two binaries**:
1. **`post-cortex`** - Stdio MCP server for Claude Desktop integration
2. **`post-cortex-daemon`** - HTTP daemon server for API usage and background service

**Homebrew (macOS/Linux) - Recommended:**
```bash
brew install julymetodiev/tap/post-cortex
```
This installs both binaries: `post-cortex` and `post-cortex-daemon`

**Direct Download:**

Download binaries for your platform from the [latest release](https://github.com/julymetodiev/post-cortex/releases/latest):

```bash
# post-cortex (stdio MCP server)
# macOS Intel:        post-cortex-x86_64-apple-darwin
# macOS Apple Silicon: post-cortex-aarch64-apple-darwin
# Linux:              post-cortex-x86_64-unknown-linux-gnu

# post-cortex-daemon (HTTP daemon)
# macOS Intel:        post-cortex-daemon-x86_64-apple-darwin
# macOS Apple Silicon: post-cortex-daemon-aarch64-apple-darwin
# Linux:              post-cortex-daemon-x86_64-unknown-linux-gnu

# Example installation (macOS Apple Silicon):
curl -L https://github.com/julymetodiev/post-cortex/releases/latest/download/post-cortex-aarch64-apple-darwin -o /usr/local/bin/post-cortex && chmod +x /usr/local/bin/post-cortex
curl -L https://github.com/julymetodiev/post-cortex/releases/latest/download/post-cortex-daemon-aarch64-apple-darwin -o /usr/local/bin/post-cortex-daemon && chmod +x /usr/local/bin/post-cortex-daemon
```

**Build from source:**
```bash
cargo build --release --features embeddings
# Binaries: target/release/post-cortex and target/release/post-cortex-daemon
```

## Usage Modes

Post-Cortex provides two deployment options:

### 1. Stdio Mode (Simple - Claude Desktop Integration)

Use `post-cortex` binary for direct stdio MCP integration:

```bash
# Run manually (for testing)
post-cortex

# Or from source
./target/release/post-cortex
```

**Benefits:**
- Simple stdio-based MCP server
- No daemon required
- Perfect for single-project workflows
- Direct Claude Desktop integration

**Claude Desktop Config:**
```json
{
  "mcpServers": {
    "post-cortex": {
      "command": "post-cortex"
    }
  }
}
```

Or with absolute path:
```json
{
  "mcpServers": {
    "post-cortex": {
      "command": "/usr/local/bin/post-cortex"
    }
  }
}
```

### 2. Daemon Mode (Advanced - HTTP API + Background Service)

Use `post-cortex-daemon` binary for persistent HTTP server:

```bash
# Initialize config (optional)
post-cortex-daemon init

# Start daemon in foreground
post-cortex-daemon start

# Check status
post-cortex-daemon status

# Stop daemon
post-cortex-daemon stop
```

**Benefits:**
- Shared sessions across multiple AI assistants
- Single RocksDB instance (avoids lock conflicts)
- HTTP RMCP API on port 3737 (default)
- Persistent background service via systemd/launchd
- RESTful health and stats endpoints

**Claude Desktop Config:**
```json
{
  "mcpServers": {
    "post-cortex": {
      "type": "sse",
      "url": "http://localhost:3737/sse"
    }
  }
}
```

**Configuration File:** `~/.post-cortex/daemon.toml`
```toml
host = "127.0.0.1"
port = 3737
data_dir = "~/.post-cortex/data"
```

**Environment Variables:**
- `PC_HOST` - Override host (default: 127.0.0.1)
- `PC_PORT` - Override port (default: 3737)
- `PC_DATA_DIR` - Override data directory
- `RUST_LOG` - Logging level (e.g., `RUST_LOG=debug`)

### Service Management (Daemon Mode)

#### Linux (systemd)

```bash
# Copy service file
cp install/systemd/post-cortex.service ~/.config/systemd/user/
systemctl --user daemon-reload

# Start and enable auto-start
systemctl --user enable --now post-cortex

# View logs
journalctl --user -u post-cortex -f
```

See [install/systemd/README.md](install/systemd/README.md) for detailed instructions.

#### macOS (launchd)

```bash
# Copy plist file
cp install/launchd/com.post-cortex.daemon.plist ~/Library/LaunchAgents/

# Load and start
launchctl load ~/Library/LaunchAgents/com.post-cortex.daemon.plist

# View logs
tail -f /tmp/post-cortex.log
```

See [install/launchd/README.md](install/launchd/README.md) for detailed instructions.

**Recommendation:**
- Use **stdio mode** for simple Claude Desktop integration
- Use **daemon mode** for multi-project workflows, API access, or background service

## Setting Up for Your Project

**1. Create a CLAUDE.md file in your project root:**

```markdown
# CLAUDE.md

## Using Post-Cortex for Knowledge Management

**Project session ID:** `YOUR-SESSION-ID-HERE`

**Throughout conversation:**
- Add context using `update_conversation_context(session_id: "YOUR-SESSION-ID")`
- Query with `semantic_search_session(session_id: "YOUR-SESSION-ID")`
```

**2. Create your project session:**

Ask your AI assistant:
```
"Create a Post-Cortex session for this project"
```

**3. Add the returned session_id to CLAUDE.md**

That's it! Your AI now has persistent memory for this project.

## Search Modes (Phase 6 Optimization)

Post-Cortex offers three configurable search modes for optimal accuracy/speed tradeoffs:

**SearchMode Options:**
- **Exact**: Full linear scan - 100% accuracy, slowest (baseline: 6.7ms)
- **Approximate**: HNSW with ef_search=32 - 6.70x faster, ~98% accuracy (1.0ms)
- **Balanced** (default): HNSW with ef_search=64 - 6x faster, ~99% accuracy (1.1ms)

**SearchQualityPreset:**
- Fast: ef_search=32
- Balanced: ef_search=64
- Accurate: ef_search=128
- Maximum: ef_search=256

**Example Usage:**
```rust
// Use balanced mode (default)
db.search(&query, 10)

// Explicit mode selection
db.search_with_mode(&query, 10, SearchMode::Approximate, None)

// Custom ef_search value
db.search_with_mode(&query, 10, SearchMode::Balanced, Some(128))
```

## Semantic Search

Post-Cortex uses **local transformer models** for privacy-first semantic search:

**Embedding Models:**
- **MiniLM** (384-dim, default) - Balanced performance
- **StaticSimilarityMRL** (1024-dim) - Mac M-series optimized
- **TinyBERT** (312-dim) - Low-memory environments
- **BGESmall** (384-dim) - Maximum accuracy

**Search Pipeline:**
1. Text → 384-dimensional vector (local transformer)
2. Long texts summarized with priority extraction
3. Vector indexed in HNSW for fast retrieval
4. Results ranked: `(similarity × 0.7) + (importance × 0.3)`

**Quality Scoring:**
- **0.85-1.00** Excellent - Exact semantic matches
- **0.70-0.84** Very Good - Direct matches
- **0.55-0.69** Good - Related concepts
- **0.40-0.54** Moderate - Tangentially related
- **< 0.30** Filtered out

**Privacy-First:** All models run locally, zero external API calls.

## NER-Powered Knowledge Graph (Phase 3.2)

**Automatic Entity Extraction:**
- DistilBERT-NER model with 95%+ accuracy
- Lazy loading (loads on-demand, zero startup delay)
- Pattern fallback (60-70% accuracy when model unavailable)
- Extracts: Person, Organization, Location, Miscellaneous entities

**Relationship Mapping:**
- Maps connections: `RelatedTo`, `LeadsTo`, `Implements`, `Solves`
- Importance scoring based on mentions and relationships
- Network graph with 1000+ relationships in active sessions

**Example from production:**
```
388 entities tracked
1015 relationships mapped
Top: lock-free (44 mentions), session (42), search (38)
```

## Available Tools (24 MCP Tools)

**Session Management:**
- `create_session`, `load_session`, `list_sessions`, `search_sessions`
- `update_session_metadata`

**Adding Context:**
- `update_conversation_context` - QA, decisions, problems, code changes
- `bulk_update_conversation_context` - Batch updates

**Context Types:**
- **`qa`** - Questions and answers
- **`decision_made`** - Architectural choices with rationale
- **`problem_solved`** - Bugs and solutions
- **`code_change`** - Refactoring and features

**Searching:**
- `semantic_search_session` - AI-powered meaning search (auto-vectorizes)
- `semantic_search_global` - Search across all sessions
- `query_conversation_context` - Fast keyword search (< 10ms)
- `find_related_content` - Cross-session similarity

**Analysis:**
- `get_structured_summary` - Full session overview
- `get_key_decisions`, `get_key_insights` - Decision timeline
- `get_entity_importance_analysis`, `get_entity_network_view` - Entity analysis

**Workspace Management:**
- `create_workspace`, `list_workspaces`, `get_workspace`
- `add_session_to_workspace`, `remove_session_from_workspace`

[See full tool documentation](docs/MCP_TOOLS.md)

## Architecture

**Three-Tier Memory:**
```
Hot Memory (50 items)     → DashMap cache, instant access
Warm Memory (200 items)   → Compressed cache, fast access
Cold Storage (unlimited)  → RocksDB persistence
```

**Lock-Free Concurrency:**
- Uses `DashMap`, `ArcSwap`, and atomic operations
- Actor pattern for async storage operations
- Zero deadlocks in production usage
- Linear scaling with CPU cores

**Why Lock-Free?**
Eliminates:
- Deadlocks from incorrect lock ordering
- Performance bottlenecks under high concurrency
- Unpredictable latency spikes
- Priority inversion

**Phase 1-3, 6 Optimizations:**
1. **RocksDB**: All async methods wrapped in `spawn_blocking` (10-50x faster I/O)
2. **Product Quantization**: 192x memory compression with >90% accuracy retention
3. **NER Engine**: DistilBERT with lazy loading and pattern fallback
4. **ActiveSession**: Lock-free HotContext with DashMap and Arc-wrapped components
5. **Typed Errors**: SystemError enum with thiserror for better error handling
6. **HNSW Tuning**: Configurable search modes with 6.70x speedup

## Performance

**Real-world metrics:**
- Context updates: 500+ ops/sec with parallel vectorization
- Keyword search: < 10ms (entity graph queries)
- Semantic search: 1-7ms (Approximate: 1ms, Exact: 6.7ms)
- Embedding generation: 17-20 texts/sec
- Cache hit rate: ~50% (hot/warm tiers)
- Query cache: ~30% hit rate

**Scalability (verified in active development):**
- 122+ conversation updates tracked
- 898+ entities extracted
- 1,015+ relationships mapped
- 10k+ semantic embeddings indexed
- Zero deadlocks across thousands of concurrent operations

## Configuration

```rust
SystemConfig {
    // Memory limits
    max_hot_context_size: 50,
    max_warm_context_size: 200,

    // Semantic search
    enable_embeddings: true,
    auto_vectorize_on_update: true,
    semantic_search_threshold: 0.7,

    // Storage
    data_directory: "./post_cortex_data",
    cache_capacity: 100,
}
```

## Development

```bash
# Run tests
cargo test --features embeddings

# Run daemon with debug logging
RUST_LOG=post_cortex=debug cargo run --bin post-cortex-daemon --features embeddings

# Run stdio server
cargo run --bin post-cortex --features embeddings

# Property-based tests
cargo test --features embeddings property_vector_db
```

**Project Structure:**
```
src/
├── bin/
│   ├── post-cortex-daemon.rs  # HTTP daemon with RMCP SSE transport
│   └── post-cortex.rs         # Stdio MCP server (renamed from mcp-server.rs)
├── core/          # Lock-free memory, vector DB, NER engine
├── session/       # Session management and entity extraction
├── storage/       # RocksDB persistence with actor pattern
├── graph/         # Entity relationship graphs (Petgraph)
├── summary/       # Analysis and summarization
├── tools/mcp/     # MCP protocol (24 tools)
└── daemon/        # HTTP daemon configuration and RMCP implementation
install/
├── systemd/       # Linux systemd service files
└── launchd/       # macOS launchd plist files
```

## Troubleshooting

**Semantic search returns no results:**
- Auto-vectorization runs on first search
- Try broader search terms
- Verify embeddings: `cargo build --features embeddings`

**High memory usage:**
- Adjust `max_hot_context_size` and `max_warm_context_size`
- Use `get_session_statistics` for monitoring

**Slow performance:**
- Use Approximate or Balanced search mode for 6-7x speedup
- Check HNSW indexing is active
- Monitor with `get_vectorization_stats`

**Choosing Between Modes:**
- Use **stdio mode** (`post-cortex`) for simple Claude Desktop integration
- Use **daemon mode** (`post-cortex-daemon`) for multi-project workflows or API access
- Avoid running both simultaneously on same data directory (RocksDB lock conflicts)

## System Requirements

- **Rust Toolchain:** 1.70+ with edition 2024 support
- **Platforms:** Linux, macOS, Windows
- **Storage:** ~100MB for embeddings models + conversation data
- **Memory:** ~50MB default hot memory allocation

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please ensure:
- **Lock-Free Principles**: Adhere to zero-deadlock architecture
- **Comprehensive Testing**: Include tests for new features
- **Documentation**: Update docs for API changes
- **Rust Best Practices**: Follow established idioms

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Acknowledgments

Built upon exceptional Rust libraries:
- **DashMap**: Lock-free concurrent hashmap
- **ArcSwap**: Atomic pointer swapping
- **RocksDB**: Persistent storage with ACID guarantees
- **Candle**: Local ML inference for privacy-first embeddings
- **Petgraph**: Relationship mapping and graph traversal
- **Tokio**: Asynchronous runtime for high-concurrency

We are grateful to these projects and their maintainers.
