# Post-Cortex

**Intelligent Memory System for AI Assistants**

Post-Cortex is a high-performance MCP server that transforms ephemeral AI conversations into persistent, searchable knowledge. Built in Rust with lock-free architecture, it enables AI assistants to maintain memory across sessions with semantic search and automatic knowledge graph construction.

## Features

- **Persistent Memory**: Three-tier storage (hot/warm/cold) with RocksDB
- **Semantic Search**: Local transformer models with configurable accuracy/speed
- **Knowledge Graph**: Automatic entity extraction with 95%+ accuracy
- **Privacy-First**: All processing local, zero external API calls
- **Lock-Free**: Zero-deadlock guarantee through atomic operations

## Quick Start

### Installation

**Homebrew (macOS/Linux):**
```bash
brew install julymetodiev/tap/post-cortex
```

**Direct Download:**
```bash
# macOS Apple Silicon
curl -L https://github.com/julymetodiev/post-cortex/releases/latest/download/pcx-aarch64-apple-darwin -o /usr/local/bin/pcx && chmod +x /usr/local/bin/pcx

# macOS Intel
curl -L https://github.com/julymetodiev/post-cortex/releases/latest/download/pcx-x86_64-apple-darwin -o /usr/local/bin/pcx && chmod +x /usr/local/bin/pcx

# Linux
curl -L https://github.com/julymetodiev/post-cortex/releases/latest/download/pcx-x86_64-unknown-linux-gnu -o /usr/local/bin/pcx && chmod +x /usr/local/bin/pcx
```

**Build from source:**
```bash
cargo build --release --features embeddings
# Binary: target/release/pcx
```

### Usage

Post-Cortex provides a **single unified binary** (`pcx`) supporting both stdio and SSE transports:

```bash
# Stdio mode (default) - for Claude Desktop
pcx

# Daemon mode - HTTP server on port 3737
pcx start

# Check status
pcx status

# Stop daemon
pcx stop
```

### Claude Desktop Configuration

**Stdio mode (simple):**
```json
{
  "mcpServers": {
    "post-cortex": {
      "command": "pcx"
    }
  }
}
```

**SSE mode (daemon):**
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

### Environment Variables

- `PC_HOST` - Host to bind (default: 127.0.0.1)
- `PC_PORT` - Port (default: 3737)
- `PC_DATA_DIR` - Data directory (default: ~/.post-cortex/data)
- `RUST_LOG` - Logging level (e.g., `RUST_LOG=debug`)

## Setting Up for Your Project

**1. Create CLAUDE.md in your project root:**

```markdown
# CLAUDE.md

## Using Post-Cortex for Knowledge Management

**Project session ID:** `YOUR-SESSION-ID-HERE`

**Throughout conversation:**
- Add context using `update_conversation_context(session_id: "YOUR-SESSION-ID")`
- Query with `semantic_search_session(session_id: "YOUR-SESSION-ID")`
```

**2. Ask your AI assistant:**
```
"Create a Post-Cortex session for this project"
```

**3. Add the returned session_id to CLAUDE.md**

## Available Tools (24 MCP Tools)

**Session Management:**
- `create_session`, `load_session`, `list_sessions`, `search_sessions`

**Adding Context:**
- `update_conversation_context` - QA, decisions, problems, code changes
- `bulk_update_conversation_context` - Batch updates

**Searching:**
- `semantic_search_session` - AI-powered semantic search (auto-vectorizes)
- `semantic_search_global` - Search across all sessions
- `query_conversation_context` - Fast keyword search (< 10ms)

**Analysis:**
- `get_structured_summary` - Full session overview
- `get_key_decisions`, `get_key_insights` - Decision timeline
- `get_entity_importance_analysis` - Entity analysis

**Workspace Management:**
- `create_workspace`, `list_workspaces`, `get_workspace`
- `add_session_to_workspace`, `remove_session_from_workspace`

## Architecture

**Memory Tiers:**
```
Hot (50 items)   → DashMap cache, instant access
Warm (200 items) → Compressed cache, fast access
Cold (unlimited) → RocksDB persistence
```

**Lock-Free Concurrency:**
- DashMap, ArcSwap, atomic operations
- Actor pattern for async storage
- Zero deadlocks, linear scaling

## Performance

- Semantic search: 1-7ms
- Keyword search: < 10ms
- Embedding generation: 17-20 texts/sec
- 500+ context updates/sec

## Development

```bash
# Run tests
cargo test --features embeddings

# Run with debug logging
RUST_LOG=post_cortex=debug cargo run --bin pcx --features embeddings
```

**Project Structure:**
```
src/
├── bin/pcx.rs     # Unified binary (stdio + SSE)
├── core/          # Lock-free memory, vector DB
├── session/       # Session management
├── storage/       # RocksDB persistence
├── graph/         # Entity relationships
└── tools/mcp/     # MCP protocol (24 tools)
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions welcome! Please ensure lock-free principles and include tests.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
