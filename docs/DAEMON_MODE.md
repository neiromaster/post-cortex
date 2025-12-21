# Daemon Mode

HTTP server for multi-client access to shared RocksDB instance.

## Overview

Daemon mode enables multiple Claude Code instances to share a single Post-Cortex database.

**Architecture:**
- Single HTTP daemon with lock-free concurrent access
- Multiple clients connect via HTTP/JSON-RPC
- Shared RocksDB storage with workspace isolation
- Streamable HTTP transport for MCP communication

## Quick Start

**1. Initialize config (optional):**
```bash
post-cortex-daemon init
```

**2. Start daemon:**
```bash
post-cortex-daemon start
```

**3. Check status:**
```bash
post-cortex-daemon status
```

## Configuration

**Priority:** Environment variables > Config file > Defaults

**Config file:** `~/.post-cortex/daemon.toml`
```toml
host = "127.0.0.1"
port = 3737
data_directory = "/Users/username/.post-cortex/data"
```

**Environment variables:**
```bash
PC_HOST=127.0.0.1
PC_PORT=3737
PC_DATA_DIR=~/.post-cortex/data
RUST_LOG=info  # debug, trace
```

**Defaults:**
- Host: 127.0.0.1 (localhost only)
- Port: 3737
- Data: ~/.post-cortex/data

## Commands

```bash
post-cortex-daemon init     # Create example config file
post-cortex-daemon start    # Start daemon (foreground)
post-cortex-daemon status   # Check if running
post-cortex-daemon stop     # Show stop instructions
post-cortex-daemon help     # Show usage
```

## Endpoints

**Health check:**
```bash
curl http://127.0.0.1:3737/health
```

**Server stats:**
```bash
curl http://127.0.0.1:3737/stats
```

**MCP JSON-RPC:**
```bash
curl -X POST http://127.0.0.1:3737/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "create_session",
    "params": {"name": "test", "description": "test session"},
    "id": 1
  }'
```

**MCP Streamable HTTP:**
```bash
# Initialize session
curl -X POST http://127.0.0.1:3737/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

## Workspaces

Organize related sessions (microservices, monorepo components).

**Tools:**
- `create_workspace` - Create workspace
- `get_workspace` - Get workspace details
- `list_workspaces` - List all workspaces
- `delete_workspace` - Delete workspace
- `add_session_to_workspace` - Add session to workspace
- `remove_session_from_workspace` - Remove session from workspace

**Session roles:**
- `Primary` - Main session
- `Related` - Related functionality
- `Dependency` - External dependency
- `Shared` - Shared utilities

**Example:**
```json
{
  "jsonrpc": "2.0",
  "method": "create_workspace",
  "params": {
    "name": "Microservices Project",
    "description": "Auth, API, Worker services"
  },
  "id": 1
}
```

## Production Usage

**Start with custom config:**
```bash
PC_PORT=8080 PC_DATA_DIR=/var/lib/post-cortex post-cortex-daemon start
```

**Debug logging:**
```bash
RUST_LOG=debug post-cortex-daemon start
```

**Stop daemon:**
```bash
# Ctrl+C in daemon terminal, or:
kill $(lsof -t -i:3737)
pkill -f post-cortex-daemon
```

## Lock-Free Architecture

- DashMap for concurrent session cache
- ArcSwap for atomic data structure updates
- Actor pattern for RocksDB operations
- Atomic counters for statistics
- Zero deadlock guarantee

## Performance

- Session operations: 1000+ ops/sec
- Concurrent clients: scales with CPU cores
- Hot/warm/cold memory hierarchy
- HNSW semantic search: 50-200ms
- Keyword search: <10ms

## Security

- Localhost only by default (127.0.0.1)
- No authentication (single-user design)
- File system permissions protect data directory
- CORS enabled for local development

## Troubleshooting

**Port already in use:**
```bash
lsof -i:3737
kill <PID>
```

**Permission denied:**
```bash
chmod 755 ~/.post-cortex
chmod 644 ~/.post-cortex/daemon.toml
```

**Config not loading:**
- Check file exists: `ls ~/.post-cortex/daemon.toml`
- Verify TOML syntax
- Use env vars to override

**High memory:**
- Reduce hot/warm cache sizes in code
- Monitor with `/stats` endpoint
