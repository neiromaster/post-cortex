# Changelog

All notable changes to Post-Cortex will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2025-11-22

### Added

- **Unified semantic search with workspace scoping**: New `semantic_search` MCP tool replaces separate session/global tools with unified interface supporting session, workspace, and global scopes
- **Workspace persistence**: Full RocksDB persistence and hydration for workspaces enabling reliable cross-session workspace management
- **Entity graph relations in vectorizer**: Entities now include up to 5 related concepts in vector embeddings through actual graph traversal
- **Comprehensive integration tests**: Added unified search integration tests covering session, workspace, and global scopes with complex multi-session scenarios

### Fixed

- **Query cache cross-session result leakage**: Critical bug where semantic search returned cached results from wrong sessions due to missing params_hash validation in similarity-based cache lookups
- **Semantic search missing updates after hot context eviction**: Implemented vectorized_update_ids tracking to ensure all updates remain searchable even after eviction from hot/warm tiers to cold storage
- **Unsafe unwraps in daemon server**: Replaced panic-prone .unwrap() calls with safe error handling in parameter parsing logic to prevent server crashes

### Changed

- **Entity graph integration**: Replaced TODO placeholders with actual graph relationship traversal in content vectorizer for both sequential and Rayon parallel processing paths
- **Vectorization strategy**: Modified to iterate incremental_updates instead of hot+warm context only, with DashSet-based tracking for efficient re-vectorization skipping

### Improved

- **Code quality**: Resolved all clippy warnings (27 → 0) including redundant clones, ptr_arg issues, and simplified boolean logic
- **Error handling**: Replaced unsafe unwraps with safe .map().unwrap_or_else() patterns in daemon endpoints
- **Performance**: Incremental vectorization with O(1) DashSet lookups for already-vectorized updates

### Removed

- Unused `extract_text_from_compressed_update` method from codebase

## [0.1.3] - 2025-11-21

### Fixed

- Data persistence: Fixed launchd /tmp data loss on reboot by moving to ~/.post-cortex/data
- Daemon logs moved to ~/.post-cortex/logs with absolute paths
- Unified storage: stdio + daemon now use ~/.post-cortex/data
- Fixed query_conversation_context data field in response

## [0.1.2] - 2025-11-20

### Fixed

- Standardized on MultilingualMiniLM embedding model
- Fixed text extraction to include question titles
- Semantic search similarity improved from 11% to 94%+

## [0.1.1] - 2025-11-19

### Fixed

- Fixed launchd configuration for proper daemon startup
- Resource URL interpolation in Homebrew formula

## [0.1.0] - 2025-11-18

### Added

- Initial release of Post-Cortex
- Lock-free conversation memory system with zero-deadlock guarantees
- Three-tier memory hierarchy (Hot/Warm/Cold)
- Local AI-powered semantic search with HNSW indexing
- NER-powered knowledge graph extraction
- 24 MCP tools for conversation context management
- RocksDB persistence layer
- Daemon mode with HTTP/SSE transport
