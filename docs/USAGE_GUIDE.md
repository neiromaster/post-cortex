# Post-Cortex Usage Guide

How to effectively use Post-Cortex for AI assistant memory management.

## Mental Model

```
Workspace = Project (e.g., "E-commerce Platform", "Mobile App")
    └── Session = Feature/Task (e.g., "Authentication", "API Design")
            └── Memory Entry = Knowledge captured during work
```

**Key insight:** Memory stays isolated per workspace. When searching within a workspace, you only get results from that project's sessions.

## Quick Setup for New Projects

### 1. Create Session and Workspace

```
# Create session for your project
session(action: "create", name: "my-project", description: "Main development session")

# Create workspace to group sessions
manage_workspace(action: "create", name: "my-project", description: "My Project Development")

# Link session to workspace
manage_workspace(action: "add_session", workspace_id: "...", session_id: "...", role: "primary")
```

### 2. Configure CLAUDE.md

Create a `CLAUDE.md` file in your project root:

```markdown
# CLAUDE.md

## Session Context

| Key | Value |
|-----|-------|
| **Session ID** | `your-session-uuid-here` |
| **Workspace ID** | `your-workspace-uuid-here` |
| **Project** | my-project |

## Mandatory Rules

### RULE 1: Search Before Answering
Before answering questions about code or architecture, search first.

### RULE 2: Log After Discovery
After discovering anything new or making decisions, log immediately.

## Agent Reference

| Agent | subagent_type | When to Use |
|-------|---------------|-------------|
| Search | `search-specialist` | Before answering questions |
| Context | `context-builder` | After discovering/deciding anything |
```

See [CLAUDE.md](../CLAUDE.md) for a complete working example.

### 3. (Optional) Add Agent Definitions

For advanced workflows, add custom agents in `.claude/agents/`:

```
.claude/
└── agents/
    └── your-project-agents/
        ├── SKILL.md           # Main skill definition
        └── agents/
            ├── search.md      # Search agent
            └── context.md     # Context logging agent
```

See [.claude/agents/](../.claude/agents/) for working examples.

## The Workflow

```
User asks question
       │
       ▼
┌─────────────────┐
│ 1. SEARCH       │  ← Call search-specialist agent
│    past memory  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. EXPLORE      │  ← If not found, explore codebase
│    codebase     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. LOG          │  ← Call context-builder agent
│    discovery    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. RESPOND      │  ← Answer user
│    to user      │
└─────────────────┘
```

## Multi-Project Setup

### Option A: One Workspace per Project

```
Workspace: frontend-app
├── frontend-auth (role: primary)
├── frontend-ui-components (role: related)
└── frontend-api-client (role: dependency)

Workspace: backend-api
├── backend-auth (role: primary)
├── backend-database (role: related)
└── backend-messaging (role: dependency)
```

**Search scope:**
- `scope: "session"` → Current task only
- `scope: "workspace"` → Entire project
- `scope: "global"` → All projects

### Option B: Shared Sessions Across Projects

```
Workspace: frontend-app
├── frontend-auth (role: primary)
└── shared-api-contracts (role: shared)  ← Same session

Workspace: backend-api
├── backend-auth (role: primary)
└── shared-api-contracts (role: shared)  ← Same session
```

Use `role: "shared"` for sessions that belong to multiple workspaces.

## Knowledge Types

Use the right type when logging:

| Type | When to Use | Example |
|------|-------------|---------|
| `qa` | Answered a question | "How does auth work?" → "Uses JWT with refresh tokens" |
| `decision_made` | Made architectural choice | "Chose PostgreSQL for better JSON support" |
| `problem_solved` | Fixed a bug | "Memory leak was due to unclosed connections" |
| `code_change` | Significant code change | "Refactored auth module to use middleware pattern" |

## Handling Knowledge Obsolescence

Old knowledge can become stale. Use `recency_bias` to prioritize recent content:

```
# Recent bugs/issues - prioritize fresh content
semantic_search(query: "timeout errors", recency_bias: 0.7)

# Architecture decisions - all time equally relevant
semantic_search(query: "database choice", recency_bias: 0.0)

# Date-filtered search
semantic_search(
  query: "API changes",
  date_from: "2024-06-01",
  date_to: "2024-12-31"
)
```

| Scenario | Recommended `recency_bias` |
|----------|---------------------------|
| Debugging recent issues | 0.5 - 1.0 |
| Finding latest solutions | 0.3 - 0.7 |
| Architecture docs (timeless) | 0.0 |
| Current sprint context | 1.0+ |

## Common Patterns

### Pattern 1: Project Onboarding

When starting work on an existing project:

```
1. semantic_search(query: "project architecture overview", scope: "workspace")
2. get_structured_summary(session_id: "...", include: ["decisions"])
3. query_conversation_context(query_type: "entity_importance")
```

### Pattern 2: Before Making Changes

```
1. Search for related past decisions
2. Check if similar problems were solved before
3. Review relevant code change history
```

### Pattern 3: End of Session

```
1. Review what was discovered/decided
2. Ensure all important context was logged
3. Update session description if scope changed
```

## Troubleshooting

### Agent doesn't use memory tools

Add explicit rules in CLAUDE.md:
```markdown
**RULE 1: Search Before Answering**
MUST call search-specialist before answering ANY question about code.
**NO EXCEPTIONS.**
```

### Memory seems mixed across projects

Check workspace isolation:
1. Verify session is linked to correct workspace
2. Use `scope: "workspace"` instead of `scope: "global"`
3. Check `manage_workspace(action: "get")` to see linked sessions

### Old information appears in results

Use recency bias:
```
semantic_search(query: "...", recency_bias: 0.5)
```

Or filter by date:
```
semantic_search(query: "...", date_from: "2024-01-01")
```

## Reference Files

| File | Purpose |
|------|---------|
| [CLAUDE.md](../CLAUDE.md) | Working example of project configuration |
| [.claude/agents/](../.claude/agents/) | Custom agent definitions |
| [PROJECT.md](../PROJECT.md) | Development documentation |
