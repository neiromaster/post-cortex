# CLAUDE.md

Project instructions for AI assistants working with Post-Cortex.

**Development docs:** [PROJECT.md](PROJECT.md) | **No "Co-Authored-By" in commits**

---

## Session Context

| Key | Value |
|-----|-------|
| **Session ID** | `bf52f62e-8e26-4e9e-8501-c42753d9a9ee` |
| **Workspace ID** | `c7c6dfa7-85c6-42b0-9b37-f2034c569a71` |
| **Project** | post-cortex |

> **Note:** These IDs are project-specific. Replace with your own when adapting this file for a different project. See [USAGE_GUIDE.md](docs/USAGE_GUIDE.md#1-create-session-and-workspace) for setup instructions.

---

## Mandatory Rules

### RULE 1: Search Before Answering

**BEFORE answering ANY question about code, architecture, or past decisions:**

1. Call `search-specialist` agent with the question
2. Check results
3. Then formulate answer

**NO EXCEPTIONS.**

### RULE 2: Log After Discovery

**AFTER discovering anything new, making decisions, or changing code:**

1. Call `context-builder` agent IMMEDIATELY
2. Log what you discovered/decided/solved
3. Do this BEFORE responding to user

| Situation | Type |
|-----------|------|
| Answered a question | `qa` |
| Made a decision | `decision_made` |
| Fixed a bug | `problem_solved` |
| Changed code | `code_change` |
| New requirement | `requirement_added` |
| Concept explained | `concept_defined` |

### RULE 3: Self-Check

After EVERY response, verify:
- Did I search before answering a codebase question?
- Did I log any new discoveries?

If NO → fix immediately.

---

## Agent Reference

Use **Task tool** with `subagent_type`:

| Agent | subagent_type | When to Use |
|-------|---------------|-------------|
| Search | `search-specialist` | Before answering questions |
| Context | `context-builder` | After discovering/deciding anything |
| Analyst | `knowledge-analyst` | For summaries and analysis |
| Curator | `memory-curator` | For session/workspace management |

### Examples

**Search:**
```
Task(subagent_type="search-specialist")
prompt: "Search for: how embeddings work
         Session ID: bf52f62e-8e26-4e9e-8501-c42753d9a9ee"
```

**Log decision:**
```
Task(subagent_type="context-builder")
prompt: "Log decision_made:
         Decision: Using VoyageAI for embeddings
         Rationale: Better multilingual support
         Session ID: bf52f62e-8e26-4e9e-8501-c42753d9a9ee"
```

**Log Q&A:**
```
Task(subagent_type="context-builder")
prompt: "Log qa:
         Question: How does semantic search work?
         Answer: Uses MiniLM embeddings with cosine similarity
         Session ID: bf52f62e-8e26-4e9e-8501-c42753d9a9ee"
```

> **Tip:** Use `recency_bias` for time-sensitive searches (e.g., recent bugs). See [USAGE_GUIDE.md](docs/USAGE_GUIDE.md#handling-knowledge-obsolescence) for recommended values.

---

## MCP Tools (6 Consolidated)

<details>
<summary>Reference for agent authors</summary>

| Tool | Purpose |
|------|---------|
| `session` | Create/list sessions |
| `update_conversation_context` | Add context (single or bulk) |
| `semantic_search` | Universal search with scope |
| `get_structured_summary` | Session summaries |
| `query_conversation_context` | Entity analysis, queries |
| `manage_workspace` | Workspace CRUD |

</details>
