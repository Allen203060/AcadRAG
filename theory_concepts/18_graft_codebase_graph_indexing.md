# Concept 18: Graft Codebase Context Graph Indexing

## 1. What is Graft?
**Graft** (from TrailHQ: [github.com/trailhq/Graft](https://github.com/trailhq/Graft)) is an advanced, ultra-fast codebase graph indexer. It parses a software repository into an AST-based **Wiring Graph** of symbol relationships (functions, classes, file spans, call graphs, imports, and references).

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      Software Codebase                      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼  graft build ($0, Fast AST Parser)
 ┌─────────────────────────────────────────────────────────────┐
 │                Graft Symbol Wiring Graph                    │
 │  48 Nodes (Functions/Classes) │ 153 Edges (Call Relationships)│
 └──────────────────────────────┬──────────────────────────────┘
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│  graft ask <query>       │          │  graft callers <symbol>  │
│  Instant symbol search   │          │  Transitive blast radius │
└──────────────────────────┘          └──────────────────────────┘
```

## 2. Why Use Graft Over Standard Grep/File Reads?

| Operation | Traditional Approach | Graft Graph Approach | Efficiency Gain |
| :--- | :--- | :--- | :--- |
| **Finding who calls `hybrid_search`** | Read 5 full `.py` files (~3,925 tokens) | `graft callers hybrid_search` (144 tokens) | **96% Token Savings** |
| **Locating a function definition** | Full regex grep + file read | `graft ask "hybrid_search"` | **Instant file:line span** |
| **API Surface Inspection** | Opening entire file | `graft skeleton <file>` | **Signatures-only view** |

## 3. How Graft Works Under the Hood
1. **Tree-Sitter / Language AST Parsing:** `graft build` walks every `.py` file, extracting function signatures, definitions, and import spans.
2. **Call Graph Edge Extraction:** Computes incoming caller edges and outgoing dependency edges between symbols.
3. **Local Cache (`graft/`):** Indexes the graph in git-ignored `.graph/wiring.json` for sub-millisecond retrieval.
4. **Agent Integration (`graft init`):** Injects MCP configuration and AI agent system prompts (`AGENTS.md`, `.gemini/settings.json`) instructing AI assistants to query the graph before opening files.
