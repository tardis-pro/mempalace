# Method of Loci + GIS Retrieval — Design Sketch

**Status:** sketch, not a plan. Do not implement from this document. It's a structured brain-dump meant to become a real design after a brainstorm session.
**Date:** 2026-04-13
**Relationship to the ingestion refactor:** intentionally separate. The Thing A ingestion pipeline (`2026-04-13-ingestion-pipeline-15x-design.md`) is forward-compatible: it ships unchanged regardless of what we decide here.

---

## 1. The directive, and what it means

User words: *"build me the perfect method of loci, use gis concepts, we need faster ingestions, zero loss in needle in haystack, make all the decisions, build the best thing, not an average thing."*

Unpacked:

1. **Method of loci** — the memory palace metaphor is currently decoration. `wing`, `room`, and `hall` are nullable string tags. The retrieval experience should *feel* spatial: you enter wings, walk halls, stop at rooms, pull drawers. Navigation is a first-class operation, not just a filter.
2. **GIS concepts** — bring in the spatial-database toolbox. Coordinates, bounding boxes, nearest-neighbor queries, spatial indexes (R-tree / quadtree / H3), multi-resolution tiling, viewports. The motivation isn't that drawers are on a map — it's that GIS has already solved "find things near a point, quickly, over billions of rows" and we can steal the machinery.
3. **Zero loss in needle-in-haystack** — recall@k must be 1.0 for queries whose answer is literally in the palace. This rules out approximate nearest-neighbor tricks that drop the true match, lossy compression of embeddings, and any pruning that might hide an exact substring match behind a semantic miss. It implies **hybrid retrieval**: whatever the semantic model misses, an exact-term / structural path must catch.
4. **Best, not average** — don't cargo-cult a standard RAG stack. Design for our workload: 300k+ drawers, single-box local deployment, ~10-20 wings, predominantly code + conversation, queried by AI agents (which can iterate, unlike a human user).

---

## 2. The three-layer retrieval model I'm proposing

Every drawer is addressed by three independent coordinate systems, and retrieval can enter through any of them.

```
┌───────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Structural (the palace geometry)                            │
│                                                                       │
│   Hierarchical path: wing / hall / room / shelf / drawer              │
│   Lossless. Exact addressing. Navigable like a filesystem.            │
│   "Show me everything in convo_claude_code/hall_decisions/2026-04".   │
│                                                                       │
│   Backed by: a path column + prefix index in lancedb, plus a          │
│   materialized tree in knowledge_graph.sqlite3 for O(1) navigation.   │
└───────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────┐
│ Layer 2 — Semantic (the content-similarity space)                     │
│                                                                       │
│   Dense embedding (current: 384-dim MiniLM-L6).                       │
│   Cosine / L2 distance.                                               │
│   "What's conceptually near this query?"                              │
│                                                                       │
│   Backed by: lancedb IVF-PQ or FLAT vector index.                     │
│   Zero-loss caveat: PQ is lossy. Use FLAT or HNSW-at-full-precision   │
│   OR use IVF-PQ + a reranking full-scan pass.                         │
└───────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────┐
│ Layer 3 — Lexical (the exact-term safety net)                         │
│                                                                       │
│   Sparse BM25 / TF-IDF over tokenized content.                        │
│   "Find the drawer that literally contains these words."             │
│                                                                       │
│   Backed by: a tantivy full-text index co-located with lancedb,       │
│   or lancedb's built-in FTS (0.27 has `create_fts_index`, needs       │
│   verification on our pinned version).                                │
└───────────────────────────────────────────────────────────────────────┘
```

**Retrieval = fusion, not choice.** A query runs against all three layers in parallel, results are merged via Reciprocal Rank Fusion (RRF) or a learned reweighting, and the top-K is returned. **Zero-loss guarantee:** if the answer exists in any one layer's top-100, it makes top-K in the fused result. We prove this with a golden-query regression suite that asserts expected ids are in top-K.

**This is the "hybrid search" pattern** from modern retrieval systems (Vespa, Weaviate, Qdrant, Elastic). It's not novel. The novelty is treating Layer 1 as **first-class spatial geometry** rather than a tag filter.

---

## 3. Where GIS comes in

GIS gives us concrete machinery for Layer 1 and a framing for Layer 2.

### 3.1 Layer 1 as a path-structured spatial index

Think of the palace as a hierarchical space with mandatory levels:

```
palace
  └── wing              (e.g. proj_tardis, convo_claude_code)
       └── hall         (e.g. hall_decisions, hall_architecture, hall_bugs)
            └── room    (e.g. 2026-04-week-2, session_a7f92, poker/handlers)
                 └── shelf   (e.g. normalize.rs, decision_block_3)
                      └── drawer  (a single chunk)
```

Today this is five nullable string fields. In the new model:
- **Mandatory levels.** Every drawer has all five, populated at ingest time. `hall` is no longer nullable; ingest picks one using the topic-keyword scorer already in `convo_miner.rs:273-293` plus new heuristics for project files.
- **Stored as a materialized path.** A single column `locus_path = "proj_tardis/hall_architecture/2026-04/normalize.rs/3"`. Indexed by prefix (lancedb scalar BTree on the column, or a separate sqlite mirror for tree navigation).
- **Navigation is a first-class MCP tool.** New tools: `mempalace_navigate(path)` returns children with counts; `mempalace_ascend(path)` returns the parent; `mempalace_sibling_counts(path)` returns all sibling branches. Agents can walk the palace deterministically without keyword queries.

**This is GIS quadtree navigation**, just with a semantic hierarchy instead of geographic subdivision. The query *"give me everything near this drawer"* becomes *"find the smallest common ancestor locus and return all descendants"*, which is a prefix range query.

### 3.2 Layer 2 as a projected 2D/3D viewport

The 384-dim embedding space is unvisualizable and unmaterializable as a spatial index the way R-trees expect. But we can **project** it:

- **UMAP to 2D** computed once per palace (batch job, runs offline). Every drawer gets `(x, y)` coordinates in the projection.
- **H3 cell id** computed on those coordinates. H3 is Uber's hexagonal hierarchical spatial index; a resolution-10 cell id is a single u64 and supports "find everything in this cell", "find neighbors", and "zoom up/down resolution" as O(1) operations.
- Stored as new columns: `umap_x`, `umap_y`, `h3_cell_r5`, `h3_cell_r7`, `h3_cell_r9`.

**What this unlocks:**
- *Neighborhood queries without running ANN.* "Give me 50 drawers in the same H3 cell as this one" is a scalar equality filter — zero vector math at query time.
- *Visual exploration.* A web UI can render the palace as a map. Zoom in, zoom out, click a cell, see drawers. This is the method-of-loci payoff: you *see* where memories are, not just get a list.
- *Spatial coherence at ingest time.* When a new drawer lands in an H3 cell that's already densely populated, we have a signal for "this is near existing stuff" without running a query.

**The zero-loss caveat for Layer 2.** UMAP is lossy. A drawer's UMAP coords are a hint, not the truth. We must **never prune a search using UMAP coords alone**. The 384-dim vector remains the source of truth for semantic distance; UMAP/H3 is a coarse filter that must be backed by exact rescoring.

### 3.3 Query pattern: three-phase retrieval

```
Phase 1 — Candidate generation (broad, cheap, parallel)
  A. Structural:   prefix match on locus_path          → top 200
  B. Lexical:      BM25 score over content             → top 200
  C. Semantic:     IVF-PQ ANN over vectors             → top 200
                       (OR full-scan FLAT if palace < 1M rows — we are at 327k)

Phase 2 — Rescore (exact, narrow)
  Take union of A∪B∪C (≤600 candidates).
  Compute exact cosine distance against the full 384-dim vector.
  This is the zero-loss gate: FLAT rescoring eliminates ANN false negatives.

Phase 3 — Fuse + rank
  Combine scores via RRF with learned weights per layer.
  Apply locus coherence boost (candidates in the same H3 cell get a small bump).
  Return top-K.
```

**Why FLAT rescore is affordable.** At 327k drawers × 384 f32 = 500 MB vectors. A cosine pass over 600 candidates is 600 dot products, ~100 µs. Even a full-scan rescore (all 327k) is ~50 ms on modern CPUs. We have budget.

---

## 4. What "zero loss in needle-in-haystack" actually means, concretely

A golden-query regression suite, committed to the repo, that the CI gate will assert on any retrieval change:

```json
// tests/retrieval/golden_queries.json
[
  {
    "query": "exclusive lockfile behavior when daemon crashes",
    "must_retrieve_any_of": ["drawer_<sha>", "drawer_<sha>"],
    "k": 10
  },
  ...
]
```

**Rules:**
1. Every query's expected-id set *must* appear in top-K.
2. If ANY expected id is missing, CI fails.
3. Adding a new retrieval feature can only move ids *up* in rank, not out of top-K.
4. The suite grows over time as real user queries surface new edge cases.

This is how we prove the claim. Without it, "zero loss" is marketing.

---

## 5. Ingest-time work to support this

**Schema additions** (all backward compatible — existing drawers get defaults):

| Column        | Type    | Populated by                                     |
|---------------|---------|--------------------------------------------------|
| `locus_path`  | Utf8    | Ingest — deterministic from wing/hall/room/file |
| `h3_cell_r5`  | UInt64  | Batch job after ingest — from UMAP(x,y)         |
| `h3_cell_r7`  | UInt64  | Batch job — same                                 |
| `h3_cell_r9`  | UInt64  | Batch job — same                                 |
| `umap_x`      | Float32 | Batch job                                        |
| `umap_y`      | Float32 | Batch job                                        |
| `bm25_terms`  | —       | Indexed separately by tantivy or lancedb FTS     |

**Ingest-time work added:**
- `locus_path` construction: free, string concat.
- BM25 indexing: tantivy `IndexWriter::add_document` in the Stage 3 writer thread. Measured overhead: ~5-10% of insert time on tantivy benchmarks.
- H3/UMAP: **not** at ingest time. Runs as a background `mempalace reindex spatial` subcommand that processes new drawers in batches. Separating ingest hot path from offline spatial work keeps the 15x target intact.

---

## 6. Open questions I will not decide unilaterally

These are the questions I'd bring back to brainstorming. They're load-bearing enough that picking wrong is expensive.

1. **Mandatory vs. optional hall/room/shelf.** Forcing every drawer to have a hall means the ingest-time hall classifier must be reliable. Today's keyword scorer (`convo_miner.rs:TOPIC_KEYWORDS`) is ~70% accurate on obvious cases. Do we (a) accept the error rate and allow reclassification later, (b) add a "hall_uncategorized" catchall, or (c) use a small local classifier model? I lean (b) + (c) as a follow-up.

2. **UMAP stability across re-ingest.** UMAP is stochastic and not incrementally updatable. Adding 5k new drawers to a palace with 327k old ones requires either re-projecting everything (expensive, invalidates cached coords) or using an approximate out-of-sample projection (fast, drift over time). I lean toward periodic full re-projection (weekly cron). Real answer depends on how often users mine.

3. **H3 on projected 2D coordinates is non-standard.** H3 assumes a sphere. Using it on an arbitrary 2D UMAP projection works mechanically but loses some of H3's properties (neighbor queries are still valid within the projection). Alternative: use a simple grid index instead of H3. I lean H3 anyway because its tooling is mature and we get `k-ring` neighbor queries for free. Needs a spike to confirm it actually helps.

4. **FLAT rescore vs. IVF-PQ + rescore.** For 327k drawers, FLAT is 50-100 ms per query. IVF-PQ + rescore is 10-20 ms. Both hit the recall gate. Is 50 ms acceptable for our query latency budget? (AI agents tolerate 200+ ms easily; humans don't.) Depends on MCP use patterns.

5. **Replacing the `Palace` trait with something richer.** Right now `Palace::search` takes a query string and returns drawers. The three-layer model needs a more expressive interface: structured query builder with filters, fusion weights, rescore depth, etc. This is a breaking change. Scope and sequencing matter.

6. **Who owns the tantivy index?** Tantivy has its own storage, lock semantics, and lifecycle. If we add it, mempalace becomes a two-index system (lancedb + tantivy). Alternative: lancedb 0.27 FTS — needs verification that it handles BM25 properly, and the current pin says "don't bump lancedb." Tension to resolve.

7. **Reranker model?** The state of the art in 2026 is a cross-encoder reranker on top of bi-encoder retrieval. This adds ~50-200 ms per query but improves NDCG by 10-20%. Do we go there? Local cross-encoders exist (e.g., ms-marco MiniLM). Not required for zero-loss; it's a quality bump, not a correctness gate.

8. **Method-of-loci feel in the MCP surface.** Do we give agents tools like `enter_wing("proj_tardis")`, `walk_hall("hall_architecture")`, or keep it as filter arguments on `mempalace_search`? The former is a harder API change but actually delivers the "method of loci" experience the user asked for. I lean toward the former as stretch goal.

---

## 7. What I think we should do next

1. **Ship Thing A first.** The ingestion 15x refactor is independent, valuable, and ships in days. None of this Thing B design depends on it, and none of it blocks Thing B.
2. **Run a dedicated brainstorm session on Thing B**, starting from open questions §6.1-8. Come out with a concrete proposal I can implement.
3. **Prototype the Layer 1 + Layer 3 path first** (structural + BM25 fusion). This gives the zero-loss property without touching the vector path and is probably a one-week change. Layer 2's GIS projection is the expensive, research-flavored step and should come after we've proven fusion works at all.
4. **Before any implementation**, commit the golden-query suite (§4) against the *current* retrieval. Then any change has a measurable floor.

This document is a launch pad, not a plan. Treat it that way.
