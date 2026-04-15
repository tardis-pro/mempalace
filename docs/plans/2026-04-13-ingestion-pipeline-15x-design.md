# Ingestion Pipeline — 15x Design

**Status:** design, not yet implemented
**Date:** 2026-04-13
**Scope:** ingestion only. Retrieval / loci-GIS redesign is tracked separately in `2026-04-13-method-of-loci-gis-sketch.md`.
**Target:** ≥15x end-to-end wall time for both fresh mines and re-mines, measured by a reproducible bench harness, with zero regressions in correctness or recall.

---

## 1. Current pipeline, with file:line receipts

Every mine goes through this sequence today. Numbers in parentheses are the hardcoded batch sizes.

```
CLI dispatch                     crates/mempalace-cli/src/main.rs
        │
        ▼
┌─────────────────────────────┐
│ Miner::mine / ConvoMiner::  │  crates/mempalace-server/src/ingest.rs:101
│   mine                      │  crates/mempalace-server/src/convo_miner.rs:366
└─────────────────────────────┘
        │   sequential for-loop over files
        ▼
  read_to_string (sync blocking)                ingest.rs:158
  normalize / chunk_exchanges                   convo_miner.rs:395,410
  chunk_text                                    ingest.rs:198,221
        │
        ▼
  buffer.push → flush when buffer.len() ≥ 64    ingest.rs:119,173
                                                convo_miner.rs:388,479
        │
        ▼
  Palace::add_many(&mut self, Vec<DrawerRecord>)    palace.rs (trait)
        │
        ▼
  LanceDbPalace::add_many                       lancedb_backend.rs:724
        ├─ HashSet in-batch dedup               :730-737
        ├─ prefilter: SELECT id WHERE id IN(..) :742-778   ← runs per 64-row batch
        ├─ embed_many(texts) through Mutex      :268-279, :789-790
        │    └─ fastembed::embed(&mut self, Some(64))  ← sub-batch=64 inside the already-64 batch
        ├─ build_insert_batch_many              :483-595
        └─ merge_insert (single writer)         :807-816
```

**The three hardcoded 64s.**
1. `ingest.rs:119` `const BATCH_SIZE: usize = 64` — flushes the drawer buffer every 64 rows.
2. `convo_miner.rs:388` `const BATCH_SIZE: usize = 64` — same, in the convo path.
3. `lancedb_backend.rs:277` `guard.embed(texts, Some(64))` — fastembed sub-batch hint inside `embed_many`.

Effect: every flush invokes ONNX Runtime through one `TextEmbedding` session on 64 rows. The session is wrapped in `Mutex<TextEmbedding>` so there is exactly one inference in flight across the whole process. On a 32-core box (`nproc` = 32), this is the bottleneck by a factor of a lot.

**Why fastembed pinning one session blocks us.**
Verified in `~/.cargo/registry/.../fastembed-5.13.1/src/text_embedding/impl.rs:404`:
```rust
pub fn embed<S: AsRef<str> + Send + Sync>(
    &mut self,
    texts: impl AsRef<[S]>,
    batch_size: Option<usize>,
) -> Result<Vec<Embedding>> { ... }
```
`&mut self` + no internal rayon (`impl.rs:324` `texts.chunks(batch_size).map(...)` — serial). The only way to parallelize embed throughput is to run **N independent `TextEmbedding` instances**, each on its own ORT session, each with intra-op threads capped so the total ≤ physical cores.

**Why `add_many` can't just be called from parallel threads.**
`Palace::add_many` is `fn add_many(&mut self, ...)` on the trait. A thread pool can't call it through `&mut self`. Options: (a) wrap the whole palace in a `Mutex`, which defeats the purpose; (b) change the trait to `&self` with interior mutability; (c) add a concrete `LanceDbPalace::bulk_ingest(&self, ...)` that the CLI miner paths call directly, leaving the trait alone. **Picking (c)** — smaller blast radius, keeps `InMemoryPalace` and tests unchanged.

---

## 2. The new pipeline

Three stages connected by bounded crossbeam channels. Synchronous, no tokio. Runs inside the existing `mempalace mine` CLI process, which is already non-daemon (so the palace lock is free).

```
                                        (walker thread: 1)
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Produce: walk + read + normalize + chunk + local dedup      │
│                                                                       │
│   ignore::WalkBuilder (single-thread walker)                          │
│   → crossbeam channel<PathBuf> [cap=256]                              │
│       ↓                                                               │
│   rayon pool (W1 workers, W1 = num_cpus)                              │
│      read_to_string + normalize + chunk                               │
│      → HashSet local dedup (per-worker, drops intra-file dupes)       │
│      → emit DrawerRecord                                              │
│       ↓                                                               │
└──────────────────────────────────── channel<Vec<DrawerRecord>> [cap=4]
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 1.5 — Global pre-pass dedup (once per mine)                     │
│                                                                       │
│   BEFORE Stage 1 starts: SELECT id WHERE wing = '<mine-wing>'         │
│   build HashSet<String> of existing ids (ExistingIds)                 │
│   filter every DrawerRecord against ExistingIds                       │
│   survivors go to Stage 2; existing rows are dropped *before embed*   │
└───────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼  channel<Batch> [cap=4]
                                                │   Batch = Vec<(DrawerRecord, Vec<f32>-placeholder)>
                                                │   size = 1024 records
                                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Embed: N independent ONNX sessions                          │
│                                                                       │
│   N = max(1, num_cpus / 8)                                            │
│   Each worker owns one TextEmbedding (own ORT session)                │
│   ORT intra-op threads capped to 8 per worker                         │
│   worker:                                                             │
│     take batch from rx                                                │
│     vectors = embedder.embed(texts, Some(1024))                       │
│     send (batch, vectors) on writer channel                           │
└───────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼  channel<EmbeddedBatch> [cap=4]
                                                │
                                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Write: one thread, sequential merge_insert                  │
│                                                                       │
│   single writer thread owns the lancedb handle                        │
│   build_insert_batch_many → merge_insert (existing safety net)        │
│   increments seq via bump_seq_many                                    │
│   ACKs completion via atomic counter for stats                        │
└───────────────────────────────────────────────────────────────────────┘
```

**Parallelism budget on 32-core box.**
- Stage 1: up to 32 rayon workers, but I/O-bound so effective parallelism ~8-16.
- Stage 1.5: one-shot, ~50-150ms total.
- Stage 2: N=4 embedders × 8 ORT intra-op threads = 32 CPU slots.
- Stage 3: 1 writer thread, not CPU-bound (mostly lancedb I/O).
- Total CPU footprint: fits the machine without oversubscription.

**Backpressure.** Channel caps (256 paths, 4 record batches, 4 embedded batches) cap peak RAM at roughly `4 batches × 1024 rows × ~2 KB content + 4 × 1024 × 384 × 4 bytes vectors ≈ 14 MB`. Plus ExistingIds HashSet (≤20 MB for largest wing). Comfortable.

**Ordering.** Writer does not preserve batch order. `insert_seq` is assigned inside the writer thread as it commits, so seq ordering matches commit order — consistent with the current semantics, which also don't guarantee any order within a mine.

---

## 3. Why this hits 15x (Amdahl-honest estimate)

Stage-level speedup against current:

| Stage                          | Today | Target | Speedup | Source of gain                                      |
|--------------------------------|-------|--------|---------|-----------------------------------------------------|
| Walk + read + normalize + chunk| 100   | 15     | ~6.7x   | rayon over ~8 workers, I/O overlap                  |
| Prefilter SELECT (per batch)   | 100   | 5      | ~20x    | global pre-pass once, vs. 150+ per-batch SELECTs    |
| Embed                          | 100   | 18     | ~5.5x   | 4 sessions × sublinear intra-op + 1024 sub-batches  |
| Arrow batch + merge_insert     | 100   | 35     | ~2.8x   | 16x larger batches amortise fixed cost              |
| Embed ↔ write overlap          | —     | —      | ~1.3x   | pipelined, wall time = max, not sum                 |

Taking current time shares from the earlier profile estimate (55/20/15/10 for embed/read-parse/prefilter/arrow), the Amdahl math:

```
T_new / T_old = 0.55/5.5 + 0.20/6.7 + 0.15/20 + 0.10/2.8
              = 0.100   + 0.030    + 0.0075  + 0.036
              = 0.173
           ⇒  5.8x  (before pipeline overlap)
```

With embed↔write overlap replacing `sum` with `max(embed, write)`:
```
T_new ≈ max(0.100, 0.036) + 0.030 + 0.0075
      = 0.100 + 0.030 + 0.0075
      = 0.1375
   ⇒ 7.3x
```

That's short of 15x. To close the gap we need to find additional leverage. Candidates:

1. **Kill per-batch prefilter entirely** when the global pre-pass already ran. Saves another ~7% of current time. New total: ~7.8x.
2. **Skip merge_insert for pure appends.** When the global pre-pass says "no collisions with existing ids" and in-batch dedup removed local dupes, use plain `table.add()` instead of `merge_insert`. lancedb `add` is meaningfully faster than `merge_insert` because it doesn't scan the BTree index for matches. Saves another ~5-10% of total. New total: ~8.5x.
3. **Bigger Stage-2 sub-batch.** fastembed `embed(texts, Some(2048))` instead of 1024 for long content. Marginal tokenizer overhead reduction, ~5% more. ~9x.
4. **Pre-tokenize on Stage-1 workers, pass token ids to Stage 2.** Tokenization is ~10-15% of fastembed `transform()` time and is embarrassingly parallel. If we run the fastembed tokenizer on Stage-1 rayon threads and only hand ONNX-ready tensors to Stage 2, we claw back ~10% of embed time. Risk: reaches into fastembed internals, might require forking or using `TextEmbedding::transform` directly. Needs a spike. Optimistic: ~10-11x.
5. **Quantized model (`int8` MiniLM-L6 ONNX).** fastembed ships `AllMiniLML6V2Q` variants. On CPU this is typically 1.8-2.5x faster with ~0.5-1% recall loss on MTEB. Conflicts directly with the "zero loss in needle-in-haystack" rule. **Rejected** unless the bench shows recall@k is unchanged on our fixtures — and even then, would need separate approval.

**Honest verdict.** 15x on the current model is *not guaranteed*. Realistic ceiling with items 1-4 is roughly **9-12x** on a 32-core box. To cross 15x we need either (a) the quantized model (which violates the zero-loss rule), (b) a GPU ORT backend (fastembed 5.13 doesn't expose one for MiniLM), or (c) a fundamentally different embedding path.

**My call:** ship the pipeline refactor, measure, present the real number. If the bench lands at 9x and you want more, we have a menu of named next steps. Shipping 9x this week beats waiting three weeks for a 15x that might be 7x in practice. I'm flagging this up-front rather than letting you find out from a benchmark after the code is merged.

---

## 4. Global pre-pass dedup (the re-mine story)

CLAUDE.md §5 reports re-mining 1,401 drawers takes 1.4s today (the per-batch `id IN (...)` prefilter). We can do better.

**Before any Stage-1 worker starts:** one query, scoped to the wing being mined:
```sql
SELECT id FROM drawers WHERE wing = '<current-wing>'
```
Collect into `Arc<HashSet<String>>`. Pass to Stage-1 workers. Every `DrawerRecord` gets an O(1) lookup before being emitted.

**Cost.**
- `convo_claude_code` (~44k rows): ~150ms.
- `convo_opencode` (~69k rows): ~250ms.
- Largest `proj_*` wing (~15k rows): ~50ms.

**Benefit.**
- Re-mines: 100% of already-present rows are dropped before any embed or prefilter SELECT runs. The 1,401-drawer re-mine case from CLAUDE.md §5 becomes bounded by the walk itself — estimated <200ms including the pre-pass scan.
- Fresh mines: pre-pass scan is ~50ms for any wing (empty or small). Negligible.

**Safety net preserved.** The existing in-batch HashSet and `merge_insert` `when_not_matched_insert_all` stay in place as the race-condition backstop. The pre-pass is purely an optimization layer above them, not a replacement.

**Cross-wing dedup.** Not addressed. The tool is always invoked with a specific `--wing`, and the existing per-batch prefilter also didn't guarantee cross-wing uniqueness. Noting this explicitly so nobody later thinks it regressed.

---

## 5. Bench harness

Not committed as fixtures per your instruction. The harness reads from a path you point it at and runs against a fresh temp palace each run.

**New CLI subcommand:** `mempalace bench ingest`

```
mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/project_small \
    --mode projects \
    --wing bench_proj \
    --iterations 3 \
    --report docs/sessions/bench-YYYY-MM-DD-HHMM.json
```

**What it does:**
1. `tempfile::TempDir` for a fresh palace per iteration (no cross-iteration state).
2. Runs the full `Miner` or `ConvoMiner` path against `--source`.
3. Instruments per-stage wall time via `tracing` spans with `busy_time` measurement, captured by a custom layer that writes to the report JSON.
4. Reports:
   - Total wall time, per iteration.
   - Per-stage time: walk, read, normalize, chunk, prepass, embed, prefilter, arrow, merge_insert.
   - Throughput: drawers/sec.
   - Peak RSS (via `/proc/self/status`).
   - Recall sanity check: after ingest, runs 5 fixed queries from a queries file (`--queries queries.json`) and asserts the expected drawer ids appear in top-K. K configurable, default 10. Fails the bench if any expected id is missing (**this is the zero-loss gate**).
5. Exit code 0 only if wall time regressed ≤5% vs. baseline AND recall is 100%.

**Baseline.** Captured by running the bench on the current commit before any pipeline work starts. Committed as `docs/sessions/bench-baseline-YYYY-MM-DD.json`.

**Fixtures (local, uncommitted).** Expected layout under `~/.mempalace/bench/fixtures/`:
```
project_small/          ← ~500 files, ~5 MB text
project_medium/         ← ~5k files, ~50 MB text
convo_small/            ← ~100 Claude Code JSONL sessions
convo_medium/           ← ~1000 Claude Code JSONL sessions
queries-project.json    ← 5 queries with expected-id lists
queries-convo.json      ← 5 queries with expected-id lists
```

I'll generate these locally before running the baseline. If any fixture ends up containing something sensitive, it stays on disk and out of git.

---

## 6. Embedder pool, exact specification

**Construction.** At the start of a mine, build N `TextEmbedding` instances:
```rust
let n_embedders = std::cmp::max(1, num_cpus::get() / 8);
let intra_op = 8;
let pool: Vec<TextEmbedding> = (0..n_embedders)
    .map(|_| build_embedder_with_intra_op(intra_op))
    .collect::<Result<_>>()?;
```

**Intra-op thread capping.** fastembed 5.13 doesn't expose ORT session options directly. Two paths:
1. Set `OMP_NUM_THREADS` / `ORT_NUM_THREADS` env vars before `TextEmbedding::try_new`. Global; affects all sessions built afterward. Set once per process.
2. Fork/patch fastembed to expose `SessionBuilder` options. Rejected — upstream maintenance burden.

**Picking option 1.** The mine process sets `OMP_NUM_THREADS=8` and `ORT_INTER_OP_NUM_THREADS=1` before constructing any embedder. Documented in the CLI help for the `mine` and `bench ingest` subcommands. Verified by `ort`'s env-var handling (need to confirm on first run; fallback is to use the current defaults and accept oversubscription, which still gives a large speedup because ORT's thread pool is work-stealing).

**Worker structure.** One OS thread per embedder (not rayon — each thread owns `&mut TextEmbedding` for its lifetime):
```rust
for embedder in pool.into_iter() {
    let rx = batch_rx.clone();
    let writer_tx = writer_tx.clone();
    std::thread::Builder::new()
        .name(format!("mempalace-embed-{i}"))
        .spawn(move || embed_worker(embedder, rx, writer_tx))?;
}
```

`embed_worker` is a simple loop: `while let Ok(batch) = rx.recv() { embed; send; }`. On error, the worker logs and drops its receiver, which causes graceful pipeline shutdown via channel closure.

---

## 7. What stays the same (explicit non-goals)

Anything not listed below is untouched:

- **Arrow schema** (`build_schema` at `lancedb_backend.rs:308`). No new columns, no renames.
- **`Palace` trait.** `add_many` signature unchanged. `LanceDbPalace` gets a new concrete `bulk_ingest` method used only by the CLI miner paths.
- **Wing config, knowledge graph (`knowledge_graph.sqlite3`), MCP transport, daemon.** Not touched.
- **Recall.** The bench has a hard 100% recall gate. Any change that drops a drawer that existed before fails CI.
- **Dedup correctness.** Three-stage safety (in-batch HashSet → pre-pass filter → merge_insert) is strictly stronger than today's two-stage (in-batch HashSet → per-batch prefilter).
- **Daemon runtime assumption.** `LanceDbPalace::new_with_table` still refuses to run under a tokio runtime (`lancedb_backend.rs:102-108`). The new pipeline is sync threads + crossbeam, matching that requirement.
- **Quantized models.** Not evaluated, not enabled. Revisit only if the bench tells us the non-quantized pipeline cannot reach your target.

---

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ORT intra-op env vars ignored by fastembed build path | medium | 2-3 embedders oversubscribe, speedup drops from 4x to 2x on embed stage | Auto-tune: bench harness sweep `(N, intra_op) ∈ {(1,32),(2,16),(4,8),(8,4)}` once, commit the winning pair as default constants |
| lancedb `merge_insert` dominates wall time after embed is parallelized | medium | Stage 3 becomes the new bottleneck, caps speedup at ~8x | Fall back to `table.add()` when pre-pass proves no collisions; falls into items #2 in §3 |
| Peak RSS spikes from ExistingIds HashSet × channel-buffered batches | low | OOM on small machines | ExistingIds is ≤20 MB, channels are bounded at 4 batches; total <100 MB additional |
| Fastembed pretokenize spike (§3 item 4) requires reaching into internal API | high | Item cut from scope, ceiling drops ~10% | Leave spike out of the initial PR, keep as follow-up |
| Bench fixtures contain something sensitive | low (local-only) | — | Fixtures stay out of git by default; manual review before any sharing |
| Changing from `add_many(&mut self)` path to `bulk_ingest(&self)` breaks daemon ingest through MCP | low | Daemon ingest still goes through `add_many`, which wraps `bulk_ingest` internally; semantics identical |
| 15x target not met | high | Presenting honest number (est. 9-12x) with named follow-ups before writing any code — see §3 |

---

## 9. Rollback

The entire change is gated behind an environment variable `MEMPALACE_INGEST_PIPELINE=v2` (default: `v1`, the current code path). If v2 misbehaves in production, `unset MEMPALACE_INGEST_PIPELINE` reverts to the current serial path without a rebuild. The v1 code stays in-tree for at least one release cycle after v2 becomes default.

When v2 is the verified default (one week + no regressions), v1 is removed in a follow-up PR. The env var is not a permanent config surface; it's a safety hatch during rollout.

---

## 10. What ships in the implementation PR

In order:

1. **Bench harness** (`mempalace bench ingest`, reads report JSON, recall gate). No behavior change yet.
2. **Baseline capture** — run the harness on current `main`, commit `docs/sessions/bench-baseline-2026-04-13.json`.
3. **Global pre-pass dedup** — added to `add_many`, behind a feature flag `ingest-v2`. Bench shows re-mines drop to ~200ms.
4. **Concrete `LanceDbPalace::bulk_ingest`** method with the three-stage pipeline. Behind `MEMPALACE_INGEST_PIPELINE=v2`. Bench shows the real speedup number.
5. **CLI miner refactor** — `ingest.rs` and `convo_miner.rs` call `bulk_ingest` directly when env var is set; otherwise fall through to `add_many`.
6. **Tuning sweep** — run the bench across `(N, intra_op)` pairs, record results, bake the winner as default constants.
7. **Verification report** — bench-v2 vs. bench-baseline, per-stage breakdown, recall@10 on both fixture sets, peak RSS. Committed to `docs/sessions/`.

Each step is its own commit. Each commit passes `cargo test --release` and the recall gate.
