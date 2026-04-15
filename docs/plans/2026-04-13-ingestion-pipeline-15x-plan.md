# Ingestion Pipeline 15x — Implementation Plan

**Design reference:** `docs/plans/2026-04-13-ingestion-pipeline-15x-design.md`
**Executor assumption:** no prior session context. Every step lists exact file, line, and acceptance criteria.
**Branch:** `ingest-pipeline-v2` (created fresh from `main`)
**Feature flag:** `MEMPALACE_INGEST_PIPELINE=v2` (default `v1` = current code)

---

## Execution order and gates

Each step ends with a **hard gate** — a command that must pass before the next step starts. If a gate fails, stop and investigate root cause. Do not advance.

```
Step 1 — Bench harness scaffolding            [gate: cargo test --release]
Step 2 — Baseline capture on main             [gate: baseline JSON committed]
Step 3 — Global per-wing pre-pass dedup       [gate: re-mine bench < 300ms]
Step 4 — Concrete LanceDbPalace::bulk_ingest  [gate: bench v2 vs baseline]
Step 5 — CLI miner routing (ingest + convo)   [gate: bench v2 on both fixtures]
Step 6 — Tuning sweep                         [gate: constants committed]
Step 7 — Final verification + results report  [gate: recall@10 = 100%]
```

---

## Step 1 — Bench harness

**Files to create:**

- `crates/mempalace-cli/src/bench.rs` (new)
- Add `mempalace bench ingest` subcommand in `crates/mempalace-cli/src/main.rs` under the existing `Command` enum (see line numbers in `main.rs` for where other subcommands are added).

**What the harness does:**

1. Takes `--source <path>`, `--mode {projects|convos}`, `--wing <name>`, `--iterations <n>`, `--report <path>`, `--queries <json>`, `--k <n>`.
2. For each iteration:
   - Creates a fresh `tempfile::TempDir` palace.
   - Constructs `LanceDbPalace::new_with_table`.
   - Starts a custom `tracing` subscriber layer that captures `busy_time` per span (see skeleton below).
   - Runs `Miner::mine` or `ConvoMiner::mine`.
   - Records `Instant::now()` before/after the mine and subtracts.
   - Reads `/proc/self/status:VmHWM` for peak RSS.
   - After ingest, loads `queries.json`, runs each query against the palace, asserts expected ids appear in top-K. Any miss → exit 2.
3. Aggregates per-iteration timings (median + p95), writes JSON report.

**Tracing layer skeleton** (goes in `crates/mempalace-cli/src/bench.rs`):

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

#[derive(Default)]
pub(crate) struct StageTimings {
    inner: Mutex<HashMap<String, Duration>>,
}

impl StageTimings {
    pub(crate) fn record(&self, name: &str, dur: Duration) {
        let mut g = self.inner.lock().expect("stage timings mutex");
        *g.entry(name.to_string()).or_default() += dur;
    }
    pub(crate) fn snapshot(&self) -> HashMap<String, Duration> {
        self.inner.lock().expect("stage timings mutex").clone()
    }
}

pub(crate) struct StageLayer {
    timings: Arc<StageTimings>,
    starts: Mutex<HashMap<Id, (String, Instant)>>,
}

impl StageLayer {
    pub(crate) fn new(timings: Arc<StageTimings>) -> Self {
        Self { timings, starts: Mutex::new(HashMap::new()) }
    }
}

impl<S: Subscriber> Layer<S> for StageLayer {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        let name = attrs.metadata().name().to_string();
        self.starts.lock().expect("starts mutex").insert(id.clone(), (name, Instant::now()));
    }
    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        if let Some((name, start)) = self.starts.lock().expect("starts mutex").remove(&id) {
            self.timings.record(&name, start.elapsed());
        }
    }
}
```

Note on lint config: workspace forbids `expect_used` and `unwrap_used`. These skeleton `expect`s will fail clippy. Replace with `map_err(|_| anyhow::anyhow!(...))` or propagate via `?` in the real code. Kept as `expect` here only for readability of the sketch.

**Instrumented spans to add during Step 1** (zero-behavior-change stubs; real parallelism happens in Step 4):

Add these `tracing::info_span!` calls at exactly these places, named exactly as shown so the bench report is stable:

| Span name     | File                                                          | Wrap around                                        |
|---------------|---------------------------------------------------------------|----------------------------------------------------|
| `walk`        | `crates/mempalace-server/src/ingest.rs:109-115`              | `WalkBuilder::new(...).build()` construction + the for-loop |
| `walk`        | `crates/mempalace-server/src/convo_miner.rs:379`             | `scan_convos(dir)` call                            |
| `read_parse`  | `ingest.rs:158-170` / `convo_miner.rs:395-414`               | `read_to_string` + `chunks_into_buffer` / `normalize` + `chunk_exchanges` |
| `prepass`     | (added in Step 3)                                             | existing-id SELECT                                 |
| `prefilter`   | `lancedb_backend.rs:742-778`                                  | `stream.try_collect().await`                       |
| `embed`       | `lancedb_backend.rs:268-279`                                  | the `guard.embed(texts, Some(64))` call            |
| `arrow`       | `lancedb_backend.rs:800-801`                                  | `build_insert_batch_many`                          |
| `merge_insert`| `lancedb_backend.rs:808-816`                                  | the `builder.execute(reader).await`                |

**Gate:**
```
cargo build --release
cargo test --release
./target/release/mempalace bench ingest --source <fixture> --mode projects --wing bench --iterations 1 --report /tmp/bench.json --queries /tmp/queries.json --k 10
```
Must produce a well-formed JSON report with all span names present and non-zero. Recall gate must pass on the hand-written `queries.json`.

---

## Step 2 — Baseline capture

**Fixtures to generate** (local, not committed):

```bash
mkdir -p ~/.mempalace/bench/fixtures
# Small project fixture: copy a known subtree
cp -r ~/workspace/tardis/poker ~/.mempalace/bench/fixtures/project_small
# Medium project: bigger subtree or tarball
cp -r ~/workspace/tardis/scribe ~/.mempalace/bench/fixtures/project_medium
# Convo fixture: a slice of Claude Code sessions
mkdir -p ~/.mempalace/bench/fixtures/convo_small
cp $(ls ~/.claude/projects/*/*.jsonl | head -50) ~/.mempalace/bench/fixtures/convo_small/
```

**Hand-write queries files.** For each fixture, create 5 queries where you know the answer lives in a specific drawer. Example structure:
```json
[
  {"query": "exclusive file lock daemon", "must_retrieve_any_of": ["<insert id after first ingest>"], "k": 10}
]
```
On the very first run this won't have ids yet — run once, grab the ids of drawers that contain the literal substring via a cli search, then paste them into the queries file. Subsequent runs enforce the gate.

**Commands:**
```bash
cargo build --release
./target/release/mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/project_small \
    --mode projects --wing bench_proj --iterations 5 \
    --report docs/sessions/bench-baseline-project-2026-04-13.json \
    --queries ~/.mempalace/bench/fixtures/queries-project.json --k 10

./target/release/mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/convo_small \
    --mode convos --wing bench_convo --iterations 5 \
    --report docs/sessions/bench-baseline-convo-2026-04-13.json \
    --queries ~/.mempalace/bench/fixtures/queries-convo.json --k 10
```

**Gate:** both reports committed. Median wall time recorded in the Step-7 results doc as the baseline.

---

## Step 3 — Global per-wing pre-pass dedup

**File:** `crates/mempalace-store/src/lancedb_backend.rs`

**Change:** add a new helper and call it once at the start of `add_many` when the feature flag is on.

```rust
// Around line 720, above add_many:

/// Load all existing drawer ids for a given wing. Used by the v2 ingest
/// pipeline to skip embedding work for rows that already exist.
///
/// Scoped to a single wing to keep the scan bounded: largest wing is
/// convo_opencode at ~69k rows, ~250ms. Cross-wing dedup was never a
/// guarantee of the current system.
fn existing_ids_for_wing(&self, wing: &str) -> Result<std::collections::HashSet<String>> {
    let wing_escaped = escape_sql_literal(wing)?;
    let filter = format!("wing = '{wing_escaped}'");
    self.runtime.block_on(async {
        let stream = self
            .table
            .query()
            .only_if(filter)
            .select(lancedb::query::Select::columns(&["id"]))
            .execute()
            .await
            .map_err(|e| PalaceError::Backend(format!("prepass query failed: {e}")))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| PalaceError::Backend(format!("prepass collect failed: {e}")))?;
        let mut set = std::collections::HashSet::with_capacity(batches.iter().map(|b| b.num_rows()).sum());
        for b in &batches {
            for row in 0..b.num_rows() {
                if let Some(id) = read_string(b, "id", row)? {
                    set.insert(id);
                }
            }
        }
        Ok(set)
    })
}
```

**Wire-up for Step 3:** Step 3 does not yet call `existing_ids_for_wing` from inside `add_many` — the caller (miner) will load it once and pass it through a new method. But `add_many` already has per-batch dedup that stays in place. Step 3 *only* adds the helper + a unit test that it returns the correct set for a seeded table.

**Unit test** in `lancedb_backend.rs` test module:
```rust
#[test]
fn existing_ids_for_wing_returns_only_matching_wing() {
    // insert 3 drawers into wing A, 2 into wing B
    // assert existing_ids_for_wing("A").len() == 3
    // assert existing_ids_for_wing("B").len() == 2
    // assert existing_ids_for_wing("C").is_empty()
}
```

**Gate:** `cargo test --release -p mempalace-store` passes. No behavior change yet — ingest path unchanged.

---

## Step 4 — Concrete `LanceDbPalace::bulk_ingest`

**This is the big step.** It adds the three-stage pipeline as a new concrete method. It does not change the `Palace` trait.

**File:** `crates/mempalace-store/src/lancedb_backend.rs`

**Dependencies to add** (`crates/mempalace-store/Cargo.toml`):
```toml
crossbeam-channel = "0.5"
num_cpus = "1"
rayon = "1"
```
Add `rayon` and `num_cpus` to the workspace `[workspace.dependencies]` block in `/Cargo.toml` too, and use `.workspace = true` in the crate.

**Method signature** (added as `impl LanceDbPalace` block, NOT on the trait):

```rust
impl LanceDbPalace {
    /// V2 bulk ingest with parallel pipeline.
    ///
    /// Takes an iterator of drawer records *produced lazily* (so the caller
    /// can stream from a file walker) plus the wing string for pre-pass
    /// dedup. Returns the number of drawers actually written (post-dedup).
    ///
    /// See docs/plans/2026-04-13-ingestion-pipeline-15x-design.md for the
    /// three-stage design. Gated behind MEMPALACE_INGEST_PIPELINE=v2 at the
    /// CLI layer — callers are expected to fall back to add_many when v1.
    pub fn bulk_ingest<I>(&self, wing: &str, records: I) -> Result<usize>
    where
        I: IntoIterator<Item = DrawerRecord> + Send,
        I::IntoIter: Send,
    {
        // 0. Global pre-pass
        let existing = Arc::new(self.existing_ids_for_wing(wing)?);

        // 1. Channels
        let (stage2_tx, stage2_rx) = crossbeam_channel::bounded::<Vec<DrawerRecord>>(4);
        let (stage3_tx, stage3_rx) =
            crossbeam_channel::bounded::<(Vec<DrawerRecord>, Vec<Vec<f32>>)>(4);

        // 2. Embedder pool
        let n_embed = std::cmp::max(1, num_cpus::get() / 8);
        let mut embed_handles = Vec::with_capacity(n_embed);
        for i in 0..n_embed {
            let rx = stage2_rx.clone();
            let tx = stage3_tx.clone();
            let embedder = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2),
            ).map_err(|e| PalaceError::Backend(format!("embedder {i} init failed: {e}")))?;
            embed_handles.push(
                std::thread::Builder::new()
                    .name(format!("mempalace-embed-{i}"))
                    .spawn(move || embed_worker(embedder, rx, tx))
                    .map_err(|e| PalaceError::Backend(format!("spawn embed worker failed: {e}")))?
            );
        }
        drop(stage3_tx); // workers hold their clones; main drops its copy

        // 3. Writer thread
        let writer_handle = {
            let this = self.runtime.handle().clone();
            let table = self.table.clone();
            let schema = build_schema();
            let next_seq = self.next_seq.clone(); // Mutex<i64>, move Arc
            std::thread::Builder::new()
                .name("mempalace-writer".into())
                .spawn(move || writer_loop(this, table, schema, next_seq, stage3_rx))
                .map_err(|e| PalaceError::Backend(format!("spawn writer failed: {e}")))?
        };

        // 4. Stage 1: producer (main thread, rayon inside)
        let producer_result = run_producer(records, existing, stage2_tx);

        // 5. Join
        for h in embed_handles {
            h.join().map_err(|_| PalaceError::Backend("embed worker panic".into()))??;
        }
        let written = writer_handle
            .join()
            .map_err(|_| PalaceError::Backend("writer thread panic".into()))??;

        producer_result?;
        Ok(written)
    }
}
```

**`run_producer`** — Stage 1 wrapper. Chunks the incoming iterator into rayon-parallel batches of file contents, performs pre-pass dedup, groups survivors into 1024-record batches, sends on `stage2_tx`.

```rust
fn run_producer<I: IntoIterator<Item = DrawerRecord>>(
    records: I,
    existing: Arc<HashSet<String>>,
    tx: crossbeam_channel::Sender<Vec<DrawerRecord>>,
) -> Result<()> {
    const BATCH: usize = 1024;
    let mut buf: Vec<DrawerRecord> = Vec::with_capacity(BATCH);
    let mut seen_local: HashSet<String> = HashSet::with_capacity(BATCH);
    for rec in records {
        if existing.contains(&rec.id) || !seen_local.insert(rec.id.clone()) {
            continue;
        }
        buf.push(rec);
        if buf.len() >= BATCH {
            let out = std::mem::replace(&mut buf, Vec::with_capacity(BATCH));
            tx.send(out)
                .map_err(|e| PalaceError::Backend(format!("stage2 send: {e}")))?;
        }
    }
    if !buf.is_empty() {
        tx.send(buf)
            .map_err(|e| PalaceError::Backend(format!("stage2 send final: {e}")))?;
    }
    drop(tx); // signal end of stream
    Ok(())
}
```

**Note on rayon.** The first version does NOT use rayon inside `run_producer` — the records iterator is already streaming from the caller's file walker. Rayon parallelism is added in Step 5 at the miner level (see below) where it applies to `read_to_string + normalize + chunk`, which is the actual CPU work. Step 4 keeps Stage 1 single-threaded to isolate the embed/write parallelism change for cleaner bench comparison.

**`embed_worker`:**

```rust
fn embed_worker(
    mut embedder: TextEmbedding,
    rx: crossbeam_channel::Receiver<Vec<DrawerRecord>>,
    tx: crossbeam_channel::Sender<(Vec<DrawerRecord>, Vec<Vec<f32>>)>,
) -> Result<()> {
    while let Ok(batch) = rx.recv() {
        let texts: Vec<String> = batch.iter().map(|r| r.content.clone()).collect();
        // Pass a large sub-batch hint: fastembed will chunk internally.
        let vectors = embedder
            .embed(texts, Some(1024))
            .map_err(|e| PalaceError::Backend(format!("embed failed: {e}")))?;
        if vectors.len() != batch.len() {
            return Err(PalaceError::Backend(format!(
                "embed count mismatch: {} vectors for {} records",
                vectors.len(),
                batch.len()
            )));
        }
        if tx.send((batch, vectors)).is_err() {
            break; // writer dropped, bail
        }
    }
    Ok(())
}
```

**`writer_loop`** — drains the writer channel, accumulates in 1024-row chunks, calls `merge_insert`:

```rust
fn writer_loop(
    runtime: tokio::runtime::Handle,
    table: lancedb::Table,
    schema: SchemaRef,
    next_seq: Arc<Mutex<i64>>,
    rx: crossbeam_channel::Receiver<(Vec<DrawerRecord>, Vec<Vec<f32>>)>,
) -> Result<usize> {
    let mut total_written = 0usize;
    while let Ok((records, vectors)) = rx.recv() {
        let n = records.len();
        let seqs = {
            let mut g = next_seq
                .lock()
                .map_err(|e| PalaceError::Backend(format!("seq poisoned: {e}")))?;
            let start = *g;
            *g = start.saturating_add(n as i64);
            (0..n as i64).map(|i| start + i).collect::<Vec<_>>()
        };
        let batch = build_insert_batch_many(schema.clone(), &records, vectors, &seqs)?;
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone()),
        );
        runtime.block_on(async {
            let mut builder = table.merge_insert(&["id"]);
            builder.when_not_matched_insert_all();
            builder
                .execute(reader)
                .await
                .map(|_| ())
                .map_err(|e| PalaceError::Backend(format!("writer merge_insert: {e}")))
        })?;
        total_written += n;
    }
    Ok(total_written)
}
```

**Refactoring prerequisite.** `LanceDbPalace` currently stores `next_seq: Mutex<i64>` (line 246). Change to `next_seq: Arc<Mutex<i64>>` so it can be moved into the writer thread. This is a local change; no API impact.

**Unit tests to add** (`lancedb_backend.rs` test module):

1. `bulk_ingest_into_empty_wing_inserts_all` — 100 records, verify count == 100, verify all ids present.
2. `bulk_ingest_is_idempotent` — run twice, verify count == 100 after second call (pre-pass catches them).
3. `bulk_ingest_partial_overlap` — seed 50, bulk-ingest 100 where 50 overlap, verify count == 100.
4. `bulk_ingest_preserves_metadata` — spot-check wing/room/hall for a few rows.
5. `bulk_ingest_drops_in_batch_duplicates` — pass duplicates in the same call, verify only uniques written.

All tests use `InMemoryPalace`... wait. The new method is on `LanceDbPalace` only. Tests must construct a real `LanceDbPalace` in a `TempDir`. Existing tests in the file already do this — copy the pattern.

**Gate:**
```
cargo test --release -p mempalace-store lancedb_backend
./target/release/mempalace bench ingest --source ~/.mempalace/bench/fixtures/project_small \
    --mode projects --wing bench_proj --iterations 5 \
    --report docs/sessions/bench-v2-step4-project-2026-04-13.json \
    --queries ~/.mempalace/bench/fixtures/queries-project.json --k 10 \
    --env MEMPALACE_INGEST_PIPELINE=v2
```
Must show ≥3x speedup over baseline on project fixture. Recall gate = 100%.

---

## Step 5 — CLI miner routing

**Files:**
- `crates/mempalace-server/src/ingest.rs`
- `crates/mempalace-server/src/convo_miner.rs`

**Change:** both `Miner::mine` and `ConvoMiner::mine` check the env var. If `v2`, route through `bulk_ingest`; otherwise use the existing `add_many` path.

**New dispatcher** (pseudocode for `ingest.rs`):

```rust
pub fn mine(&self, root: &Path, palace: &mut dyn Palace) -> Result<IngestStats> {
    if std::env::var("MEMPALACE_INGEST_PIPELINE").ok().as_deref() == Some("v2") {
        self.mine_v2(root, palace)
    } else {
        self.mine_v1(root, palace) // existing code, renamed
    }
}
```

**`mine_v2` shape.**

The challenge: `bulk_ingest` takes `&self` on `LanceDbPalace`, but `Miner::mine` receives `&mut dyn Palace`. We need to downcast — acceptable because the feature flag is only enabled when the caller knows they have a `LanceDbPalace`.

```rust
fn mine_v2(&self, root: &Path, palace: &mut dyn Palace) -> Result<IngestStats> {
    let lance = palace
        .as_any()                         // ← requires adding this method to Palace trait
        .downcast_ref::<LanceDbPalace>()
        .ok_or_else(|| IngestError::Palace(PalaceError::Backend(
            "MEMPALACE_INGEST_PIPELINE=v2 requires LanceDbPalace backend".into()
        )))?;

    // Stage 1 producer: file walker + rayon parallel chunk builder.
    let (records, stats) = self.walk_and_chunk_parallel(root)?;
    // This returns a lazy iterator via a bounded channel, not a Vec.

    let wing = self.options.wing.clone().unwrap_or_default();
    let written = lance
        .bulk_ingest(&wing, records)
        .map_err(IngestError::Palace)?;

    Ok(IngestStats { drawers_written: written, ..stats })
}
```

**`walk_and_chunk_parallel`** uses rayon with `par_bridge` on the walker iterator:

```rust
use rayon::prelude::*;

fn walk_and_chunk_parallel(&self, root: &Path) -> Result<(impl Iterator<Item = DrawerRecord>, IngestStats)> {
    let canonical_root = root.canonicalize()?;
    let walker = WalkBuilder::new(&canonical_root)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .hidden(true)
        .follow_links(false)
        .build();

    let (tx, rx) = crossbeam_channel::bounded::<DrawerRecord>(4096);
    let stats = Arc::new(Mutex::new(IngestStats::default()));
    let options = self.options.clone();
    let root_for_thread = canonical_root.clone();
    let stats_for_thread = stats.clone();

    std::thread::spawn(move || {
        walker
            .flatten()
            .filter(|e| e.path().is_file())
            .par_bridge()
            .for_each(|entry| {
                let path = entry.path();
                // eligibility checks (skip files, binary, size, symlink)
                // read_to_string
                // chunk_text
                // emit DrawerRecords via tx.send
                // mutex-update stats
            });
        drop(tx);
    });

    Ok((rx.into_iter(), Arc::try_unwrap(stats).unwrap_or_default().into_inner().unwrap_or_default()))
    // ^ stats handling above is awkward; real impl uses atomic counters instead of a Mutex to avoid the unwrap gymnastics
}
```

**Real impl note:** use `std::sync::atomic::{AtomicUsize, Ordering}` for each stats field. Drop the `Mutex<IngestStats>`. Cleaner and the clippy gates will be happy.

**Same pattern** for `convo_miner.rs::mine` with `ExtractMode` branching preserved.

**Palace trait `as_any` addition.** `crates/mempalace-store/src/palace.rs` — add `fn as_any(&self) -> &dyn std::any::Any` to the `Palace` trait, default impl `{ self }` on each concrete type (`InMemoryPalace`, `LanceDbPalace`). This is the standard Rust downcast pattern.

**Gate:**
```
cargo test --release
MEMPALACE_INGEST_PIPELINE=v2 ./target/release/mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/project_small --mode projects \
    --wing bench_proj --iterations 5 \
    --report docs/sessions/bench-v2-step5-project.json \
    --queries ~/.mempalace/bench/fixtures/queries-project.json --k 10

MEMPALACE_INGEST_PIPELINE=v2 ./target/release/mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/convo_small --mode convos \
    --wing bench_convo --iterations 5 \
    --report docs/sessions/bench-v2-step5-convo.json \
    --queries ~/.mempalace/bench/fixtures/queries-convo.json --k 10
```
Both must show ≥5x speedup over baseline. Recall = 100% on both fixtures.

---

## Step 6 — Tuning sweep

**Single script, run locally, commit results.**

`scripts/tune-ingest.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

export MEMPALACE_INGEST_PIPELINE=v2
FIX=~/.mempalace/bench/fixtures/project_medium
Q=~/.mempalace/bench/fixtures/queries-project.json
OUT=docs/sessions/bench-tune-$(date +%Y-%m-%d).json

for N in 1 2 4 8; do
    for OMP in 32 16 8 4; do
        echo "N=$N OMP=$OMP"
        MEMPALACE_EMBED_WORKERS=$N OMP_NUM_THREADS=$OMP \
            ./target/release/mempalace bench ingest \
            --source "$FIX" --mode projects --wing tune \
            --iterations 3 \
            --report "/tmp/tune-$N-$OMP.json" \
            --queries "$Q" --k 10
    done
done

# Aggregate /tmp/tune-*.json into $OUT (pick lowest median)
python3 scripts/aggregate-tune.py /tmp/tune-*.json > "$OUT"
```

**Changes to code:** `bulk_ingest` reads `MEMPALACE_EMBED_WORKERS` env var and uses it instead of `num_cpus::get() / 8` when set. This is a tuning-only knob; the default stays the computed value.

**After the sweep:** update the default constant `N_EMBEDDERS_DEFAULT` and `OMP_DEFAULT` in `lancedb_backend.rs` to match the winner. Commit the tune report JSON.

**Gate:** tune report committed with a clear winner. Default constants updated.

---

## Step 7 — Final verification

**Run the full before/after comparison.**

```bash
# Baseline is already captured from Step 2. Rerun v2 with final constants.
MEMPALACE_INGEST_PIPELINE=v2 ./target/release/mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/project_medium --mode projects \
    --wing final --iterations 10 \
    --report docs/sessions/bench-v2-final-project-2026-04-13.json \
    --queries ~/.mempalace/bench/fixtures/queries-project.json --k 20

MEMPALACE_INGEST_PIPELINE=v2 ./target/release/mempalace bench ingest \
    --source ~/.mempalace/bench/fixtures/convo_medium --mode convos \
    --wing final --iterations 10 \
    --report docs/sessions/bench-v2-final-convo-2026-04-13.json \
    --queries ~/.mempalace/bench/fixtures/queries-convo.json --k 20
```

**Write the results report:** `docs/sessions/2026-04-13-ingestion-v2-results.md`. Include:
- Baseline median and p95 for both fixtures.
- V2 median and p95 for both fixtures.
- Per-stage breakdown (walk, read_parse, prepass, embed, arrow, merge_insert) in both.
- Peak RSS baseline vs. v2.
- Recall@20 on both fixtures (must be 100%).
- Observed speedup vs. the 15x target. Be honest if it's lower.
- Named follow-ups (quantized model, tokenize-in-stage-1, etc.) if the target isn't met.

**Final gate:**
- `cargo test --release` passes.
- `cargo clippy --release --all-targets -- -D warnings` passes (workspace lints are strict).
- Recall@20 = 100% on both fixtures.
- V2 wall time ≤ baseline / 5 (soft floor; target is 15x but we ship at ≥5x).

**Merge to main** only when all gates pass and the results report is reviewed.

---

## Files touched summary

**New files:**
- `docs/plans/2026-04-13-ingestion-pipeline-15x-design.md` (Thing A design) ✓ written
- `docs/plans/2026-04-13-method-of-loci-gis-sketch.md` (Thing B sketch) ✓ written
- `docs/plans/2026-04-13-ingestion-pipeline-15x-plan.md` (this file) ✓ written
- `crates/mempalace-cli/src/bench.rs`
- `scripts/tune-ingest.sh`
- `scripts/aggregate-tune.py`
- `docs/sessions/bench-baseline-project-2026-04-13.json`
- `docs/sessions/bench-baseline-convo-2026-04-13.json`
- `docs/sessions/bench-v2-*.json` (several)
- `docs/sessions/2026-04-13-ingestion-v2-results.md`

**Modified files:**
- `Cargo.toml` — add `crossbeam-channel`, `num_cpus`, `rayon` to `[workspace.dependencies]`
- `crates/mempalace-store/Cargo.toml` — pull in the three new deps
- `crates/mempalace-store/src/palace.rs` — add `as_any` to the `Palace` trait
- `crates/mempalace-store/src/lancedb_backend.rs` — `existing_ids_for_wing`, `bulk_ingest`, `embed_worker`, `writer_loop`, `run_producer`, `Arc<Mutex<i64>>` seq refactor, default constants, tests
- `crates/mempalace-server/src/ingest.rs` — v1/v2 dispatcher, `mine_v2`, rayon-parallel walker
- `crates/mempalace-server/src/convo_miner.rs` — same dispatcher pattern
- `crates/mempalace-cli/src/main.rs` — register `bench ingest` subcommand

**Untouched (explicit non-goals):**
- Arrow schema (`build_schema`)
- `knowledge_graph.sqlite3` / any sqlite code
- MCP transport, daemon, hooks
- Wing config
- Retrieval code paths (`Palace::search`, MCP `mempalace_search` tool)

---

## What the executor should do BEFORE writing any Rust

1. Re-read `docs/plans/2026-04-13-ingestion-pipeline-15x-design.md` in full.
2. Verify `LanceDbPalace::new_with_table` still refuses tokio runtimes (`lancedb_backend.rs:102-108`). If that changed, the whole thread-based pipeline needs re-checking.
3. Verify `Palace::add_many` trait signature is still `fn add_many(&mut self, ...)`. If it became `&self` already, the `as_any` downcast dance is unnecessary.
4. Verify fastembed version is still `5` in `Cargo.toml`. If it bumped, re-check `embed(&mut self, ...)` in the new version's source.
5. Run the current tests once on a clean `main` to confirm the starting point is green: `cargo test --release`.
6. Create the branch: `git checkout -b ingest-pipeline-v2`.
7. Proceed with Step 1.
