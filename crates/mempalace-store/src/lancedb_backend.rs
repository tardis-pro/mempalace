//! Persistent [`Palace`] backend on top of `lancedb` + `fastembed`.
//!
//! This module provides [`LanceDbPalace`], a real semantic vector store
//! that replaces the in-memory reference backend for production use. It is
//! gated behind the `lancedb-backend` Cargo feature (on by default).
//!
//! ## Design notes
//!
//! - The [`Palace`] trait is **synchronous**, but `lancedb` is fully async.
//!   We solve this by owning a dedicated `tokio::runtime::Runtime` inside
//!   the struct.  Trait methods call `rt_block_on(...)`, which detects
//!   whether the current thread is already inside a tokio runtime (e.g.
//!   the daemon's async MCP handler) and wraps the call in
//!   `tokio::task::block_in_place` when it is.  This avoids the classic
//!   "Cannot start a runtime from within a runtime" panic.
//!
//! - Embeddings use `fastembed` 5 with `AllMiniLML6V2` (384 dim).
//!
//! - Distance metric is **cosine**, so similarity is `1.0 - distance`.
//!
//! - The table schema keeps one Arrow row per drawer. `insert_seq` preserves
//!   insertion order for `list` / `list_filtered` (lancedb has no implicit
//!   row order guarantee).
//!
//! - Filter values are sanitised before being embedded in a SQL `where`
//!   clause: control characters cause a hard error, single quotes are
//!   escaped by doubling (`'` → `''`). This is the same rule Postgres and
//!   SQLite use for single-quoted literals.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use fs2::FileExt;
use futures::TryStreamExt;
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::index::IndexType;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, DistanceType, Table};
use tokio::runtime::{Handle, Runtime};
use tracing::debug;
use tracing::warn;

use crate::palace::{
    DrawerMetadata, DrawerRecord, Palace, PalaceError, Result, SearchFilter, SearchResult,
};

pub const EMBEDDING_DIM: i32 = 384;
pub const DEFAULT_TABLE_NAME: &str = "mempalace_drawers";

const ORT_DYLIB_SEARCH_PATHS: &[&str] = &["/home/pronit/.local/lib/onnxruntime/libonnxruntime.so"];

fn ensure_ort_dylib() {
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }
    for path in ORT_DYLIB_SEARCH_PATHS {
        if std::path::Path::new(path).exists() {
            std::env::set_var("ORT_DYLIB_PATH", path);
            return;
        }
    }
}

fn init_embedder() -> Result<(TextEmbedding, bool)> {
    ensure_ort_dylib();

    let cuda_ep = ort::ep::CUDA::default().with_device_id(0).build();
    let cpu_ep = ort::ep::CPU::default().build();
    let opts = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
        .with_execution_providers(vec![cuda_ep, cpu_ep]);
    match TextEmbedding::try_new(opts) {
        Ok(emb) => {
            debug!("initialized embedder with CUDA + CPU fallback");
            Ok((emb, true))
        }
        Err(e) => {
            warn!("CUDA init failed ({e}), trying CPU-only");
            let cpu_opts = InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_execution_providers(vec![ort::ep::CPU::default().build()]);
            let emb = TextEmbedding::try_new(cpu_opts)
                .map_err(|e| PalaceError::Backend(format!("embedder init failed: {e}")))?;
            Ok((emb, false))
        }
    }
}

/// Real, persistent [`Palace`] backed by `lancedb` + `fastembed`.
/// Minimum row count before we build an IVF_PQ vector index. Below this
/// threshold, brute-force is fast enough and training data is too sparse
/// for meaningful partitions.
const VECTOR_INDEX_MIN_ROWS: usize = 1_000;

pub struct LanceDbPalace {
    runtime: Runtime,
    connection: Connection,
    table: Table,
    embedder: Mutex<TextEmbedding>,
    next_seq: Mutex<i64>,
    schema: SchemaRef,
    table_name: String,
    has_vector_index: bool,
    using_gpu: bool,
    pending_write: Option<tokio::task::JoinHandle<std::result::Result<(), String>>>,
    #[allow(dead_code)]
    lock_file: File,
}

impl std::fmt::Debug for LanceDbPalace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceDbPalace")
            .field("table_name", &self.table_name)
            .finish()
    }
}

impl Drop for LanceDbPalace {
    fn drop(&mut self) {
        let _ = self.drain_pending_write();
        let _ = FileExt::unlock(&self.lock_file);
    }
}

impl LanceDbPalace {
    /// Open (or create) a palace at `path` using the default table name.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Self::new_with_table(path, DEFAULT_TABLE_NAME)
    }

    /// Open (or create) a palace at `path` with a specific table name.
    ///
    /// Acquires an exclusive file lock at `<path>/.lock` so that at most one
    /// `LanceDbPalace` handle can exist per directory across processes on the
    /// same host. If the lock is held by another process, retries for up to
    /// 5 seconds before returning `PalaceError::Backend("palace is locked …")`.
    pub fn new_with_table(path: impl AsRef<Path>, table_name: &str) -> Result<Self> {
        if Handle::try_current().is_ok() {
            return Err(PalaceError::Backend(
                "LanceDbPalace cannot be constructed from within a running tokio runtime; \
                 use PalaceAsync-style callers or spawn a dedicated thread"
                    .to_string(),
            ));
        }

        // Ensure the palace dir exists, then acquire the exclusive lock.
        let path_ref = path.as_ref();
        std::fs::create_dir_all(path_ref).map_err(|e| {
            PalaceError::Backend(format!(
                "failed to create palace dir {}: {e}",
                path_ref.display()
            ))
        })?;
        let lock_path = path_ref.join(".lock");
        let lock_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| {
                PalaceError::Backend(format!(
                    "failed to open palace lockfile {}: {e}",
                    lock_path.display()
                ))
            })?;

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut attempt = 0u32;
        loop {
            match FileExt::try_lock_exclusive(&lock_file) {
                Ok(()) => break,
                Err(_) if Instant::now() < deadline => {
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(200));
                    if attempt % 5 == 0 {
                        debug!(
                            path = %lock_path.display(),
                            "waiting for palace lock (another mempalace process holds it)"
                        );
                    }
                }
                Err(e) => {
                    return Err(PalaceError::Backend(format!(
                        "palace is locked by another mempalace process ({}): {e}. \
                         stop the running daemon or mining job before opening this palace.",
                        lock_path.display()
                    )));
                }
            }
        }
        // Record our PID so humans grepping lsof know who holds it. Ignored on failure.
        let _ = std::fs::write(path_ref.join(".lock.pid"), std::process::id().to_string());

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .thread_name("mempalace-lancedb")
            .build()
            .map_err(|e| PalaceError::Backend(format!("failed to build tokio runtime: {e}")))?;

        let path_str = path_ref
            .to_str()
            .ok_or_else(|| PalaceError::Backend("lancedb path is not valid UTF-8".to_string()))?
            .to_string();
        let table_name_owned = table_name.to_string();

        let schema = build_schema();

        let (connection, table, indices) = runtime.block_on(async {
            let conn = lancedb::connect(&path_str)
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("lancedb connect failed: {e}")))?;

            let existing = conn
                .table_names()
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("list tables failed: {e}")))?;

            let table = if existing.iter().any(|n| n == &table_name_owned) {
                conn.open_table(&table_name_owned)
                    .execute()
                    .await
                    .map_err(|e| PalaceError::Backend(format!("open table failed: {e}")))?
            } else {
                conn.create_empty_table(&table_name_owned, schema.clone())
                    .execute()
                    .await
                    .map_err(|e| PalaceError::Backend(format!("create table failed: {e}")))?
            };

            // Ensure a BTree scalar index exists on the `id` column.
            // `merge_insert` uses this for O(log n) duplicate-key lookups
            // instead of a full-table scan. Idempotent: we skip if a scalar
            // index on `id` already exists. One-time ~seconds cost on
            // existing data; free on subsequent opens.
            let indices = table
                .list_indices()
                .await
                .map_err(|e| PalaceError::Backend(format!("list_indices failed: {e}")))?;
            let has_id_index = indices
                .iter()
                .any(|cfg| cfg.columns.len() == 1 && cfg.columns[0] == "id");
            if !has_id_index {
                debug!("creating BTree scalar index on id column (first run)");
                table
                    .create_index(
                        &["id"],
                        lancedb::index::Index::BTree(
                            lancedb::index::scalar::BTreeIndexBuilder::default(),
                        ),
                    )
                    .execute()
                    .await
                    .map_err(|e| PalaceError::Backend(format!("create id index failed: {e}")))?;
            }

            // BTree scalar index on `wing` for fast metadata filtering in
            // search (prefilter) and load_existing_ids_for_wing pre-pass.
            let has_wing_index = indices
                .iter()
                .any(|cfg| cfg.columns.len() == 1 && cfg.columns[0] == "wing");
            if !has_wing_index {
                debug!("creating BTree scalar index on wing column (first run)");
                table
                    .create_index(
                        &["wing"],
                        lancedb::index::Index::BTree(
                            lancedb::index::scalar::BTreeIndexBuilder::default(),
                        ),
                    )
                    .execute()
                    .await
                    .map_err(|e| PalaceError::Backend(format!("create wing index failed: {e}")))?;
            }

            Ok::<_, PalaceError>((conn, table, indices))
        })?;

        let has_vector_index = indices.iter().any(|cfg| {
            matches!(
                cfg.index_type,
                IndexType::IvfPq
                    | IndexType::IvfHnswSq
                    | IndexType::IvfHnswPq
                    | IndexType::IvfFlat
                    | IndexType::IvfSq
            )
        });

        let (embedder, using_gpu) = init_embedder()?;

        let next_seq = runtime.block_on(async { scan_max_seq(&table).await })?;

        if !has_vector_index {
            let row_count = runtime.block_on(async {
                table
                    .count_rows(None)
                    .await
                    .map_err(|e| PalaceError::Backend(format!("count_rows failed: {e}")))
            })?;
            if row_count >= VECTOR_INDEX_MIN_ROWS {
                debug!(
                    row_count,
                    "building IVF_PQ vector index (first run on existing data)"
                );
                runtime.block_on(async {
                    table
                        .create_index(
                            &["vector"],
                            lancedb::index::Index::IvfPq(
                                IvfPqIndexBuilder::default()
                                    .distance_type(DistanceType::Cosine)
                                    .num_partitions(std::cmp::max(1, (row_count / 4096) as u32))
                                    .num_sub_vectors(EMBEDDING_DIM as u32 / 16),
                            ),
                        )
                        .execute()
                        .await
                        .map_err(|e| {
                            PalaceError::Backend(format!("create vector index failed: {e}"))
                        })
                })?;
            }
        }

        let has_vector_index = has_vector_index || {
            runtime.block_on(async { table.count_rows(None).await.unwrap_or(0) })
                >= VECTOR_INDEX_MIN_ROWS
        };

        debug!(
            table = %table_name_owned,
            next_seq = next_seq,
            has_vector_index,
            using_gpu,
            "LanceDbPalace opened"
        );

        Ok(Self {
            runtime,
            connection,
            table,
            embedder: Mutex::new(embedder),
            next_seq: Mutex::new(next_seq),
            schema,
            table_name: table_name_owned,
            has_vector_index,
            using_gpu,
            pending_write: None,
            lock_file,
        })
    }

    /// Run a future on the palace's dedicated runtime.  When the current
    /// thread is already inside a tokio runtime (e.g. the daemon's async MCP
    /// handler), wraps the call in [`tokio::task::block_in_place`] so the
    /// outer scheduler can move work off this thread.  Otherwise calls
    /// `block_on` directly (CLI / mining paths that run before any runtime).
    fn rt_block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        if Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| self.runtime.block_on(f))
        } else {
            self.runtime.block_on(f)
        }
    }

    fn drain_pending_write(&mut self) -> Result<()> {
        if let Some(handle) = self.pending_write.take() {
            self.rt_block_on(handle)
                .map_err(|e| PalaceError::Backend(format!("pending write task panicked: {e}")))?
                .map_err(|e| PalaceError::Backend(e))?;
        }
        Ok(())
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut guard = self
            .embedder
            .lock()
            .map_err(|e| PalaceError::Backend(format!("embedder mutex poisoned: {e}")))?;
        let out = guard
            .embed(vec![text.to_string()], None)
            .map_err(|e| PalaceError::Backend(format!("fastembed embed failed: {e}")))?;
        out.into_iter()
            .next()
            .ok_or_else(|| PalaceError::Backend("fastembed returned empty output".to_string()))
    }

    /// Embed a batch of texts in one fastembed call. Amortises tokenizer +
    /// ONNX session overhead across the whole batch — much faster than
    /// calling [`Self::embed_one`] per record during bulk ingest.
    fn embed_many(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut guard = self
            .embedder
            .lock()
            .map_err(|e| PalaceError::Backend(format!("embedder mutex poisoned: {e}")))?;
        // Sub-batch hint for fastembed's internal chunking. 256 is big
        // enough to amortise tokenizer + session overhead across a flush
        // (vs. the previous 64 which turned each 1024-row flush into 16
        // tiny ONNX calls). MiniLM-L6 at batch=256 fits comfortably in
        // ORT's arena on a 32-core box.
        guard
            .embed(texts, Some(256))
            .map_err(|e| PalaceError::Backend(format!("fastembed batch embed failed: {e}")))
    }

    fn bump_seq_many(&self, n: usize) -> Result<Vec<i64>> {
        let mut g = self
            .next_seq
            .lock()
            .map_err(|e| PalaceError::Backend(format!("seq mutex poisoned: {e}")))?;
        let start = *g;
        *g = start.saturating_add(n as i64);
        Ok((0..n as i64).map(|i| start + i).collect())
    }

    fn bump_seq(&self) -> Result<i64> {
        let mut g = self
            .next_seq
            .lock()
            .map_err(|e| PalaceError::Backend(format!("seq mutex poisoned: {e}")))?;
        let v = *g;
        *g = v.saturating_add(1);
        Ok(v)
    }
}

// ── Arrow schema ────────────────────────────────────────────────────────

/// Build the canonical Arrow schema used by [`LanceDbPalace`].
///
/// Exposed as `pub(crate)` so the unit test can assert on it without
/// instantiating the embedder (which would download model files).
pub(crate) fn build_schema() -> SchemaRef {
    let vector_field = Field::new("item", DataType::Float32, true);
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(vector_field), EMBEDDING_DIM),
            false,
        ),
        Field::new("wing", DataType::Utf8, true),
        Field::new("room", DataType::Utf8, true),
        Field::new("hall", DataType::Utf8, true),
        Field::new("source_file", DataType::Utf8, true),
        Field::new("date", DataType::Utf8, true),
        Field::new("importance", DataType::Float64, true),
        Field::new("extra_json", DataType::Utf8, true),
        Field::new("insert_seq", DataType::Int64, false),
    ]))
}

// ── SQL-safe filter building ────────────────────────────────────────────

/// Escape a string value for inclusion as a SQL single-quoted literal.
///
/// Rejects any string containing control characters (`\x00..=\x1f` or
/// `\x7f`). Otherwise doubles single quotes.
pub(crate) fn escape_sql_literal(value: &str) -> Result<String> {
    if value.chars().any(|c| c.is_control()) {
        return Err(PalaceError::Backend(format!(
            "filter value contains control characters: {value:?}"
        )));
    }
    Ok(value.replace('\'', "''"))
}

fn build_where_clause(filter: &SearchFilter) -> Result<Option<String>> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(w) = filter.wing.as_ref() {
        parts.push(format!("wing = '{}'", escape_sql_literal(w)?));
    }
    if let Some(r) = filter.room.as_ref() {
        parts.push(format!("room = '{}'", escape_sql_literal(r)?));
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parts.join(" AND ")))
    }
}

// ── Helpers for reading lancedb rows back into DrawerRecord ────────────

async fn scan_max_seq(table: &Table) -> Result<i64> {
    let stream = table
        .query()
        .select(lancedb::query::Select::columns(&["insert_seq"]))
        .execute()
        .await
        .map_err(|e| PalaceError::Backend(format!("scan insert_seq failed: {e}")))?;
    let batches: Vec<RecordBatch> = stream
        .try_collect()
        .await
        .map_err(|e| PalaceError::Backend(format!("collect insert_seq failed: {e}")))?;
    let mut max_seen: i64 = -1;
    for batch in batches {
        let col = batch
            .column_by_name("insert_seq")
            .ok_or_else(|| PalaceError::Backend("insert_seq column missing".to_string()))?;
        let arr = col
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PalaceError::Backend("insert_seq not Int64".to_string()))?;
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                let v = arr.value(i);
                if v > max_seen {
                    max_seen = v;
                }
            }
        }
    }
    Ok(max_seen + 1)
}

fn read_string(batch: &RecordBatch, col: &str, row: usize) -> Result<Option<String>> {
    let Some(arr) = batch.column_by_name(col) else {
        return Ok(None);
    };
    let s = arr
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| PalaceError::Backend(format!("column `{col}` is not Utf8")))?;
    if s.is_null(row) {
        Ok(None)
    } else {
        Ok(Some(s.value(row).to_string()))
    }
}

fn read_f64(batch: &RecordBatch, col: &str, row: usize) -> Result<Option<f64>> {
    let Some(arr) = batch.column_by_name(col) else {
        return Ok(None);
    };
    let f = arr
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PalaceError::Backend(format!("column `{col}` is not Float64")))?;
    if f.is_null(row) {
        Ok(None)
    } else {
        Ok(Some(f.value(row)))
    }
}

fn read_i64(batch: &RecordBatch, col: &str, row: usize) -> Result<Option<i64>> {
    let Some(arr) = batch.column_by_name(col) else {
        return Ok(None);
    };
    let f = arr
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| PalaceError::Backend(format!("column `{col}` is not Int64")))?;
    if f.is_null(row) {
        Ok(None)
    } else {
        Ok(Some(f.value(row)))
    }
}

fn row_to_record(batch: &RecordBatch, row: usize) -> Result<DrawerRecord> {
    let id = read_string(batch, "id", row)?
        .ok_or_else(|| PalaceError::Backend("row missing id".to_string()))?;
    let content = read_string(batch, "content", row)?
        .ok_or_else(|| PalaceError::Backend("row missing content".to_string()))?;

    let extra = match read_string(batch, "extra_json", row)? {
        Some(s) if !s.is_empty() => serde_json::from_str(&s)
            .map_err(|e| PalaceError::Backend(format!("extra_json deserialize failed: {e}")))?,
        _ => Default::default(),
    };

    let metadata = DrawerMetadata {
        wing: read_string(batch, "wing", row)?,
        room: read_string(batch, "room", row)?,
        hall: read_string(batch, "hall", row)?,
        source_file: read_string(batch, "source_file", row)?,
        date: read_string(batch, "date", row)?,
        importance: read_f64(batch, "importance", row)?,
        extra,
    };

    Ok(DrawerRecord {
        id,
        content,
        metadata,
    })
}

fn row_similarity(batch: &RecordBatch, row: usize) -> Result<f64> {
    let Some(col) = batch.column_by_name("_distance") else {
        return Ok(0.0);
    };
    let arr = col
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PalaceError::Backend("_distance not Float32".to_string()))?;
    if arr.is_null(row) {
        return Ok(0.0);
    }
    // Cosine distance is in [0, 2]; similarity = 1 - distance puts it in [-1, 1].
    let distance = f64::from(arr.value(row));
    Ok(1.0 - distance)
}

fn build_insert_batch_many(
    schema: SchemaRef,
    records: Vec<DrawerRecord>,
    vectors: Vec<Vec<f32>>,
    seqs: &[i64],
) -> Result<RecordBatch> {
    if records.len() != vectors.len() || records.len() != seqs.len() {
        return Err(PalaceError::Backend(format!(
            "build_insert_batch_many length mismatch: records={}, vectors={}, seqs={}",
            records.len(),
            vectors.len(),
            seqs.len()
        )));
    }
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != EMBEDDING_DIM as usize {
            return Err(PalaceError::Backend(format!(
                "embedding at row {i} has wrong dim: got {}, expected {}",
                v.len(),
                EMBEDDING_DIM
            )));
        }
    }

    let mut ids: Vec<String> = Vec::with_capacity(records.len());
    let mut contents: Vec<String> = Vec::with_capacity(records.len());
    let mut wings: Vec<Option<String>> = Vec::with_capacity(records.len());
    let mut rooms: Vec<Option<String>> = Vec::with_capacity(records.len());
    let mut halls: Vec<Option<String>> = Vec::with_capacity(records.len());
    let mut source_files: Vec<Option<String>> = Vec::with_capacity(records.len());
    let mut dates: Vec<Option<String>> = Vec::with_capacity(records.len());
    let mut importances: Vec<Option<f64>> = Vec::with_capacity(records.len());
    let mut extras: Vec<Option<String>> = Vec::with_capacity(records.len());

    for r in records {
        ids.push(r.id);
        contents.push(r.content);
        wings.push(r.metadata.wing);
        rooms.push(r.metadata.room);
        halls.push(r.metadata.hall);
        source_files.push(r.metadata.source_file);
        dates.push(r.metadata.date);
        importances.push(r.metadata.importance);
        let extra =
            if r.metadata.extra.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&r.metadata.extra).map_err(|e| {
                    PalaceError::Backend(format!("extra_json serialize failed: {e}"))
                })?)
            };
        extras.push(extra);
    }

    let id = Arc::new(StringArray::from(ids)) as Arc<dyn Array>;
    let content = Arc::new(StringArray::from(contents)) as Arc<dyn Array>;

    let vector_rows: Vec<Option<Vec<Option<f32>>>> = vectors
        .into_iter()
        .map(|v| Some(v.into_iter().map(Some).collect()))
        .collect();
    let vector_array = FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >(vector_rows.into_iter(), EMBEDDING_DIM);
    let vector: Arc<dyn Array> = Arc::new(vector_array);

    let wing = Arc::new(StringArray::from(wings)) as Arc<dyn Array>;
    let room = Arc::new(StringArray::from(rooms)) as Arc<dyn Array>;
    let hall = Arc::new(StringArray::from(halls)) as Arc<dyn Array>;
    let source_file = Arc::new(StringArray::from(source_files)) as Arc<dyn Array>;
    let date = Arc::new(StringArray::from(dates)) as Arc<dyn Array>;
    let importance = Arc::new(Float64Array::from(importances)) as Arc<dyn Array>;
    let extra_json = Arc::new(StringArray::from(extras)) as Arc<dyn Array>;

    let insert_seq = Arc::new(Int64Array::from(seqs.to_vec())) as Arc<dyn Array>;

    RecordBatch::try_new(
        schema,
        vec![
            id,
            content,
            vector,
            wing,
            room,
            hall,
            source_file,
            date,
            importance,
            extra_json,
            insert_seq,
        ],
    )
    .map_err(|e| PalaceError::Backend(format!("build record batch failed: {e}")))
}

fn build_insert_batch(
    schema: SchemaRef,
    record: &DrawerRecord,
    vector: Vec<f32>,
    seq: i64,
) -> Result<RecordBatch> {
    if vector.len() != EMBEDDING_DIM as usize {
        return Err(PalaceError::Backend(format!(
            "embedding has wrong dim: got {}, expected {}",
            vector.len(),
            EMBEDDING_DIM
        )));
    }

    let id = Arc::new(StringArray::from(vec![record.id.clone()])) as Arc<dyn Array>;
    let content = Arc::new(StringArray::from(vec![record.content.clone()])) as Arc<dyn Array>;

    let vector_array =
        FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
            std::iter::once(Some(vector.into_iter().map(Some).collect::<Vec<_>>())),
            EMBEDDING_DIM,
        );
    let vector: Arc<dyn Array> = Arc::new(vector_array);

    let wing = Arc::new(StringArray::from(vec![record.metadata.wing.clone()])) as Arc<dyn Array>;
    let room = Arc::new(StringArray::from(vec![record.metadata.room.clone()])) as Arc<dyn Array>;
    let hall = Arc::new(StringArray::from(vec![record.metadata.hall.clone()])) as Arc<dyn Array>;
    let source_file =
        Arc::new(StringArray::from(vec![record.metadata.source_file.clone()])) as Arc<dyn Array>;
    let date = Arc::new(StringArray::from(vec![record.metadata.date.clone()])) as Arc<dyn Array>;
    let importance =
        Arc::new(Float64Array::from(vec![record.metadata.importance])) as Arc<dyn Array>;

    let extra_json_str = if record.metadata.extra.is_empty() {
        None
    } else {
        Some(
            serde_json::to_string(&record.metadata.extra)
                .map_err(|e| PalaceError::Backend(format!("extra_json serialize failed: {e}")))?,
        )
    };
    let extra_json = Arc::new(StringArray::from(vec![extra_json_str])) as Arc<dyn Array>;

    let insert_seq = Arc::new(Int64Array::from(vec![seq])) as Arc<dyn Array>;

    RecordBatch::try_new(
        schema,
        vec![
            id,
            content,
            vector,
            wing,
            room,
            hall,
            source_file,
            date,
            importance,
            extra_json,
            insert_seq,
        ],
    )
    .map_err(|e| PalaceError::Backend(format!("build record batch failed: {e}")))
}

impl Palace for LanceDbPalace {
    fn count(&self) -> Result<usize> {
        self.rt_block_on(async {
            self.table
                .count_rows(None)
                .await
                .map_err(|e| PalaceError::Backend(format!("count_rows failed: {e}")))
        })
    }

    fn add(&mut self, record: DrawerRecord) -> Result<()> {
        // Pre-check for duplicate id — lancedb does not enforce primary keys.
        let id_escaped = escape_sql_literal(&record.id)?;
        let filter = format!("id = '{id_escaped}'");
        let existing = self.rt_block_on(async {
            self.table
                .count_rows(Some(filter.clone()))
                .await
                .map_err(|e| PalaceError::Backend(format!("duplicate check failed: {e}")))
        })?;
        if existing > 0 {
            return Err(PalaceError::Duplicate(record.id));
        }

        let vector = self.embed_one(&record.content)?;
        let seq = self.bump_seq()?;
        let schema = self.schema.clone();
        let batch = build_insert_batch(schema.clone(), &record, vector, seq)?;

        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
        );

        self.rt_block_on(async {
            self.table
                .add(reader)
                .execute()
                .await
                .map(|_| ())
                .map_err(|e| PalaceError::Backend(format!("add failed: {e}")))
        })
    }

    /// Bulk insert with batched embedding and a single Arrow write.
    ///
    /// Three-stage dedup pipeline:
    /// 1. **In-batch dedup** collapses any duplicate ids the caller passed
    ///    in a single batch (merge_insert's behavior on duplicate source
    ///    keys is undefined per the lancedb docs).
    /// 2. **Id prefilter** queries the table with `id IN (...)` using the
    ///    BTree scalar index built on the `id` column in [`new_with_table`]
    ///    — O(batch_size × log N) key lookups. Rows whose id already exists
    ///    are dropped *before* embedding. This is the big win: on a re-mine
    ///    of fully-indexed data we do zero ONNX forward passes.
    /// 3. **merge_insert safety net** handles the race where another writer
    ///    inserts the same id between our prefilter SELECT and our INSERT.
    ///    With `when_matched` defaulting to no-op and `when_not_matched_insert_all`
    ///    enabled, matched rows are silently dropped.
    ///
    /// The old implementation did a full-table SELECT for dedup and
    /// embedded every row regardless. With a 300k-drawer palace that
    /// turned ingest quadratic; this version scales linearly with *new*
    /// rows and is free for re-mines.
    fn add_many(&mut self, records: Vec<DrawerRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.drain_pending_write()?;

        // Stage 1: in-batch dedup.
        let mut seen_in_batch: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(records.len());
        let mut staged: Vec<DrawerRecord> = Vec::with_capacity(records.len());
        for r in records {
            if seen_in_batch.insert(r.id.clone()) {
                staged.push(r);
            }
        }
        if staged.is_empty() {
            return Ok(());
        }

        // Stage 2: id prefilter. Fast now that the BTree index exists.
        let id_literals: Vec<String> = staged
            .iter()
            .map(|r| escape_sql_literal(&r.id))
            .collect::<Result<Vec<_>>>()?;
        let in_clause = id_literals
            .iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(",");
        let filter = format!("id IN ({in_clause})");

        let existing_ids: std::collections::HashSet<String> = self.rt_block_on(async {
            let stream = self
                .table
                .query()
                .only_if(filter)
                .select(lancedb::query::Select::columns(&["id"]))
                .execute()
                .await
                .map_err(|e| {
                    PalaceError::Backend(format!("add_many prefilter query failed: {e}"))
                })?;
            let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| {
                PalaceError::Backend(format!("add_many prefilter collect failed: {e}"))
            })?;
            let mut set = std::collections::HashSet::new();
            for b in &batches {
                for row in 0..b.num_rows() {
                    if let Some(id) = read_string(b, "id", row)? {
                        set.insert(id);
                    }
                }
            }
            Ok::<_, PalaceError>(set)
        })?;

        let fresh: Vec<DrawerRecord> = staged
            .into_iter()
            .filter(|r| !existing_ids.contains(&r.id))
            .collect();
        if fresh.is_empty() {
            return Ok(());
        }

        // Embed only the survivors.
        let texts: Vec<String> = fresh.iter().map(|r| r.content.clone()).collect();
        let vectors = self.embed_many(texts)?;
        if vectors.len() != fresh.len() {
            return Err(PalaceError::Backend(format!(
                "embed_many returned {} vectors for {} records",
                vectors.len(),
                fresh.len()
            )));
        }

        let seqs = self.bump_seq_many(fresh.len())?;
        let schema = self.schema.clone();
        let batch = build_insert_batch_many(schema.clone(), fresh, vectors, &seqs)?;

        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
        );

        // Stage 3: merge_insert as safety net for concurrent writers.
        self.rt_block_on(async {
            let mut builder = self.table.merge_insert(&["id"]);
            builder.when_not_matched_insert_all();
            builder
                .execute(reader)
                .await
                .map(|_| ())
                .map_err(|e| PalaceError::Backend(format!("add_many merge_insert failed: {e}")))
        })
    }

    /// Fast path for bulk insert when the caller has already filtered out
    /// ids that exist in the table (typically via a global per-wing
    /// pre-pass). Skips the per-batch `id IN (...)` SELECT that
    /// [`Palace::add_many`] does, which is pure overhead here.
    ///
    /// Still does:
    /// - in-batch HashSet dedup (cheap, catches caller-side duplicates)
    /// - merge_insert safety net (catches concurrent-writer races)
    ///
    /// Measurement against v1 baseline (edtech-platform, 7470 drawers): the
    /// per-batch SELECT was firing 117 times in the old batch=64 flow.
    /// With batch=1024 that drops to 8, and this override drops it to 0.
    fn add_many_prededuped(&mut self, records: Vec<DrawerRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let mut seen_in_batch: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(records.len());
        let mut staged: Vec<DrawerRecord> = Vec::with_capacity(records.len());
        for r in records {
            if seen_in_batch.insert(r.id.clone()) {
                staged.push(r);
            }
        }
        if staged.is_empty() {
            return Ok(());
        }

        // Drain previous write before starting the next embed+write cycle.
        // This ensures at most one in-flight merge_insert at any time.
        self.drain_pending_write()?;

        let t_embed = std::time::Instant::now();
        let texts: Vec<String> = staged.iter().map(|r| r.content.clone()).collect();
        let vectors = self.embed_many(texts)?;
        let embed_ms = t_embed.elapsed().as_millis() as u64;
        if vectors.len() != staged.len() {
            return Err(PalaceError::Backend(format!(
                "embed_many returned {} vectors for {} records",
                vectors.len(),
                staged.len()
            )));
        }
        debug!(n = staged.len(), embed_ms, "batch embedded");

        let seqs = self.bump_seq_many(staged.len())?;
        let schema = self.schema.clone();
        let batch = build_insert_batch_many(schema.clone(), staged, vectors, &seqs)?;

        let reader: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
        );

        let table = self.table.clone();
        self.pending_write = Some(self.runtime.spawn(async move {
            let t_write = std::time::Instant::now();
            let mut builder = table.merge_insert(&["id"]);
            builder.when_not_matched_insert_all();
            let result = builder
                .execute(reader)
                .await
                .map(|_| ())
                .map_err(|e| format!("merge_insert failed: {e}"));
            tracing::debug!(
                write_ms = t_write.elapsed().as_millis() as u64,
                "merge_insert done"
            );
            result
        }));

        Ok(())
    }

    /// Load all drawer ids in the given wing, for ingest-side pre-pass
    /// dedup. Scoped to a single wing to keep the scan bounded: largest
    /// wing today is `convo_opencode` at ~69k rows (~250 ms). Cross-wing
    /// dedup was never a guarantee of the current system.
    fn load_existing_ids_for_wing(&self, wing: &str) -> Result<std::collections::HashSet<String>> {
        let wing_escaped = escape_sql_literal(wing)?;
        let filter = format!("wing = '{wing_escaped}'");
        self.rt_block_on(async {
            let stream = self
                .table
                .query()
                .only_if(filter)
                .select(lancedb::query::Select::columns(&["id"]))
                .execute()
                .await
                .map_err(|e| {
                    PalaceError::Backend(format!("load_existing_ids_for_wing query failed: {e}"))
                })?;
            let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| {
                PalaceError::Backend(format!("load_existing_ids_for_wing collect failed: {e}"))
            })?;
            let cap: usize = batches.iter().map(|b| b.num_rows()).sum();
            let mut set = std::collections::HashSet::with_capacity(cap);
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

    fn flush(&mut self) -> Result<()> {
        self.drain_pending_write()
    }

    fn rebuild_vector_index(&mut self) -> Result<()> {
        self.drain_pending_write()?;
        let row_count = self.rt_block_on(async {
            self.table
                .count_rows(None)
                .await
                .map_err(|e| PalaceError::Backend(format!("count_rows failed: {e}")))
        })?;
        if row_count < VECTOR_INDEX_MIN_ROWS {
            return Ok(());
        }
        debug!(row_count, "rebuilding IVF_PQ vector index after ingest");
        self.rt_block_on(async {
            self.table
                .create_index(
                    &["vector"],
                    lancedb::index::Index::IvfPq(
                        IvfPqIndexBuilder::default()
                            .distance_type(DistanceType::Cosine)
                            .num_partitions(std::cmp::max(1, (row_count / 4096) as u32))
                            .num_sub_vectors(EMBEDDING_DIM as u32 / 16),
                    ),
                )
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("rebuild vector index failed: {e}")))
        })?;
        self.has_vector_index = true;
        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        self.drain_pending_write()?;
        let id_escaped = escape_sql_literal(id)?;
        let predicate = format!("id = '{id_escaped}'");
        self.rt_block_on(async {
            let before = self
                .table
                .count_rows(Some(predicate.clone()))
                .await
                .map_err(|e| PalaceError::Backend(format!("pre-delete count failed: {e}")))?;
            if before == 0 {
                return Ok(false);
            }
            self.table
                .delete(&predicate)
                .await
                .map_err(|e| PalaceError::Backend(format!("delete failed: {e}")))?;
            Ok(true)
        })
    }

    fn get(&self, id: &str) -> Result<Option<DrawerRecord>> {
        let id_escaped = escape_sql_literal(id)?;
        let filter = format!("id = '{id_escaped}'");
        let batches: Vec<RecordBatch> = self.rt_block_on(async {
            let stream = self
                .table
                .query()
                .only_if(filter)
                .limit(1)
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("get query failed: {e}")))?;
            stream
                .try_collect()
                .await
                .map_err(|e| PalaceError::Backend(format!("get collect failed: {e}")))
        })?;

        for batch in &batches {
            if batch.num_rows() > 0 {
                return Ok(Some(row_to_record(batch, 0)?));
            }
        }
        Ok(None)
    }

    fn list(&self, limit: usize, offset: usize) -> Result<Vec<DrawerRecord>> {
        let want = limit.saturating_add(offset);
        if want == 0 {
            return Ok(Vec::new());
        }

        // Collect records with their insert_seq, sort, then slice.
        let batches: Vec<RecordBatch> = self.rt_block_on(async {
            let stream = self
                .table
                .query()
                .limit(want.max(1))
                // No `order_by` public API; we sort client-side below.
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("list query failed: {e}")))?;
            stream
                .try_collect()
                .await
                .map_err(|e| PalaceError::Backend(format!("list collect failed: {e}")))
        })?;

        // If the first query was truncated by `want`, re-scan with no limit.
        // (Cheap safety net for correctness — lancedb does not guarantee order.)
        let all_batches =
            if total_rows(&batches) < want {
                batches
            } else {
                self.rt_block_on(async {
                    let stream =
                        self.table.query().execute().await.map_err(|e| {
                            PalaceError::Backend(format!("list rescan failed: {e}"))
                        })?;
                    let out: Vec<RecordBatch> = stream.try_collect().await.map_err(|e| {
                        PalaceError::Backend(format!("list rescan collect failed: {e}"))
                    })?;
                    Ok::<_, PalaceError>(out)
                })?
            };

        let mut with_seq: Vec<(i64, DrawerRecord)> = Vec::new();
        for batch in &all_batches {
            for row in 0..batch.num_rows() {
                let seq = read_i64(batch, "insert_seq", row)?.unwrap_or(i64::MAX);
                with_seq.push((seq, row_to_record(batch, row)?));
            }
        }
        with_seq.sort_by_key(|(s, _)| *s);

        Ok(with_seq
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|(_, r)| r)
            .collect())
    }

    fn list_filtered(&self, filter: &SearchFilter, limit: usize) -> Result<Vec<DrawerRecord>> {
        let where_clause = build_where_clause(filter)?;
        let batches: Vec<RecordBatch> = self.rt_block_on(async {
            let mut q = self.table.query();
            if let Some(w) = where_clause {
                q = q.only_if(w);
            }
            let stream = q
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("list_filtered failed: {e}")))?;
            stream
                .try_collect()
                .await
                .map_err(|e| PalaceError::Backend(format!("list_filtered collect failed: {e}")))
        })?;

        let mut with_seq: Vec<(i64, DrawerRecord)> = Vec::new();
        for batch in &batches {
            for row in 0..batch.num_rows() {
                let seq = read_i64(batch, "insert_seq", row)?.unwrap_or(i64::MAX);
                with_seq.push((seq, row_to_record(batch, row)?));
            }
        }
        with_seq.sort_by_key(|(s, _)| *s);
        Ok(with_seq.into_iter().take(limit).map(|(_, r)| r).collect())
    }

    fn search(
        &self,
        query: &str,
        filter: &SearchFilter,
        n_results: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.trim().is_empty() || n_results == 0 {
            return Ok(Vec::new());
        }

        let vector = self.embed_one(query)?;
        let where_clause = build_where_clause(filter)?;

        let use_index = self.has_vector_index;
        let batches: Vec<RecordBatch> = self.rt_block_on(async {
            let mut q = self
                .table
                .query()
                .nearest_to(vector)
                .map_err(|e| PalaceError::Backend(format!("nearest_to failed: {e}")))?
                .distance_type(DistanceType::Cosine)
                .limit(n_results);
            if use_index {
                q = q.nprobes(20).refine_factor(2);
            }
            if let Some(w) = where_clause {
                q = q.only_if(w);
            }
            let stream = q
                .execute()
                .await
                .map_err(|e| PalaceError::Backend(format!("search execute failed: {e}")))?;
            stream
                .try_collect()
                .await
                .map_err(|e| PalaceError::Backend(format!("search collect failed: {e}")))
        })?;

        let mut out: Vec<SearchResult> = Vec::new();
        for batch in &batches {
            for row in 0..batch.num_rows() {
                let rec = row_to_record(batch, row)?;
                let similarity = row_similarity(batch, row)?;
                out.push(SearchResult {
                    id: rec.id,
                    content: rec.content,
                    metadata: rec.metadata,
                    similarity,
                });
            }
        }
        out.truncate(n_results);
        Ok(out)
    }
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

// Hint: connection is held to keep the database handle alive; we don't
// currently call it after construction but dropping it would close the
// underlying object store.
impl LanceDbPalace {
    #[allow(dead_code)]
    fn _keep_alive(&self) -> &Connection {
        &self.connection
    }
}

// ── tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn schema_has_expected_fields() {
        let s = build_schema();
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "id",
                "content",
                "vector",
                "wing",
                "room",
                "hall",
                "source_file",
                "date",
                "importance",
                "extra_json",
                "insert_seq",
            ]
        );

        // Spot-check types.
        assert_eq!(
            s.field_with_name("id").unwrap().data_type(),
            &DataType::Utf8
        );
        assert!(!s.field_with_name("id").unwrap().is_nullable());
        assert!(s.field_with_name("wing").unwrap().is_nullable());

        match s.field_with_name("vector").unwrap().data_type() {
            DataType::FixedSizeList(inner, sz) => {
                assert_eq!(*sz, EMBEDDING_DIM);
                assert_eq!(inner.data_type(), &DataType::Float32);
            }
            other => panic!("vector field wrong type: {other:?}"),
        }

        assert_eq!(
            s.field_with_name("insert_seq").unwrap().data_type(),
            &DataType::Int64
        );
    }

    #[test]
    fn escape_sql_doubles_quotes() {
        assert_eq!(escape_sql_literal("abc").unwrap(), "abc");
        assert_eq!(escape_sql_literal("o'brien").unwrap(), "o''brien");
        assert_eq!(escape_sql_literal("a'b'c").unwrap(), "a''b''c");
    }

    #[test]
    fn escape_sql_rejects_control_chars() {
        assert!(escape_sql_literal("line1\nline2").is_err());
        assert!(escape_sql_literal("tab\there").is_err());
        assert!(escape_sql_literal("nul\0byte").is_err());
    }

    #[test]
    fn build_where_clause_empty_filter_is_none() {
        let w = build_where_clause(&SearchFilter::default()).unwrap();
        assert!(w.is_none());
    }

    #[test]
    fn build_where_clause_escapes_values() {
        let f = SearchFilter {
            wing: Some("o'reilly".to_string()),
            room: Some("r1".to_string()),
        };
        let w = build_where_clause(&f).unwrap().unwrap();
        assert_eq!(w, "wing = 'o''reilly' AND room = 'r1'");
    }

    // ── Full round-trip tests (require model download) ─────────────────

    fn model_download_allowed() -> bool {
        std::env::var("MEMPALACE_ALLOW_MODEL_DOWNLOAD")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    #[test]
    #[ignore = "downloads ONNX model; set MEMPALACE_ALLOW_MODEL_DOWNLOAD=1 and remove ignore locally"]
    fn roundtrip_add_get_search_delete() {
        if !model_download_allowed() {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        let mut palace = LanceDbPalace::new(tmp.path()).unwrap();

        let rec = DrawerRecord {
            id: "d1".to_string(),
            content: "The quick brown fox jumps over the lazy dog".to_string(),
            metadata: DrawerMetadata {
                wing: Some("code".to_string()),
                room: Some("rust".to_string()),
                ..Default::default()
            },
        };
        palace.add(rec.clone()).unwrap();
        assert_eq!(palace.count().unwrap(), 1);

        let got = palace.get("d1").unwrap().unwrap();
        assert_eq!(got.content, rec.content);

        // Duplicate add should fail.
        assert!(matches!(
            palace.add(rec.clone()),
            Err(PalaceError::Duplicate(_))
        ));

        // Search should find the drawer.
        let hits = palace
            .search("fast brown animal", &SearchFilter::default(), 5)
            .unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, "d1");

        // Filter narrows results.
        let hits = palace
            .search(
                "fox",
                &SearchFilter {
                    wing: Some("code".to_string()),
                    room: None,
                },
                5,
            )
            .unwrap();
        assert_eq!(hits.len(), 1);

        let hits = palace
            .search(
                "fox",
                &SearchFilter {
                    wing: Some("nope".to_string()),
                    room: None,
                },
                5,
            )
            .unwrap();
        assert!(hits.is_empty());

        // list_filtered.
        let listed = palace
            .list_filtered(
                &SearchFilter {
                    wing: Some("code".to_string()),
                    room: None,
                },
                10,
            )
            .unwrap();
        assert_eq!(listed.len(), 1);

        // delete.
        assert!(palace.delete("d1").unwrap());
        assert!(!palace.delete("d1").unwrap());
        assert_eq!(palace.count().unwrap(), 0);
    }

    #[test]
    #[ignore = "downloads ONNX model; set MEMPALACE_ALLOW_MODEL_DOWNLOAD=1 and remove ignore locally"]
    fn persistence_across_reopen() {
        if !model_download_allowed() {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        {
            let mut palace = LanceDbPalace::new(tmp.path()).unwrap();
            palace
                .add(DrawerRecord {
                    id: "a".to_string(),
                    content: "hello world".to_string(),
                    metadata: DrawerMetadata::default(),
                })
                .unwrap();
            palace
                .add(DrawerRecord {
                    id: "b".to_string(),
                    content: "second drawer".to_string(),
                    metadata: DrawerMetadata::default(),
                })
                .unwrap();
        }

        let palace = LanceDbPalace::new(tmp.path()).unwrap();
        assert_eq!(palace.count().unwrap(), 2);
        let listed = palace.list(10, 0).unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].id, "a");
        assert_eq!(listed[1].id, "b");
    }
}
