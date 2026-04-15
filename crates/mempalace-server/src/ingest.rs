//! Project file ingest — port of Python `mempalace/miner.py`.
//!
//! Walks a directory (respecting `.gitignore`), reads text-like files
//! under [`MAX_FILE_SIZE`] bytes, splits them into overlapping chunks,
//! and inserts the chunks as drawers into a [`Palace`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::OnceLock;

use ignore::WalkBuilder;
use mempalace_store::palace::{DrawerMetadata, DrawerRecord, Palace};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const CHUNK_SIZE: usize = 800;
pub const CHUNK_OVERLAP: usize = 100;
pub const MIN_CHUNK_SIZE: usize = 50;
pub const MAX_FILE_SIZE: u64 = 1024 * 1024;

static READABLE_EXTENSIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "txt", "md", "py", "js", "ts", "jsx", "tsx", "json", "yaml", "yml", "html", "css", "java",
        "go", "rs", "rb", "sh", "csv", "sql", "toml",
    ]
    .into_iter()
    .collect()
});

static SKIP_FILENAMES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "mempalace.yaml",
        "mempalace.yml",
        "mempal.yaml",
        "mempal.yml",
        ".gitignore",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "Cargo.lock",
        "poetry.lock",
        "Pipfile.lock",
        "composer.lock",
        "Gemfile.lock",
    ]
    .into_iter()
    .collect()
});

/// Directory names that should always be skipped even if git-tracked.
/// Covers virtualenvs, build artifacts, and vendored dependencies that
/// were accidentally committed.
static SKIP_DIRS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "node_modules",
        "venv",
        ".venv",
        "__pycache__",
        ".git",
        "target",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "vendor",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        ".fastembed_cache",
        ".cache",
        "coverage",
        ".nyc_output",
    ]
    .into_iter()
    .collect()
});

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("palace error: {0}")]
    Palace(#[from] mempalace_store::palace::PalaceError),
    #[error("root is not a directory: {0}")]
    NotADirectory(PathBuf),
    #[error("symlinked file rejected: {0}")]
    SymlinkRejected(PathBuf),
}

pub type Result<T> = std::result::Result<T, IngestError>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestStats {
    pub files_scanned: usize,
    pub files_skipped_size: usize,
    pub files_skipped_symlink: usize,
    pub files_skipped_binary: usize,
    pub files_indexed: usize,
    pub drawers_written: usize,
    pub drawers_skipped_existing: usize,
}

#[derive(Debug, Clone)]
pub struct MinerOptions {
    pub wing: Option<String>,
    pub default_room: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_file_size: u64,
}

impl Default for MinerOptions {
    fn default() -> Self {
        Self {
            wing: None,
            default_room: "general".to_string(),
            chunk_size: CHUNK_SIZE,
            chunk_overlap: CHUNK_OVERLAP,
            max_file_size: MAX_FILE_SIZE,
        }
    }
}

#[derive(Debug)]
pub struct Miner {
    options: MinerOptions,
}

impl Miner {
    pub fn new(options: MinerOptions) -> Self {
        Self { options }
    }

    pub fn mine(&self, root: &Path, palace: &mut dyn Palace) -> Result<IngestStats> {
        if !root.is_dir() {
            return Err(IngestError::NotADirectory(root.to_path_buf()));
        }

        let canonical_root = root.canonicalize()?;
        let mut stats = IngestStats::default();

        let t0 = std::time::Instant::now();
        let existing_ids: std::collections::HashSet<String> = match &self.options.wing {
            Some(w) => palace.load_existing_ids_for_wing(w)?,
            None => std::collections::HashSet::new(),
        };
        tracing::info!(
            existing = existing_ids.len(),
            elapsed_ms = t0.elapsed().as_millis() as u64,
            "loaded existing ids for dedup prefilter"
        );

        // Flush in larger batches to amortise fastembed + lancedb overhead.
        // Previous value was 64, which produced ~117 tiny ONNX calls on
        // the edtech-platform baseline (7470 drawers). 1024 drops that to
        // ~8, cutting per-call fixed cost dramatically.
        const BATCH_SIZE: usize = 1024;

        // Stage 1: walk + read + chunk in parallel. Rayon par_bridge drains
        // the (sequential) ignore walker across all cores. Each worker emits
        // DrawerRecords into a bounded channel; the main thread drains and
        // flushes to the palace. This overlaps file I/O + chunking with the
        // embed-bound palace write wall time.
        use crossbeam_channel::bounded;
        use rayon::iter::{ParallelBridge, ParallelIterator};

        let (record_tx, record_rx) = bounded::<DrawerRecord>(BATCH_SIZE * 4);
        let (file_stat_tx, file_stat_rx) = bounded::<FileStat>(4096);

        let walker = WalkBuilder::new(&canonical_root)
            .git_ignore(true)
            .git_exclude(true)
            .git_global(true)
            .hidden(true)
            .follow_links(false)
            .filter_entry(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map_or(true, |name| !SKIP_DIRS.contains(name))
            })
            .build();

        let canonical_root_cloned = canonical_root.clone();
        let options = self.options.clone();
        let existing_ids_arc = std::sync::Arc::new(existing_ids);

        let producer = {
            let record_tx = record_tx.clone();
            let file_stat_tx = file_stat_tx.clone();
            let existing_for_thread = existing_ids_arc.clone();
            std::thread::Builder::new()
                .name("mempalace-ingest-produce".to_string())
                .spawn(move || {
                    walker.flatten().par_bridge().for_each(|entry| {
                        let rtx = &record_tx;
                        let stx = &file_stat_tx;
                        let path = entry.path();
                        if !path.is_file() {
                            return;
                        }
                        let _ = stx.send(FileStat::Scanned);

                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            if SKIP_FILENAMES.contains(name) {
                                return;
                            }
                        }

                        let ext = path
                            .extension()
                            .and_then(|e| e.to_str())
                            .map(str::to_lowercase);
                        let ext_ok = ext
                            .as_deref()
                            .is_some_and(|e| READABLE_EXTENSIONS.contains(e));
                        if !ext_ok {
                            let _ = stx.send(FileStat::SkippedBinary);
                            return;
                        }

                        let metadata = match std::fs::symlink_metadata(path) {
                            Ok(m) => m,
                            Err(_) => {
                                let _ = stx.send(FileStat::SkippedBinary);
                                return;
                            }
                        };
                        if metadata.file_type().is_symlink() {
                            let _ = stx.send(FileStat::SkippedSymlink);
                            return;
                        }
                        if metadata.len() > options.max_file_size {
                            let _ = stx.send(FileStat::SkippedSize);
                            return;
                        }

                        let content = match std::fs::read_to_string(path) {
                            Ok(t) => t,
                            Err(_) => {
                                let _ = stx.send(FileStat::SkippedBinary);
                                return;
                            }
                        };

                        let relative = path
                            .strip_prefix(&canonical_root_cloned)
                            .map(Path::to_path_buf)
                            .unwrap_or_else(|_| path.to_path_buf());

                        let records = chunks_to_records(&content, &relative, &options);
                        let _ = stx.send(FileStat::Indexed);
                        for rec in records {
                            if existing_for_thread.contains(&rec.id) {
                                continue;
                            }
                            if rtx.send(rec).is_err() {
                                break;
                            }
                        }
                    });
                })
                .map_err(IngestError::Io)?
        };
        // Drop the main thread's copies of the senders so the channels close
        // when all rayon workers finish.
        drop(record_tx);
        drop(file_stat_tx);

        // Main thread drains file-stat events (drop after Drop of stx happens
        // when producer finishes) and record events into BATCH_SIZE flushes.
        // We loop until the record channel is closed.
        let mut buffer: Vec<DrawerRecord> = Vec::with_capacity(BATCH_SIZE);
        let drain_stats = |stats: &mut IngestStats| {
            while let Ok(stat) = file_stat_rx.try_recv() {
                match stat {
                    FileStat::Scanned => stats.files_scanned += 1,
                    FileStat::Indexed => stats.files_indexed += 1,
                    FileStat::SkippedBinary => stats.files_skipped_binary += 1,
                    FileStat::SkippedSize => stats.files_skipped_size += 1,
                    FileStat::SkippedSymlink => stats.files_skipped_symlink += 1,
                }
            }
        };

        while let Ok(rec) = record_rx.recv() {
            buffer.push(rec);
            if buffer.len() >= BATCH_SIZE {
                let flushed = buffer.len();
                palace.add_many_prededuped(std::mem::take(&mut buffer))?;
                buffer.reserve(BATCH_SIZE);
                stats.drawers_written += flushed;
                drain_stats(&mut stats);
            }
        }

        if !buffer.is_empty() {
            let flushed = buffer.len();
            palace.add_many_prededuped(buffer)?;
            stats.drawers_written += flushed;
        }
        palace.flush()?;

        drain_stats(&mut stats);

        if let Err(e) = producer.join() {
            return Err(IngestError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("ingest producer thread panicked: {e:?}"),
            )));
        }
        // Final drain after join in case producer pushed stats between
        // the last try_recv and its thread exit.
        drain_stats(&mut stats);

        Ok(stats)
    }
}

/// File-level outcome events emitted by the producer thread so the main
/// thread can maintain [`IngestStats`] without sharing mutable state.
#[derive(Debug, Clone, Copy)]
enum FileStat {
    Scanned,
    Indexed,
    SkippedBinary,
    SkippedSize,
    SkippedSymlink,
}

/// Pure function form of the old `Miner::chunks_into_buffer` — takes
/// options by ref instead of `&self` so a rayon worker can call it from
/// inside a `move` closure that doesn't capture the miner itself.
fn chunks_to_records(
    content: &str,
    relative_path: &Path,
    options: &MinerOptions,
) -> Vec<DrawerRecord> {
    let chunks = chunk_text(content, options.chunk_size, options.chunk_overlap);
    let mut out = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.into_iter().enumerate() {
        if chunk.len() < MIN_CHUNK_SIZE {
            continue;
        }
        let drawer_id = compute_drawer_id(relative_path, idx, &chunk);
        out.push(DrawerRecord {
            id: drawer_id,
            content: chunk,
            metadata: DrawerMetadata {
                wing: options.wing.clone(),
                room: Some(options.default_room.clone()),
                hall: Some("hall_facts".to_string()),
                source_file: Some(relative_path.to_string_lossy().into_owned()),
                date: Some(today_iso()),
                importance: Some(3.0),
                ..DrawerMetadata::default()
            },
        });
    }
    out
}

pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    if size == 0 {
        return vec![text.to_string()];
    }

    let char_offsets: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let total = char_offsets.len();
    if total <= size {
        return vec![text.to_string()];
    }

    let byte_end_of = |char_idx: usize| -> usize {
        if char_idx >= total {
            text.len()
        } else {
            char_offsets[char_idx]
        }
    };

    let mut chunks = Vec::new();
    let stride = size.saturating_sub(overlap).max(1);
    let mut start = 0usize;
    while start < total {
        let end = (start + size).min(total);
        let byte_start = char_offsets[start];
        let byte_end = byte_end_of(end);
        chunks.push(text[byte_start..byte_end].to_string());
        if end == total {
            break;
        }
        start += stride;
    }
    chunks
}

pub fn compute_drawer_id(source_file: &Path, chunk_index: usize, chunk: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source_file.to_string_lossy().as_bytes());
    hasher.update(b"::");
    hasher.update(chunk_index.to_le_bytes());
    hasher.update(b"::");
    hasher.update(chunk.as_bytes());
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .take(12)
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    format!("drawer_{hex}")
}

fn today_iso() -> String {
    static INIT: OnceLock<()> = OnceLock::new();
    let _ = INIT;
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
