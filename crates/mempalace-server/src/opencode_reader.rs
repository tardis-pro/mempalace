//! Opencode session reader.
//!
//! Opencode stores conversations two ways. The modern layout is a single
//! SQLite database at `~/.local/share/opencode/opencode.db` with tables
//! `session`, `message`, `part` — metadata in columns, full content in a
//! `data` TEXT column holding a JSON blob. The legacy layout is split
//! key-value JSON files under `~/.local/share/opencode/storage/{session,
//! message,part}/...`.
//!
//! This module prefers the SQLite DB when present and falls back to the
//! file walker otherwise. Both paths emit the same [`DrawerRecord`] shape.
//!
//! Part types of interest: `text`, `reasoning`, `tool`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mempalace_store::palace::{DrawerMetadata, DrawerRecord, Palace, PalaceError};
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const MIN_CHUNK_SIZE: usize = 80;
pub const MAX_CHUNK_CHARS: usize = 2000;
const BATCH_SIZE: usize = 64;

#[derive(Debug, Error)]
pub enum OpencodeError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("palace error: {0}")]
    Palace(#[from] PalaceError),
    #[error("opencode storage root not found: {0}")]
    RootMissing(PathBuf),
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("json decode error at {context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },
}

pub type Result<T> = std::result::Result<T, OpencodeError>;

#[derive(Debug, Default, Clone)]
pub struct OpencodeMineStats {
    pub sessions_scanned: usize,
    pub sessions_emitted: usize,
    pub messages_joined: usize,
    pub parts_joined: usize,
    pub drawers_filed: usize,
}

/// Top-level entry. Accepts either a path to `opencode.db` directly, or a
/// directory containing either `opencode.db` (modern) or the `session/`,
/// `message/`, `part/` tree (legacy). Picks whichever is present.
pub fn mine_opencode(
    input: &Path,
    palace: &mut dyn Palace,
    wing: Option<String>,
) -> Result<OpencodeMineStats> {
    let wing = wing.unwrap_or_else(|| "convo_opencode".to_string());

    // Case 1: user pointed at a .db file directly.
    if input.is_file() {
        return mine_from_sqlite(input, palace, &wing);
    }

    if !input.is_dir() {
        return Err(OpencodeError::RootMissing(input.to_path_buf()));
    }

    // Case 2: directory containing opencode.db — prefer it.
    let candidate_db = input.join("opencode.db");
    if candidate_db.is_file() {
        return mine_from_sqlite(&candidate_db, palace, &wing);
    }

    // Case 3: directory that looks like the file-based storage tree.
    let session_root = input.join("session");
    if session_root.is_dir() {
        return mine_from_files(input, palace, &wing);
    }

    Err(OpencodeError::RootMissing(input.to_path_buf()))
}

// ── SQLite-backed reader ────────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
struct MessageData {
    #[serde(default)]
    role: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct PartData {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    state: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct TimeData {
    #[serde(default)]
    created: Option<i64>,
}

fn mine_from_sqlite(
    db_path: &Path,
    palace: &mut dyn Palace,
    wing: &str,
) -> Result<OpencodeMineStats> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    let mut stats = OpencodeMineStats::default();
    let mut buffer: Vec<DrawerRecord> = Vec::with_capacity(BATCH_SIZE);

    // Iterate sessions ordered by creation time so the palace wing stays
    // roughly chronological.
    let mut session_stmt = conn.prepare(
        "SELECT id, title, directory \
         FROM session \
         ORDER BY time_created ASC",
    )?;

    let session_rows = session_stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    })?;

    // Prepared statements for per-session joins. We re-prepare inside the
    // loop because rusqlite borrows the connection for each stmt.
    for session_row in session_rows {
        let (session_id, title, directory) = session_row?;
        stats.sessions_scanned += 1;

        let transcript =
            build_sqlite_transcript(&conn, &session_id, title.as_deref(), &mut stats)?;
        if transcript.trim().len() < MIN_CHUNK_SIZE {
            continue;
        }

        let source_ref = match directory.as_deref() {
            Some(d) => format!("opencode://{session_id}?dir={d}"),
            None => format!("opencode://{session_id}"),
        };
        let room = detect_room(title.as_deref().unwrap_or(""));

        for (idx, chunk) in paragraph_chunks(&transcript).into_iter().enumerate() {
            if chunk.len() < MIN_CHUNK_SIZE {
                continue;
            }
            let drawer_id = make_drawer_id(&session_id, idx, &chunk);
            buffer.push(DrawerRecord {
                id: drawer_id,
                content: chunk,
                metadata: DrawerMetadata {
                    wing: Some(wing.to_string()),
                    room: Some(room.to_string()),
                    hall: Some("hall_facts".to_string()),
                    source_file: Some(source_ref.clone()),
                    importance: Some(3.0),
                    ..DrawerMetadata::default()
                },
            });
        }
        stats.sessions_emitted += 1;

        while buffer.len() >= BATCH_SIZE {
            let rest = buffer.split_off(BATCH_SIZE);
            let flushed = buffer.len();
            palace.add_many(std::mem::replace(&mut buffer, rest))?;
            stats.drawers_filed += flushed;
        }
    }

    if !buffer.is_empty() {
        let flushed = buffer.len();
        palace.add_many(buffer)?;
        stats.drawers_filed += flushed;
    }

    Ok(stats)
}

fn build_sqlite_transcript(
    conn: &Connection,
    session_id: &str,
    title: Option<&str>,
    stats: &mut OpencodeMineStats,
) -> Result<String> {
    let mut out = String::new();
    if let Some(t) = title {
        out.push_str("# ");
        out.push_str(t);
        out.push_str("\n\n");
    }

    // Fetch messages for this session in chronological order.
    let mut msg_stmt = conn.prepare(
        "SELECT id, data \
         FROM message \
         WHERE session_id = ?1 \
         ORDER BY time_created ASC",
    )?;
    let msg_rows = msg_stmt.query_map([session_id], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;

    // Collect them first so we can release the message stmt before we
    // prepare the parts stmt (rusqlite only allows one active stmt per
    // connection at a time during iteration).
    let mut messages: Vec<(String, String)> = Vec::new();
    for r in msg_rows {
        messages.push(r?);
    }
    stats.messages_joined += messages.len();
    drop(msg_stmt);

    let mut part_stmt = conn.prepare(
        "SELECT id, data \
         FROM part \
         WHERE message_id = ?1 \
         ORDER BY time_created ASC",
    )?;

    for (msg_id, msg_data_json) in &messages {
        let msg_data: MessageData = serde_json::from_str(msg_data_json).unwrap_or_default();
        let role = msg_data.role.as_deref().unwrap_or("unknown");
        let prefix = match role {
            "user" => "> user:",
            "assistant" => ">> assistant:",
            _ => "> ?:",
        };

        // Ordered by time within this message. The `part.data` blob is
        // opaque JSON — parse per row.
        let mut message_text = String::new();

        // Use a BTreeMap keyed on id so ordering is stable even if
        // time_created ties exist. query_map preserves ORDER BY order,
        // but we also want a deterministic tie-break.
        let part_iter = part_stmt.query_map([msg_id.as_str()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut parts: BTreeMap<String, String> = BTreeMap::new();
        for p in part_iter {
            let (id, data) = p?;
            parts.insert(id, data);
        }

        for (_id, data) in parts {
            stats.parts_joined += 1;
            let part: PartData = match serde_json::from_str(&data) {
                Ok(p) => p,
                Err(_) => continue,
            };
            match part.kind.as_deref() {
                Some("text") => {
                    if let Some(t) = part.text {
                        let t = t.trim();
                        if !t.is_empty() {
                            message_text.push_str(t);
                            message_text.push('\n');
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(t) = part.text {
                        let t = t.trim();
                        if !t.is_empty() {
                            message_text.push_str("[reasoning] ");
                            message_text.push_str(t);
                            message_text.push('\n');
                        }
                    }
                }
                Some("tool") => {
                    let tool_name = part.tool.as_deref().unwrap_or("tool");
                    message_text.push_str(&format!("[tool {tool_name}] "));
                    if let Some(state) = part.state {
                        if let Some(input) = state.get("input") {
                            if let Ok(s) = serde_json::to_string(input) {
                                let trimmed: String = s.chars().take(300).collect();
                                message_text.push_str(&trimmed);
                                message_text.push(' ');
                            }
                        }
                        if let Some(output) = state.get("output").and_then(|o| o.as_str()) {
                            let trimmed: String = output.chars().take(500).collect();
                            message_text.push_str("→ ");
                            message_text.push_str(&trimmed);
                        }
                    }
                    message_text.push('\n');
                }
                _ => {}
            }
        }

        if !message_text.trim().is_empty() {
            out.push_str(prefix);
            out.push(' ');
            out.push_str(message_text.trim());
            out.push_str("\n\n");
        }
    }

    Ok(out)
}

// ── Legacy file-based reader (kept for backwards compat) ────────────────

#[derive(Debug, Deserialize)]
struct SessionJson {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    directory: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageJson {
    id: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    time: Option<TimeData>,
}

#[derive(Debug, Deserialize)]
struct PartJson {
    #[serde(default)]
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    state: Option<serde_json::Value>,
}

fn mine_from_files(
    storage_root: &Path,
    palace: &mut dyn Palace,
    wing: &str,
) -> Result<OpencodeMineStats> {
    let session_root = storage_root.join("session");
    let message_root = storage_root.join("message");
    let part_root = storage_root.join("part");

    let mut stats = OpencodeMineStats::default();
    let mut buffer: Vec<DrawerRecord> = Vec::with_capacity(BATCH_SIZE);

    let project_dirs = std::fs::read_dir(&session_root).map_err(|e| OpencodeError::Io {
        path: session_root.clone(),
        source: e,
    })?;

    for project_entry in project_dirs.flatten() {
        let project_path = project_entry.path();
        if !project_path.is_dir() {
            continue;
        }
        let session_files = match std::fs::read_dir(&project_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        for session_entry in session_files.flatten() {
            let session_file = session_entry.path();
            if session_file.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            stats.sessions_scanned += 1;

            let session = match read_json_file::<SessionJson>(&session_file) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let transcript =
                build_file_transcript(&session, &message_root, &part_root, &mut stats);
            if transcript.trim().len() < MIN_CHUNK_SIZE {
                continue;
            }

            let source_ref = match session.directory.as_deref() {
                Some(d) => format!("opencode://{}?dir={}", session.id, d),
                None => format!("opencode://{}", session.id),
            };
            let room = detect_room(session.title.as_deref().unwrap_or(""));

            for (idx, chunk) in paragraph_chunks(&transcript).into_iter().enumerate() {
                if chunk.len() < MIN_CHUNK_SIZE {
                    continue;
                }
                let drawer_id = make_drawer_id(&session.id, idx, &chunk);
                buffer.push(DrawerRecord {
                    id: drawer_id,
                    content: chunk,
                    metadata: DrawerMetadata {
                        wing: Some(wing.to_string()),
                        room: Some(room.to_string()),
                        hall: Some("hall_facts".to_string()),
                        source_file: Some(source_ref.clone()),
                        importance: Some(3.0),
                        ..DrawerMetadata::default()
                    },
                });
            }
            stats.sessions_emitted += 1;

            while buffer.len() >= BATCH_SIZE {
                let rest = buffer.split_off(BATCH_SIZE);
                let flushed = buffer.len();
                palace.add_many(std::mem::replace(&mut buffer, rest))?;
                stats.drawers_filed += flushed;
            }
        }
    }

    if !buffer.is_empty() {
        let flushed = buffer.len();
        palace.add_many(buffer)?;
        stats.drawers_filed += flushed;
    }

    Ok(stats)
}

fn build_file_transcript(
    session: &SessionJson,
    message_root: &Path,
    part_root: &Path,
    stats: &mut OpencodeMineStats,
) -> String {
    let msg_dir = message_root.join(&session.id);
    let entries = match std::fs::read_dir(&msg_dir) {
        Ok(d) => d,
        Err(_) => return String::new(),
    };

    let mut messages: Vec<(i64, String, String)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let msg = match read_json_file::<MessageJson>(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let ts = msg
            .time
            .as_ref()
            .and_then(|t| t.created)
            .unwrap_or(i64::MAX);
        let role = msg.role.unwrap_or_else(|| "unknown".to_string());
        messages.push((ts, msg.id, role));
    }
    messages.sort_by_key(|(t, _, _)| *t);
    stats.messages_joined += messages.len();

    let mut out = String::new();
    if let Some(ref title) = session.title {
        out.push_str("# ");
        out.push_str(title);
        out.push_str("\n\n");
    }

    for (_ts, msg_id, role) in &messages {
        let part_dir = part_root.join(msg_id);
        let mut parts: BTreeMap<String, PartJson> = BTreeMap::new();
        if let Ok(entries) = std::fs::read_dir(&part_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(part) = read_json_file::<PartJson>(&path) {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        parts.insert(name.to_string(), part);
                    }
                }
            }
        }

        let prefix = match role.as_str() {
            "user" => "> user:",
            "assistant" => ">> assistant:",
            _ => "> ?:",
        };
        let mut message_text = String::new();
        for (_name, part) in parts {
            stats.parts_joined += 1;
            match part.kind.as_deref() {
                Some("text") => {
                    if let Some(t) = part.text {
                        if !t.trim().is_empty() {
                            message_text.push_str(t.trim());
                            message_text.push('\n');
                        }
                    }
                }
                Some("reasoning") => {
                    if let Some(t) = part.text {
                        if !t.trim().is_empty() {
                            message_text.push_str("[reasoning] ");
                            message_text.push_str(t.trim());
                            message_text.push('\n');
                        }
                    }
                }
                Some("tool") => {
                    let tool_name = part.tool.as_deref().unwrap_or("tool");
                    message_text.push_str(&format!("[tool {tool_name}] "));
                    if let Some(state) = part.state {
                        if let Some(input) = state.get("input") {
                            if let Ok(s) = serde_json::to_string(input) {
                                let trimmed: String = s.chars().take(300).collect();
                                message_text.push_str(&trimmed);
                                message_text.push(' ');
                            }
                        }
                        if let Some(output) = state.get("output").and_then(|o| o.as_str()) {
                            let trimmed: String = output.chars().take(500).collect();
                            message_text.push_str("→ ");
                            message_text.push_str(&trimmed);
                        }
                    }
                    message_text.push('\n');
                }
                _ => {}
            }
        }

        if !message_text.trim().is_empty() {
            out.push_str(prefix);
            out.push(' ');
            out.push_str(message_text.trim());
            out.push_str("\n\n");
        }
    }

    out
}

// ── Shared helpers ──────────────────────────────────────────────────────

fn paragraph_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if current.len() + para.len() + 2 > MAX_CHUNK_CHARS && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks
}

fn detect_room(title: &str) -> &'static str {
    let t = title.to_lowercase();
    if t.contains("bug") || t.contains("fix") || t.contains("error") || t.contains("debug") {
        "problems"
    } else if t.contains("plan") || t.contains("design") || t.contains("architect") {
        "architecture"
    } else if t.contains("test") {
        "technical"
    } else {
        "general"
    }
}

fn make_drawer_id(session_id: &str, chunk_index: usize, content: &str) -> String {
    let mut h = Sha256::new();
    h.update(b"opencode::");
    h.update(session_id.as_bytes());
    h.update(b"::");
    h.update(chunk_index.to_le_bytes());
    h.update(b"::");
    h.update(content.as_bytes());
    let digest = h.finalize();
    let hex: String = digest.iter().take(12).map(|b| format!("{b:02x}")).collect();
    format!("opencode_{session_id}_{chunk_index}_{hex}")
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|e| OpencodeError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| OpencodeError::Json {
        context: path.display().to_string(),
        source: e,
    })
}
