#![forbid(unsafe_code)]
#![allow(clippy::pedantic)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mempalace_server::convo_miner::{ConvoMiner, ExtractMode};
use mempalace_server::hooks::{SaveHook, SaveRequest};
use mempalace_server::ingest::{Miner, MinerOptions};
use mempalace_server::mcp::McpServer;
use mempalace_server::onboarding::WingConfig;
use mempalace_server::opencode_reader::mine_opencode;
use mempalace_server::searcher::{format_human, search_memories, SearchQuery};
use mempalace_store::knowledge_graph::KnowledgeGraph;
use mempalace_store::layers::MemoryStack;
use mempalace_store::palace::{DrawerRecord, InMemoryPalace, Palace, SearchFilter};
use mempalace_store::LanceDbPalace;
use mempalace_text::dialect::Dialect;
use tokio::io::AsyncWriteExt;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "mempalace",
    version = mempalace_core::VERSION,
    about = "Give your AI a memory — mine projects and conversations into a searchable palace.",
)]
struct Cli {
    #[arg(long, global = true)]
    palace: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Show palace status and drawer counts")]
    Status,

    #[command(about = "Search the palace")]
    Search {
        query: Vec<String>,
        #[arg(long)]
        wing: Option<String>,
        #[arg(long)]
        room: Option<String>,
        #[arg(long, default_value_t = 5)]
        n: usize,
    },

    #[command(about = "Guided first-run onboarding: write wing_config.json")]
    Init {
        #[arg(long)]
        person: Vec<String>,
        #[arg(long)]
        project: Vec<String>,
    },

    #[command(
        about = "Mine a directory into the palace (project files, conversations, or opencode storage)"
    )]
    Mine {
        dir: PathBuf,
        #[arg(long)]
        wing: Option<String>,
        #[arg(long, default_value = "general")]
        room: String,
        #[arg(long, default_value = "projects", value_parser = ["projects", "convos", "opencode"])]
        mode: String,
        #[arg(long, default_value = "exchange", value_parser = ["exchange", "general"])]
        extract: String,
    },

    #[command(about = "Wake-up text: L0 identity + L1 essential story")]
    WakeUp {
        #[arg(long)]
        wing: Option<String>,
    },

    #[command(about = "Split concatenated transcript mega-files")]
    Split {
        dir: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },

    #[command(about = "Print MCP setup command for Claude / ChatGPT / Cursor")]
    Mcp,

    #[command(about = "Hook save trigger (for editor / shell integrations)")]
    HookSave {
        #[arg(long)]
        wing: Option<String>,
        #[arg(long)]
        room: Option<String>,
        #[arg(long)]
        source: Option<String>,
        content: String,
    },

    #[command(about = "Run MCP server over stdio (for Claude / ChatGPT / Cursor)")]
    McpServe {
        /// Proxy stdin/stdout to a running mempalace daemon over this Unix socket
        /// instead of owning the palace directly. Lets multiple agents share a
        /// single palace-holding process, which is required for concurrent writes.
        #[arg(long)]
        connect: Option<PathBuf>,
    },

    #[command(
        about = "Run the palace daemon: one process owns the palace and accepts MCP connections over a Unix socket"
    )]
    Daemon {
        /// Unix socket path to bind. Defaults to <palace>/mempalace.sock.
        #[arg(long)]
        socket: Option<PathBuf>,
    },

    #[command(about = "Compress palace drawers to AAAK Dialect for token savings")]
    Compress {
        #[arg(long, help = "Only compress drawers from this wing")]
        wing: Option<String>,
        #[arg(long, help = "Show stats without writing compressed records")]
        dry_run: bool,
    },

    #[command(about = "Print version and build info")]
    Instructions,
}

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("warn"))
        .context("failed to build tracing env filter")?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|e| anyhow::anyhow!("failed to initialise tracing subscriber: {e}"))?;

    let cli = Cli::parse();
    let palace_path = cli.palace.clone();

    // Commands that manage their own palace + runtime must run BEFORE
    // open_palace() — LanceDbPalace refuses to construct inside a running
    // tokio runtime, and mcp-serve/daemon both create their own runtime.
    match &cli.command {
        Command::McpServe { connect } => {
            return cmd_mcp_serve(palace_path.as_deref(), connect.as_deref());
        }
        Command::Daemon { socket } => {
            return cmd_daemon(palace_path.as_deref(), socket.as_deref());
        }
        _ => {}
    }

    let mut palace = open_palace(palace_path.as_deref())?;

    match cli.command {
        Command::Status => cmd_status(palace.as_ref()),
        Command::Search {
            query,
            wing,
            room,
            n,
        } => cmd_search(palace.as_ref(), &query.join(" "), wing, room, n),
        Command::Init { person, project } => cmd_init(&person, &project),
        Command::Mine {
            dir,
            wing,
            room,
            mode,
            extract,
        } => cmd_mine(palace.as_mut(), &dir, wing, room, &mode, &extract),
        Command::WakeUp { wing } => cmd_wake_up(palace.as_ref(), wing.as_deref()),
        Command::Split { dir, dry_run } => cmd_split(&dir, dry_run),
        Command::Mcp => cmd_mcp(),
        Command::McpServe { .. } | Command::Daemon { .. } => unreachable!("handled above"),
        Command::HookSave {
            wing,
            room,
            source,
            content,
        } => cmd_hook_save(palace.as_mut(), wing, room, source, content),
        Command::Compress { wing, dry_run } => cmd_compress(palace.as_mut(), wing, dry_run),
        Command::Instructions => cmd_instructions(),
    }
}

/// Open a palace backend. With `Some(path)` returns a persistent
/// `LanceDbPalace`; with `None` returns an ephemeral `InMemoryPalace`.
///
/// Must be called before any tokio runtime is active — `LanceDbPalace::new`
/// refuses to construct from within a running runtime.
fn open_palace(path: Option<&Path>) -> Result<Box<dyn Palace>> {
    match path {
        Some(p) => {
            std::fs::create_dir_all(p)
                .with_context(|| format!("failed to create palace directory {}", p.display()))?;
            let palace = LanceDbPalace::new(p)
                .with_context(|| format!("failed to open LanceDbPalace at {}", p.display()))?;
            Ok(Box::new(palace))
        }
        None => Ok(Box::new(InMemoryPalace::new())),
    }
}

fn cmd_status(palace: &dyn Palace) -> Result<()> {
    let drawers = palace.count().unwrap_or(0);
    println!("mempalace {}", mempalace_core::VERSION);
    println!("drawers: {drawers}");
    println!("tools:   19");
    Ok(())
}

fn cmd_search(
    palace: &dyn Palace,
    query: &str,
    wing: Option<String>,
    room: Option<String>,
    n: usize,
) -> Result<()> {
    let q = SearchQuery {
        query: query.to_string(),
        wing,
        room,
        n_results: n,
    };
    let resp = search_memories(palace, &q);
    print!("{}", format_human(&resp));
    Ok(())
}

fn cmd_init(persons: &[String], projects: &[String]) -> Result<()> {
    let mut cfg = WingConfig::new_empty();
    for p in persons {
        let wing = cfg.add_person(p)?;
        println!("added person: {p} -> {wing}");
    }
    for p in projects {
        let wing = cfg.add_project(p, &[])?;
        println!("added project: {p} -> {wing}");
    }
    let path = WingConfig::default_path();
    cfg.save(&path)
        .with_context(|| format!("failed to write wing config to {}", path.display()))?;
    println!("wing_config written to {}", path.display());
    Ok(())
}

fn cmd_mine(
    palace: &mut dyn Palace,
    dir: &std::path::Path,
    wing: Option<String>,
    room: String,
    mode: &str,
    extract: &str,
) -> Result<()> {
    match mode {
        "projects" => {
            let miner = Miner::new(MinerOptions {
                wing,
                default_room: room,
                ..MinerOptions::default()
            });
            let stats = miner
                .mine(dir, palace)
                .with_context(|| format!("mining {}", dir.display()))?;
            println!("{stats:#?}");
        }
        "convos" => {
            let extract_mode = match extract {
                "general" => ExtractMode::General,
                _ => ExtractMode::Exchange,
            };
            let mut miner = ConvoMiner::new();
            miner.wing = wing;
            miner.extract_mode = extract_mode;
            let stats = miner
                .mine(dir, palace)
                .with_context(|| format!("mining conversations from {}", dir.display()))?;
            println!("{stats:#?}");
        }
        "opencode" => {
            let stats = mine_opencode(dir, palace, wing)
                .with_context(|| format!("mining opencode storage from {}", dir.display()))?;
            println!("{stats:#?}");
        }
        _ => unreachable!("clap value_parser restricts to projects|convos|opencode"),
    }
    Ok(())
}

fn cmd_wake_up(palace: &dyn Palace, wing: Option<&str>) -> Result<()> {
    let mut stack = MemoryStack::new(palace, None);
    println!("{}", stack.wake_up(wing));
    Ok(())
}

fn cmd_split(dir: &std::path::Path, dry_run: bool) -> Result<()> {
    use mempalace_text::split_mega_files::{find_session_boundaries, split_file};

    let entries = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    let mut total = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        let lines: Vec<String> = content.lines().map(|l| format!("{l}\n")).collect();
        let boundaries = find_session_boundaries(&lines);
        if boundaries.len() < 2 {
            continue;
        }
        println!("  {}  ({} sessions)", path.display(), boundaries.len());
        let written = split_file(&path, None, dry_run)
            .with_context(|| format!("splitting {}", path.display()))?;
        total += written.len();
    }
    if dry_run {
        println!("DRY RUN — would create {total} files");
    } else {
        println!("created {total} files");
    }
    Ok(())
}

fn cmd_mcp() -> Result<()> {
    println!("# Claude Code / ChatGPT / Cursor / Gemini MCP setup\n");
    println!("claude mcp add mempalace -- mempalace mcp-serve");
    println!();
    println!("The Rust binary speaks MCP over stdio via rmcp.");
    Ok(())
}

fn cmd_hook_save(
    palace: &mut dyn Palace,
    wing: Option<String>,
    room: Option<String>,
    source: Option<String>,
    content: String,
) -> Result<()> {
    let result = SaveHook::default()
        .save(
            palace,
            SaveRequest {
                wing,
                room,
                source,
                content,
            },
        )
        .context("save hook failed")?;
    println!("{result:#?}");
    Ok(())
}

fn cmd_mcp_serve(palace_path: Option<&Path>, connect: Option<&Path>) -> Result<()> {
    // Client mode: proxy stdin/stdout to a running daemon over a Unix
    // socket. Does not open the palace — the daemon owns it.
    if let Some(socket) = connect {
        return run_mcp_proxy(socket);
    }

    // Standalone mode: this process owns the palace directly over stdio.
    // LanceDbPalace::new() refuses to run inside an active tokio runtime,
    // so we must build the palace and knowledge graph BEFORE creating
    // the runtime below.
    let palace = open_palace(palace_path)?;

    let kg = match palace_path {
        Some(p) => {
            let kg_path = p.join("knowledge_graph.sqlite3");
            KnowledgeGraph::open(&kg_path).with_context(|| {
                format!("failed to open knowledge graph at {}", kg_path.display())
            })?
        }
        None => {
            KnowledgeGraph::open(":memory:").context("failed to open in-memory knowledge graph")?
        }
    };

    let server = McpServer::new(palace, kg);

    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    rt.block_on(mempalace_server::serve_stdio(server))
}

/// Run the palace daemon: bind a Unix socket, accept MCP connections,
/// and serve them from a single shared `McpServer` holding the palace
/// exclusive lock. Exits cleanly on SIGINT/SIGTERM.
fn cmd_daemon(palace_path: Option<&Path>, socket_override: Option<&Path>) -> Result<()> {
    let palace_dir = palace_path.ok_or_else(|| {
        anyhow::anyhow!(
            "--palace <DIR> is required for daemon mode (in-memory palace is not shareable)"
        )
    })?;

    let socket_path: PathBuf = match socket_override {
        Some(p) => p.to_path_buf(),
        None => palace_dir.join("mempalace.sock"),
    };

    // Palace + KG before tokio runtime (lock acquisition happens here).
    let palace = open_palace(Some(palace_dir))?;
    let kg_path = palace_dir.join("knowledge_graph.sqlite3");
    let kg = KnowledgeGraph::open(&kg_path)
        .with_context(|| format!("failed to open knowledge graph at {}", kg_path.display()))?;

    let server = McpServer::new(palace, kg);

    // Write pid file for observability.
    let pid_path = palace_dir.join("mempalace.pid");
    let _ = std::fs::write(&pid_path, std::process::id().to_string());

    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;

    let socket_for_cleanup = socket_path.clone();
    let pid_for_cleanup = pid_path.clone();
    let result = rt.block_on(async move {
        tokio::select! {
            r = mempalace_server::serve_unix_socket(server, &socket_path) => r,
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("received SIGINT, shutting down daemon");
                Ok(())
            }
        }
    });

    // Best-effort cleanup: remove socket + pid file on shutdown.
    let _ = std::fs::remove_file(&socket_for_cleanup);
    let _ = std::fs::remove_file(&pid_for_cleanup);

    result
}

fn run_mcp_proxy(socket_path: &Path) -> Result<()> {
    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    rt.block_on(async move {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let stream = tokio::net::UnixStream::connect(socket_path)
            .await
            .with_context(|| {
                format!(
                    "failed to connect to daemon socket {}. is the daemon running?",
                    socket_path.display()
                )
            })?;
        let (rsock, mut wsock) = stream.into_split();
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut stdin_lines = BufReader::new(stdin).lines();
        let mut rsock_lines = BufReader::new(rsock).lines();

        // MCP handshake gate: rmcp drops messages arriving before the session
        // transitions to "initialized". Forward the initialize request, wait
        // for the response, then switch to raw bidirectional relay.
        let init_done = std::sync::Arc::new(tokio::sync::Notify::new());

        let init_done_up = std::sync::Arc::clone(&init_done);
        let up = async move {
            let mut handshake_done = false;
            while let Ok(Some(line)) = stdin_lines.next_line().await {
                let is_init = !handshake_done
                    && line.contains("\"initialize\"")
                    && !line.contains("initialized");
                let waiter = if is_init {
                    Some(init_done_up.notified())
                } else {
                    None
                };
                let _ = wsock.write_all(line.as_bytes()).await;
                let _ = wsock.write_all(b"\n").await;
                let _ = wsock.flush().await;
                if let Some(w) = waiter {
                    w.await;
                    handshake_done = true;
                }
            }
            let _ = wsock.shutdown().await;
        };
        let down = async move {
            let mut first_response = true;
            while let Ok(Some(line)) = rsock_lines.next_line().await {
                let _ = stdout.write_all(line.as_bytes()).await;
                let _ = stdout.write_all(b"\n").await;
                let _ = stdout.flush().await;
                if first_response {
                    init_done.notify_one();
                    first_response = false;
                }
            }
        };
        tokio::join!(up, down);
        Ok::<(), anyhow::Error>(())
    })
}

fn cmd_compress(palace: &mut dyn Palace, wing: Option<String>, dry_run: bool) -> Result<()> {
    let filter = SearchFilter { wing, room: None };
    let drawers = palace
        .list_filtered(&filter, usize::MAX)
        .context("failed to list palace drawers")?;

    if drawers.is_empty() {
        println!("No drawers to compress.");
        return Ok(());
    }

    let dialect = Dialect::new(HashMap::new(), vec![]);
    let mut total_original_chars: usize = 0;
    let mut total_compressed_chars: usize = 0;
    let mut total_drawers: usize = 0;

    for drawer in &drawers {
        // Skip already-compressed drawers
        if drawer.id.starts_with("compressed_") {
            continue;
        }

        let mut meta_map = HashMap::new();
        if let Some(ref w) = drawer.metadata.wing {
            meta_map.insert("wing".to_string(), w.clone());
        }
        if let Some(ref r) = drawer.metadata.room {
            meta_map.insert("room".to_string(), r.clone());
        }
        if let Some(ref d) = drawer.metadata.date {
            meta_map.insert("date".to_string(), d.clone());
        }
        if let Some(ref s) = drawer.metadata.source_file {
            meta_map.insert("source_file".to_string(), s.clone());
        }

        let compressed = dialect.compress(&drawer.content, Some(&meta_map));
        let stats = dialect.compression_stats(&drawer.content, &compressed);

        total_original_chars += stats.original_chars;
        total_compressed_chars += stats.summary_chars;
        total_drawers += 1;

        if dry_run {
            println!(
                "  {} — {} chars → {} chars (ratio {:.1}x)",
                drawer.id, stats.original_chars, stats.summary_chars, stats.size_ratio
            );
        } else {
            let compressed_id = format!("compressed_{}", drawer.id);
            let mut metadata = drawer.metadata.clone();
            metadata.extra.insert(
                "compression_ratio".to_string(),
                serde_json::Value::from(stats.size_ratio),
            );
            let record = DrawerRecord {
                id: compressed_id,
                content: compressed,
                metadata,
            };
            palace
                .add(record)
                .context("failed to add compressed drawer")?;
        }
    }

    let overall_ratio = if total_compressed_chars > 0 {
        total_original_chars as f64 / total_compressed_chars as f64
    } else {
        0.0
    };

    println!();
    if dry_run {
        println!("DRY RUN — no records written");
    }
    println!(
        "Summary: {total_drawers} drawers, {total_original_chars} → {total_compressed_chars} chars ({overall_ratio:.1}x)"
    );
    Ok(())
}

fn cmd_instructions() -> Result<()> {
    println!("mempalace {}", mempalace_core::VERSION);
    println!("Rust port of MemPalace. Run `mempalace --help` for commands.");
    Ok(())
}
