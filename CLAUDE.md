# MemPalace — operator's notes

Local-first semantic memory store. Ingests Claude Code / Codex / opencode /
Gemini conversation history and project source code into a single searchable
"palace" exposed to every AI agent over MCP.

This file is the working runbook. It captures the architecture, exact commands,
and every sharp edge that's bitten us so far. Read top to bottom the first time;
after that it's a reference.

---

## 1. Architecture in one diagram

```
                                    ┌─────────────────────┐
   Claude Code stdio ────┐          │                     │
   opencode stdio ───────┤  ──→     │  mempalace daemon   │  ──→  ~/.mempalace/
   Codex stdio ──────────┤          │  (holds file lock,  │        ├── mempalace_drawers.lance/
   Gemini stdio ─────────┘          │   serves MCP,       │        ├── knowledge_graph.sqlite3
        ▲                           │   one writer)       │        ├── .lock          (fs2 exclusive)
        │ each spawns                └─────────────────────┘        ├── .lock.pid
        │                                     ▲                    ├── mempalace.sock  (0600 stdio proxy)
   mempalace mcp-serve                        │                    └── mempalace.pid
   --connect ~/.mempalace/mempalace.sock ─────┘

   Offline: mempalace mine …  (requires daemon stopped — direct palace open)
```

**Key rules:**

1. **Single writer.** The daemon holds an exclusive `fs2::lock_exclusive()` on
   `~/.mempalace/.lock`. Anything else that tries to open the palace with
   `LanceDbPalace::new` waits 5 s then errors with *"palace is locked by
   another mempalace process"*. No races, no corruption.
2. **MCP agents never touch LanceDB directly.** They run `mempalace mcp-serve
   --connect <sock>`, which is a pure byte pump: `stdin → UnixStream → daemon`,
   `daemon → UnixStream → stdout`. No palace construction, no lock.
3. **Mining is offline.** `mempalace mine` opens the palace directly, so you
   must stop the daemon first. Four agents go temporarily unhealthy, restart
   daemon when done.
4. **Embeddings are local.** `fastembed 5` + `AllMiniLML6V2` (384-dim). First
   time the model isn't cached, fastembed downloads ~80 MB from HuggingFace.
   CPU-only (no working GPU path in this fastembed version).

---

## 2. One-time setup

```bash
# Prereqs: Rust ≥1.82, protoc, C toolchain.
which protoc cargo rustc || { echo "missing build deps"; exit 1; }

# Build.
cd ~/workspace/tardis/mempalace
cargo build --release
# → target/release/mempalace  (~135 MB, includes lancedb + fastembed + ONNX)

# Stable symlink so configs don't break when the repo moves.
mkdir -p ~/.local/bin
ln -sf $PWD/target/release/mempalace ~/.local/bin/mempalace

# Create palace dir + wing config.
mempalace init
# → writes ~/.mempalace/wing_config.json
```

---

## 3. Starting and stopping the daemon

```bash
# Start (foreground — wrap in tmux/nohup/systemd for real daemonising).
mempalace --palace ~/.mempalace daemon

# Check it's up.
pgrep -af 'mempalace daemon'
ls -la ~/.mempalace/mempalace.sock ~/.mempalace/mempalace.pid

# Stop (cleanly — current daemon only reacts to SIGINT via Ctrl-C; SIGTERM
# also works but does NOT trigger the select! cleanup branch, so stale
# socket + pidfile are left behind. Harmless; next start removes them).
kill -TERM $(cat ~/.mempalace/mempalace.pid)

# After any unclean shutdown:
rm -f ~/.mempalace/mempalace.sock ~/.mempalace/mempalace.pid
# .lock stays — it's the lockfile handle, not the PID marker.
```

The daemon binds `~/.mempalace/mempalace.sock` at mode `0600` (owner-only). No
TCP listener, no network exposure.

---

## 4. Wiring MCP agents

All four agents point at the **same** daemon socket. Never pass `--palace`
directly — that would try to open LanceDB from inside the agent process and
collide with the daemon's exclusive lock.

### Claude Code
```bash
claude mcp add mempalace --scope user -- \
    /home/pronit/.local/bin/mempalace mcp-serve \
    --connect /home/pronit/.mempalace/mempalace.sock
claude mcp list | grep mempalace    # should say ✓ Connected
```

### opencode — `~/.config/opencode/opencode.json`
```json
"mcp": {
  "mempalace": {
    "type": "local",
    "enabled": true,
    "command": [
      "/home/pronit/.local/bin/mempalace",
      "mcp-serve",
      "--connect",
      "/home/pronit/.mempalace/mempalace.sock"
    ]
  }
}
```

### Codex — `~/.codex/config.toml`
```toml
[mcp_servers.mempalace]
command = "/home/pronit/.local/bin/mempalace"
args = ["mcp-serve", "--connect", "/home/pronit/.mempalace/mempalace.sock"]
```

### Gemini CLI — `~/.gemini/settings.json`
```json
"mcpServers": {
  "mempalace": {
    "command": "/home/pronit/.local/bin/mempalace",
    "args": ["mcp-serve", "--connect", "/home/pronit/.mempalace/mempalace.sock"]
  }
}
```

---

## 5. Mining recipes

Always stop the daemon first:
```bash
kill -TERM $(cat ~/.mempalace/mempalace.pid) 2>/dev/null
sleep 1
rm -f ~/.mempalace/mempalace.sock ~/.mempalace/mempalace.pid
```

### Convo sources (natively supported)

```bash
# Claude Code sessions (~/.claude/projects/**/*.jsonl)
mempalace --palace ~/.mempalace mine ~/.claude/projects \
    --mode convos --wing convo_claude_code

# OpenAI Codex CLI sessions (~/.codex/sessions/**/*.jsonl)
mempalace --palace ~/.mempalace mine ~/.codex/sessions \
    --mode convos --wing convo_codex

# opencode — point at the SQLite DB, not the stale storage/ dir.
# The real data lives in opencode.db (2500+ sessions); the file-based
# storage/ tree only has a handful of residual sessions.
mempalace --palace ~/.mempalace mine ~/.local/share/opencode/opencode.db \
    --mode opencode --wing convo_opencode
```

### Project source code

```bash
# One project:
mempalace --palace ~/.mempalace mine ~/workspace/someproj \
    --mode projects --wing proj_someproj --room general

# Loop over all tardis projects, one wing per folder:
for d in ~/workspace/tardis/*/; do
    name=$(basename "$d")
    case "$name" in
        mempalace|bmapd.zip:Zone.Identifier) continue ;;
    esac
    wing="proj_$(echo "$name" | tr '[:upper:]-' '[:lower:]_' | tr -cd 'a-z0-9_')"
    echo ">>> $name -> $wing"
    mempalace --palace ~/.mempalace mine "$d" \
        --mode projects --wing "$wing" --room general
done
```

**Wing naming convention:**
- `convo_<agent>` — conversation sources
- `proj_<snake_case_name>` — project source dirs

**Re-mines are fast.** The `add_many` path has a three-stage dedup:
1. In-batch HashSet (collapses dupes in the same batch)
2. Prefilter: `SELECT id WHERE id IN (...)` over the BTree scalar index
   on `id` — drops rows that already exist **before** embedding
3. `merge_insert` safety net for concurrent-writer races

Before the prefilter: re-mining a fully-indexed project took ~66 s (all
embedding work, thrown away). After: **1.4 s** for 1,401 drawers. Fresh mining
is unaffected — new rows flow straight through to embed + merge_insert.

Restart the daemon when the mine finishes:
```bash
mempalace --palace ~/.mempalace daemon &
claude mcp list | grep mempalace    # verify ✓ Connected
```

---

## 6. Supported formats

Conversation parsers (`crates/mempalace-text/src/normalize.rs`, tried in order):

| Format                       | Detection                          | Notes                                      |
|------------------------------|------------------------------------|--------------------------------------------|
| Claude Code JSONL            | `type: human/user/assistant`       | Native                                     |
| OpenAI Codex CLI JSONL       | `type: session_meta` + entries     | Native                                     |
| claude.ai web export JSON    | top-level JSON                     | Native                                     |
| ChatGPT export JSON          | top-level JSON                     | Native (works on `conversations.json`)     |
| Slack workspace export JSON  | top-level JSON                     | Native                                     |
| Plain transcript with `> `   | ≥3 `>`-prefixed lines              | Native passthrough                         |
| opencode SQLite DB           | `--mode opencode` + `.db` path     | Custom reader, joins session/message/part  |
| Project source files         | `--mode projects`                  | 21-ext allowlist, `.gitignore` respected   |

Project source allowlist: `txt md py js ts jsx tsx json yaml yml html css java
go rs rb sh csv sql toml`. Everything else is skipped as binary. Files over 10
MB are skipped; symlinks are always skipped.

---

## 7. Searching from the command line

Right now, **while the daemon is running**, you cannot run `mempalace search`
or `mempalace status` directly — both open the palace and get blocked by the
daemon's exclusive lock. Two workarounds:

- **Use an MCP agent.** In Claude Code, ask: *"Use the mempalace tool to
  search for 'oauth'."* Claude invokes `mempalace_search` via the proxied
  MCP channel.
- **Stop the daemon, run the command, restart.** Fine for one-off checks.

There's an open improvement (not done yet): add `--connect <sock>` to
`search` and `status` so they proxy over MCP the same way `mcp-serve` does.

---

## 8. Daemon life-cycle troubleshooting

| Symptom                                              | Fix                                                                  |
|------------------------------------------------------|----------------------------------------------------------------------|
| `palace is locked by another mempalace process`     | Find the holder: `cat ~/.mempalace/.lock.pid`; kill or wait for it   |
| Agents report MCP unhealthy but daemon is running   | `claude mcp list` — it may just be a transient reconnect             |
| `failed to bind unix socket … Address already in use`| `rm ~/.mempalace/mempalace.sock` then restart daemon                  |
| Daemon dead but `.lock.pid` stale                    | Safe to remove; next open overwrites it                              |
| Mine stuck near-100% CPU forever                     | Either very large project or dedup-quadratic bug (pre-prefilter builds). Check binary mtime. |
| `add_many merge_insert failed: duplicate source`    | In-batch dup leaked past the HashSet; shouldn't happen — file a bug  |

---

## 9. Known limitations (as of 2026-04-13)

1. **Daemon only catches SIGINT cleanly.** SIGTERM skips the cleanup branch,
   leaving stale `mempalace.sock` + `mempalace.pid` files. Not dangerous — the
   next start removes the socket — but noisy.
2. **Miner cannot run while daemon holds the lock.** Design, not bug. The
   alternative (route `add_many` through the daemon over MCP) would require
   a bulk-insert MCP tool that doesn't exist.
3. **`search` / `status` / `compress` CLI commands** also need the daemon
   stopped. Same root cause.
4. **No auto-start.** Daemon must be launched manually (or via a systemd
   user unit you write yourself). If you reboot, restart it.
5. **Embedding is CPU-only.** `fastembed 5.13`'s `cuda` feature only covers
   Qwen3 / nomic-v2-moe (via candle), not the `AllMiniLML6V2` ort path. A
   working GPU story needs `ort-load-dynamic` + a CUDA-built
   `libonnxruntime.so` + matching CUDA toolkit + cuDNN. Not worth the rabbit
   hole yet; the model is small and batching + prefilter already get us
   linear scaling.
6. **Opencode reader has two code paths.** File-based walker (legacy, for the
   8-session `storage/` tree) and SQLite reader (modern, for `opencode.db`).
   The modern path is what you actually want — always point `mempalace mine`
   at the `.db` file, not the parent directory.
7. **First daemon startup after a build rebuilds the BTree index on `id`.**
   ~seconds for a 300k-row palace, idempotent (`list_indices()` check).
   Subsequent opens are free.
8. **GitHub PAT + z.ai key in plaintext in `~/.config/opencode/opencode.json`.**
   Flagged during setup. Not mempalace's problem, but noting it since the
   file lives near our configs.

---

## 10. Current state reference

Palace: `~/.mempalace/` — LanceDB + SQLite knowledge graph + wing config.

| Source                          | Drawers    |
|---------------------------------|------------|
| `convo_claude_code`             | ~43,767    |
| `convo_codex`                   | ~1,148     |
| `convo_opencode`                | ~69,297    |
| 27 `proj_*` tardis wings (partial) | ~213,506 |
| **Total**                       | ~327,718   |

16 tardis projects still pending ingest: `music-gen`, `music-spleeter`,
`music.tardis.digital`, `navratna`, `navratna-workflow-step`, `packages`,
`poker`, `research`, `sacred-concord`, `scribe`, `sentinels`, `shopify`,
`tardis-lander`, `usage-dashboard`, `valhalla-docker`, `vector-custom`, `zero`
— plus `multi` which was in-progress when killed. The tardis loop above handles
all of them; the prefilter makes re-runs cheap so you can re-invoke the loop
any time to top up.

---

## 11. Code layout quick reference

```
crates/
├── mempalace-core/       constants, version, path helpers, sanitization
├── mempalace-text/       chunking, normalization, 6 chat-format parsers, dialect
├── mempalace-store/      Palace trait, InMemoryPalace, LanceDbPalace, KnowledgeGraph
├── mempalace-server/     ingest.Miner, convo_miner.ConvoMiner, opencode_reader,
│                         hooks, searcher, MCP server + stdio/unix transports
└── mempalace-cli/        main.rs — clap dispatch, daemon/mcp-serve/mine commands
```

Key files to touch when changing behavior:

| Change                                    | File                                                        |
|-------------------------------------------|-------------------------------------------------------------|
| New chat format parser                    | `crates/mempalace-text/src/normalize.rs`                    |
| New ingest source (like opencode)         | new module in `crates/mempalace-server/src/` + lib.rs       |
| Dedup / embedding pipeline                | `crates/mempalace-store/src/lancedb_backend.rs::add_many`   |
| MCP tool registration                     | `crates/mempalace-server/src/mcp_transport.rs::build_router`|
| CLI subcommand                            | `crates/mempalace-cli/src/main.rs::Command` + dispatch      |

Tests: `cargo test --release` (the release profile is much faster for the
lancedb tests).
