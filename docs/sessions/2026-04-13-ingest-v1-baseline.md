# v1 Baseline — current code on main (552ed21)

Machine: 32 logical cores (`nproc`), Linux WSL2, fastembed 5.13, lancedb 0.27.2 pinned.
Model: `AllMiniLML6V2` (non-quantized, 384-dim, ONNX Runtime CPU).
Date: 2026-04-13.
Commit at time of run: current `main`.

## Fixtures

| Fixture                             | Path                                          | Files indexed | Drawers written |
|-------------------------------------|-----------------------------------------------|---------------|-----------------|
| edtech-platform (small real)        | `~/workspace/tardis/edtech-platform`          | 540           | 7,470           |
| maps.tardis.digital (medium real)   | `~/workspace/tardis/maps.tardis.digital`      | 331           | 2,657           |

Note: maps on-disk is ~856 MB but most is `.gitignore`-excluded (node_modules, build output). edtech is smaller on disk but yields 2.8x more drawers — it's the real workload shape.

## Timings (wall clock, single fresh mine, temp palace)

| Fixture        | Wall  | User CPU | Sys   | %CPU  | Peak RSS | Drawers/sec |
|----------------|-------|----------|-------|-------|----------|-------------|
| edtech-platform| 127.87 s | 2519.5 s | 69.0 s | 2024% | 4.2 GB | **58.4** |
| maps.tardis    | 50.42 s  | 1025.2 s | 35.8 s | 2104% | 5.4 GB | **52.7** |

## Findings

**The single ORT session is already CPU-saturating.** %CPU of 2024% = 20.2 logical cores actively used by one `TextEmbedding` through `Mutex<TextEmbedding>`. fastembed uses ORT's default intra-op thread pool, which scales MiniLM-L6 to roughly 20-21 cores on this box. The earlier design estimate of "one embedder only uses 4-8 cores" was wrong by ~3x.

**Implication for the 15x plan:** spawning N=4 independent embedders was going to deliver much less than I estimated, because the total CPU ceiling is 32, not 32×N. At best N=1 already uses 20 cores, N=2 would use ~30 with diminishing returns, N=4 would thrash. The right design is N=1 + everything else (bigger batches, parallel Stage 1, pre-pass dedup, pipeline overlap).

**The 2.2M voluntary context switches on edtech** point to heavy lock contention and/or channel/sync overhead. Batch size 64 means 7470/64 ≈ 117 flushes, each doing: prefilter SELECT, Mutex lock, embed, Mutex release, Arrow build, merge_insert. Dropping that to batch 1024 = 8 flushes is the single biggest lever.

**Peak RSS (4-5 GB)** is the ONNX Runtime session heap + arena. Not a concern unless we multiply it by N embedders — another reason to stick with N=1.

## Bottleneck rank (revised post-measurement)

1. **Tiny batches cause tiny ONNX calls + many lancedb operations.** 64 → 1024 is expected to give 3-5x alone.
2. **Stage 1 (walk + read + chunk) runs serially while ORT is on the hot path.** rayon-parallelizing Stage 1 overlaps that wall time into the embed wall time: expected 1.3-1.5x.
3. **Per-batch prefilter SELECT fires 117 times for edtech, redundantly.** Global per-wing pre-pass eliminates this: expected 1.1-1.2x on fresh mines, 50-100x on re-mines.
4. **No embed↔write pipelining.** Single crossbeam channel with a dedicated writer thread: expected 1.1-1.2x.
5. **Embedder pool N>1 is now a low-value lever** given the 20-core saturation — maybe 1.2-1.5x if we cap intra-op threads and use 2 sessions total. Keeping as a follow-up, not in the first PR.

**Realistic compounded target for the v2 PR:** 4-5 × 1.4 × 1.15 × 1.15 = **~7-10x** on edtech. Re-mines should hit >50x because the pre-pass skips all embedding.

## Raw output

```
=== edtech-platform ===
IngestStats { files_scanned: 568, files_skipped_binary: 25, files_indexed: 540,
              drawers_written: 7470, drawers_skipped_existing: 0 }
User time: 2519.51 s    Wall: 2:07.87    %CPU: 2024%    Max RSS: 4,191,520 KB
Voluntary ctx-switches: 2,214,293    Involuntary: 114,462

=== maps.tardis.digital ===
IngestStats { files_scanned: ?, files_skipped_binary: 18, files_indexed: 331,
              drawers_written: 2657, drawers_skipped_existing: 0 }
User time: 1025.19 s    Wall: 0:50.42    %CPU: 2104%    Max RSS: 5,445,392 KB
Voluntary ctx-switches: 742,728    Involuntary: 53,314
```
