
# --- CONFIGURATION ---
# METADATA_FILE = "/olga-data1/Ezekiel/protein_embeddings/metadata.tsv"
# FASTA_FILE = "/olga-data1/Ezekiel/protein_embeddings/spikeprot0723/spikeprot0723.fasta"
# OUTPUT_FASTA = "/olga-data1/Ezekiel/protein_embeddings/data_processing/cleaned_data/spike_sequences.fasta"
# OUTPUT_META = "/olga-data1/Ezekiel/protein_embeddings/data_processing/cleaned_data/metadata.csv"
#!/usr/bin/env python3
"""
GISAID Spike Protein Preprocessing (Simple + Robust)

Pipeline:
1) Ensure input files exist (can optionally extract .tar.xz)
2) PASS 1 over FASTA: build accession_int set (EPI_ISL_#### -> int)
3) Stream metadata.tsv in chunks:
     - parse accession_int
     - FIRST filter: accession_int in FASTA set
     - THEN apply filters: USA, complete, high coverage, date range, valid lineage
     - dedupe by accession_int
4) Add stratification columns (quarter + variant_bucket + stratum)
5) Iterative top-up:
     - stratified exact sample a batch from remaining pool (by stratum)
     - PASS 2 over FASTA: extract targets; QC; dedup by ungapped sequence
     - repeat until >= 20k unique sequences OR pool exhausted
6) Final: exact stratified downsample to 20k; write FASTA + metadata CSV

Notes:
- "Stratified exact" keeps fill/trim *within strata*, not from a global pool.
- Gap ranges are reported in ALIGNED coordinates if '-' exists (not reference coords).
"""

from __future__ import annotations

import sys
import re
import tarfile
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import numpy as np


# 
# CONFIG (edit if needed)
# 



# If you already extracted, set these to the extracted files:
DEFAULT_FASTA_FILE = Path("/olga-data1/Ezekiel/protein_embeddings/spikeprot0723/spikeprot0723.fasta")
DEFAULT_META_FILE  = Path("/olga-data1/Ezekiel/protein_embeddings/metadata.tsv")

OUTPUT_DIR = Path("cleaned_data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUT_FASTA = OUTPUT_DIR / "spike_sequences.fasta"
OUT_META  = OUTPUT_DIR / "metadata.csv"
OUT_LOG   = OUTPUT_DIR / "run_log.txt"

TARGET_UNIQUE = 20_000

# Filters
USA_ONLY = True
DATE_START = "2019-01-01"
DATE_END   = "2024-12-31"

MIN_LENGTH_AA = 1270
MAX_AMBIG_FRAC = 0.05  # ambiguous letters: X,B,Z,J

# Sampling
INITIAL_OVERSAMPLE_FACTOR = 1.8  # first batch size = need * factor
MIN_BATCH = 5_000

MAX_PER_STRATUM = 500   # soft balancing cap (used in initial allocation)
MIN_PER_STRATUM = 10    # diversity floor (where possible)

RANDOM_SEED = 42

# Metadata columns expected (adjust if your package differs)
META_COLS = [
    "Accession ID",
    "Collection date",
    "Location",
    "Pango lineage",
    "Is complete?",
    "Is high coverage?",
]


ACC_RE = re.compile(r"EPI_ISL_(\d+)")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# 
# Small utilities
# 

def log(msg: str) -> None:
    print(msg)
    try:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        # logging must never crash the pipeline
        pass


def parse_accession_int(text: str) -> Optional[int]:
    """Extract integer from any string containing EPI_ISL_####."""
    if not isinstance(text, str):
        return None
    m = ACC_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def safe_extract_tar_xz(tar_path: Path, out_dir: Path) -> None:
    """Extract tar.xz safely (no path traversal)."""
    try:
        if not tar_path.exists():
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[extract] Extracting {tar_path} -> {out_dir}")
        with tarfile.open(tar_path, mode="r:*") as tf:
            for member in tf.getmembers():
                member_path = out_dir / member.name
                # prevent path traversal
                if not str(member_path.resolve()).startswith(str(out_dir.resolve())):
                    raise RuntimeError(f"Unsafe path in tar: {member.name}")
            tf.extractall(path=out_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract {tar_path}: {e}") from e


def find_first_file(root: Path, patterns: Tuple[str, ...]) -> Optional[Path]:
    """Find first file in root (recursive) matching any suffix/pattern."""
    try:
        if not root.exists():
            return None
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(name.endswith(sfx) for sfx in patterns):
                return p
        return None
    except Exception:
        return None


# 
# FASTA parsing (simple)
# 

@dataclass
class FastaRecord:
    header: str
    seq: str


def fasta_stream(path: Path) -> Iterable[FastaRecord]:
    """Minimal FASTA streaming parser (fast, no dependencies)."""
    header = None
    seq_parts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield FastaRecord(header=header, seq="".join(seq_parts))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            yield FastaRecord(header=header, seq="".join(seq_parts))


def index_fasta_accessions(fasta_path: Path) -> Set[int]:
    """PASS 1: build set of accession ints from FASTA headers."""
    log("\n=== STEP 1: Index FASTA accessions ===")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    acc_set: Set[int] = set()
    year_counts = Counter()
    min_date: Optional[str] = None
    max_date: Optional[str] = None

    n = 0
    for rec in fasta_stream(fasta_path):
        n += 1
        if n % 1_000_000 == 0:
            log(f"  indexed {n:,} records; unique accessions={len(acc_set):,}")

        acc = parse_accession_int(rec.header)
        if acc is not None:
            acc_set.add(acc)

        # optional date stats (if header is pipe-delimited with date in field 3)
        parts = rec.header.split("|")
        if len(parts) >= 3:
            ds = parts[2].strip()
            if DATE_RE.match(ds):
                if min_date is None or ds < min_date:
                    min_date = ds
                if max_date is None or ds > max_date:
                    max_date = ds
                year_counts[ds[:4]] += 1

    log(f"✓ FASTA records scanned: {n:,}")
    log(f"✓ Unique accessions in FASTA: {len(acc_set):,}")
    if min_date and max_date:
        log(f"FASTA collection-date range (from headers): {min_date} .. {max_date}")
        for y in sorted(year_counts):
            log(f"  {y}: {year_counts[y]:,} (sampled from headers)")

    return acc_set


# 
# Metadata filtering
# 

def filter_metadata(meta_path: Path, fasta_acc: Set[int]) -> pd.DataFrame:
    """Stream metadata TSV, filter to FASTA intersection first, then apply filters."""
    log("\n=== STEP 2: Filter metadata (intersection-first) ===")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    kept_chunks: List[pd.DataFrame] = []
    chunk_size = 500_000
    total_rows = 0
    in_fasta = 0
    after_filters = 0

    try:
        reader = pd.read_csv(
            meta_path,
            sep="\t",
            usecols=META_COLS,
            chunksize=chunk_size,
            low_memory=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open metadata TSV with pandas: {e}") from e

    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        try:
            # accession_int as int64
            s = chunk["Accession ID"].astype(str)
            chunk["accession_int"] = s.str.extract(ACC_RE, expand=False)
            chunk = chunk.dropna(subset=["accession_int"])
            chunk["accession_int"] = chunk["accession_int"].astype("int64")

            # FIRST: in FASTA
            chunk = chunk[chunk["accession_int"].isin(fasta_acc)]
            in_fasta += len(chunk)

            # THEN: filters
            if USA_ONLY:
                chunk = chunk[chunk["Location"].astype(str).str.contains("USA", case=False, na=False)]

            chunk = chunk[chunk["Is complete?"] == True]
            chunk = chunk[chunk["Is high coverage?"] == True]

            cd = chunk["Collection date"].astype(str)
            chunk = chunk[cd.str.match(DATE_RE, na=False)]
            chunk = chunk[(chunk["Collection date"] >= DATE_START) & (chunk["Collection date"] <= DATE_END)]

            pl = chunk["Pango lineage"].astype(str)
            chunk = chunk[pl.notna()]
            chunk = chunk[~pl.isin(["None", "Unassigned", "unclassifiable"])]

            after_filters += len(chunk)
            kept_chunks.append(chunk)

        except Exception as e:
            log(f"  [warn] chunk {i} failed to process: {e}")
            continue

        if i % 5 == 0:
            log(f"  processed {total_rows:,} rows → in_fasta {in_fasta:,} → kept {after_filters:,}")

    if not kept_chunks:
        raise RuntimeError("No metadata rows survived filtering; check column names and filters.")

    df = pd.concat(kept_chunks, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset="accession_int", keep="first")
    log(f"✓ Filtered metadata rows: {before:,} (dedup -> {len(df):,})")

    return df


# 
# Stratification + exact sampling
# 

def variant_bucket(lineage: str) -> str:
    """Simple, stable-ish bucketing (keep it simple; refine later if needed)."""
    s = str(lineage)
    if s.startswith(("B.1.1.7", "Q.")): return "Alpha"
    if s.startswith("B.1.351"): return "Beta"
    if s.startswith("P.1"): return "Gamma"
    if s.startswith(("B.1.617.2", "AY.")): return "Delta"
    if s.startswith("BA.1"): return "BA.1"
    if s.startswith("BA.2.86"): return "BA.2.86"
    if s.startswith("BA.2"): return "BA.2"
    if s.startswith("BA.4"): return "BA.4"
    if s.startswith("BA.5"): return "BA.5"
    if s.startswith(("XBB", "EG.", "HK.", "XDV")): return "XBB"
    if s.startswith(("JN.1", "KP.")): return "JN.1"
    return "Other"


def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    log("\n=== STEP 3: Add stratification columns ===")
    out = df.copy()
    out["collection_date_parsed"] = pd.to_datetime(out["Collection date"], errors="coerce")
    out = out.dropna(subset=["collection_date_parsed"])
    out["year"] = out["collection_date_parsed"].dt.year.astype(int)
    out["quarter"] = out["year"].astype(str) + "-Q" + (((out["collection_date_parsed"].dt.month - 1) // 3) + 1).astype(int).astype(str)
    out["variant_bucket"] = out["Pango lineage"].apply(variant_bucket)
    out["stratum"] = out["quarter"].astype(str) + "_" + out["variant_bucket"].astype(str)
    log(f"✓ strata: {out['stratum'].nunique()} unique")
    return out


def compute_exact_stratum_sizes(
    sizes: Dict[str, int],
    target: int,
    cap: int,
    min_per: int,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Compute EXACT per-stratum sample sizes that sum to target.

    Initial allocation: n_s = min(size_s, cap) if size_s >= min_per else size_s
    Then:
      - if over target: trim within strata, not below effective_min
      - if under target: fill within strata up to remaining capacity
    """
    strata = list(sizes.keys())
    n = {}
    effective_min = {}

    for s in strata:
        sz = sizes[s]
        if sz <= 0:
            n[s] = 0
            effective_min[s] = 0
            continue
        if sz >= min_per:
            n[s] = min(sz, cap)
            effective_min[s] = min_per
        else:
            n[s] = sz
            effective_min[s] = sz  # cannot force min if stratum is tiny

    total = sum(n.values())

    if total == target:
        return n

    if total > target:
        excess = total - target
        # how much can we trim per stratum?
        trim_cap = {s: max(0, n[s] - min(effective_min[s], sizes[s])) for s in strata}
        total_trim_cap = sum(trim_cap.values())
        if total_trim_cap < excess:
            raise RuntimeError(f"Cannot trim enough to hit target (need {excess}, capacity {total_trim_cap}).")

        # proportional trim + remainder distribution
        trims = {s: int(excess * (trim_cap[s] / total_trim_cap)) if total_trim_cap > 0 else 0 for s in strata}
        # fix rounding
        trimmed = sum(trims.values())
        leftover = excess - trimmed
        if leftover > 0:
            candidates = [s for s in strata if trims[s] < trim_cap[s]]
            rng.shuffle(candidates)
            for s in candidates:
                if leftover <= 0:
                    break
                if trims[s] < trim_cap[s]:
                    trims[s] += 1
                    leftover -= 1

        for s in strata:
            n[s] -= trims[s]

        assert sum(n.values()) == target, "Trim reconciliation failed"
        return n

    # total < target => fill
    shortage = target - total
    cap_rem = {s: max(0, sizes[s] - n[s]) for s in strata}
    total_cap = sum(cap_rem.values())
    if total_cap < shortage:
        raise RuntimeError(f"Cannot fill enough to hit target (need {shortage}, capacity {total_cap}).")

    adds = {s: int(shortage * (cap_rem[s] / total_cap)) if total_cap > 0 else 0 for s in strata}
    added = sum(adds.values())
    leftover = shortage - added
    if leftover > 0:
        candidates = [s for s in strata if adds[s] < cap_rem[s]]
        rng.shuffle(candidates)
        for s in candidates:
            if leftover <= 0:
                break
            if adds[s] < cap_rem[s]:
                adds[s] += 1
                leftover -= 1

    for s in strata:
        n[s] += adds[s]

    assert sum(n.values()) == target, "Fill reconciliation failed"
    return n


def stratified_sample_exact(
    df: pd.DataFrame,
    target: int,
    exclude_accessions: Set[int],
    cap: int,
    min_per: int,
    seed: int,
) -> pd.DataFrame:
    """Exact stratified sample (count exact), fill/trim within strata."""
    rng = np.random.default_rng(seed)

    pool = df[~df["accession_int"].isin(exclude_accessions)]
    if len(pool) < target:
        raise RuntimeError(f"Not enough remaining rows to sample {target} (only {len(pool)} left).")

    sizes = pool["stratum"].value_counts().to_dict()
    desired_by_stratum = compute_exact_stratum_sizes(sizes, target, cap, min_per, rng)

    parts = []
    for stratum, n in desired_by_stratum.items():
        if n <= 0:
            continue
        g = pool[pool["stratum"] == stratum]
        if len(g) < n:
            # should not happen if compute_exact_stratum_sizes is correct
            raise RuntimeError(f"Stratum {stratum} has only {len(g)} rows, need {n}.")
        parts.append(g.sample(n=n, random_state=seed))

    out = pd.concat(parts, ignore_index=True)
    if len(out) != target:
        raise RuntimeError(f"Sampling contract violated: got {len(out)} != {target}")
    return out


# 
# Sequence QC / extraction
# 

def gap_ranges_aligned(aligned_seq: str) -> str:
    """Return gap ranges in aligned coordinates (1-indexed)."""
    ranges = []
    in_gap = False
    start = 0
    for i, ch in enumerate(aligned_seq, start=1):
        if ch == "-":
            if not in_gap:
                in_gap = True
                start = i
        else:
            if in_gap:
                end = i - 1
                ranges.append(f"{start}" if start == end else f"{start}-{end}")
                in_gap = False
    if in_gap:
        end = len(aligned_seq)
        ranges.append(f"{start}" if start == end else f"{start}-{end}")
    return ";".join(ranges)


def extract_targets_from_fasta(
    fasta_path: Path,
    target_acc: Set[int],
    meta_lookup: Dict[int, Dict],
    accepted_by_seq: Dict[str, Dict],
) -> Tuple[int, int, Dict[str, int]]:
    """
    PASS over FASTA: extract sequences for target_acc.
    - Marks an accession as "seen" once encountered (even if QC fails)
    - Adds only NEW unique ungapped sequences into accepted_by_seq

    Returns: (seen_targets, newly_added_unique, qc_failures_counter)
    """
    qc_fail = defaultdict(int)
    seen = 0
    new_unique = 0
    remaining = set(target_acc)

    for rec in fasta_stream(fasta_path):
        if not remaining:
            break

        acc = parse_accession_int(rec.header)
        if acc is None or acc not in remaining:
            continue

        remaining.discard(acc)
        seen += 1

        aligned = rec.seq.upper()
        ungapped = aligned.replace("-", "").rstrip("*")

        # QC - exact length of 1273
        if len(ungapped) != 1273:
            qc_fail["length_not_1273"] += 1
            continue

        # ambiguous fraction
        amb = sum(ungapped.count(c) for c in ("X", "B", "Z", "J"))
        frac = amb / len(ungapped) if ungapped else 1.0
        if frac > MAX_AMBIG_FRAC:
            qc_fail["too_ambiguous"] += 1
            continue

        if "*" in ungapped:
            qc_fail["internal_stop"] += 1
            continue

        if ungapped in accepted_by_seq:
            qc_fail["duplicate_sequence"] += 1
            continue

        meta = meta_lookup.get(acc)
        if meta is None:
            qc_fail["missing_meta"] += 1
            continue

        accepted_by_seq[ungapped] = {
            "accession": f"EPI_ISL_{acc}",
            "accession_int": acc,
            "collection_date": meta["Collection date"],
            "lineage": meta["Pango lineage"],
            "variant_bucket": meta["variant_bucket"],
            "location": meta["Location"],
            "year": int(meta["year"]),
            "quarter": meta["quarter"],
            "stratum": meta["stratum"],
            "sequence_ungapped": ungapped,
            "length_ungapped": len(ungapped),
            "length_with_gaps": len(aligned),
            "gap_ranges_aligned_coords": gap_ranges_aligned(aligned) if "-" in aligned else "",
            "indel_count": aligned.count("-"),
            "ambiguous_count": amb,
            "ambiguous_fraction": frac,
        }
        new_unique += 1

    return seen, new_unique, dict(qc_fail)


# 
# Main iterative top-up
# 

def iterative_collect_20k(df_pool: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    log("\n=== STEP 4: Iterative sample → extract → QC → dedup (until >=20k) ===")

    accepted_by_seq: Dict[str, Dict] = {}
    attempted_accessions: Set[int] = set()

    # simple retention estimate (updated each round)
    retention_est = 0.55

    round_num = 0
    while len(accepted_by_seq) < TARGET_UNIQUE:
        round_num += 1
        need = TARGET_UNIQUE - len(accepted_by_seq)

        # batch sizing: simple + adaptive
        batch = int(max(MIN_BATCH, need / max(retention_est, 0.10) * 1.25))
        # never exceed remaining pool
        remaining_rows = df_pool[~df_pool["accession_int"].isin(attempted_accessions)]
        if len(remaining_rows) == 0:
            raise RuntimeError(f"Pool exhausted. Collected {len(accepted_by_seq):,} unique (<{TARGET_UNIQUE:,}).")
        batch = min(batch, len(remaining_rows))

        log(f"\n--- Round {round_num} ---")
        log(f"need={need:,}  retention_est≈{retention_est:.2f}  sampling batch={batch:,}  remaining_pool={len(remaining_rows):,}")

        # exact stratified sample (or fail loudly)
        try:
            sampled = stratified_sample_exact(
                df_pool,
                target=batch,
                exclude_accessions=attempted_accessions,
                cap=MAX_PER_STRATUM,
                min_per=MIN_PER_STRATUM,
                seed=RANDOM_SEED + round_num,
            )
        except Exception as e:
            raise RuntimeError(f"Stratified exact sampling failed in round {round_num}: {e}") from e

        batch_acc = set(sampled["accession_int"].astype("int64").tolist())
        attempted_accessions.update(batch_acc)

        meta_lookup = sampled.set_index("accession_int").to_dict("index")

        try:
            seen, new_unique, qc_fail = extract_targets_from_fasta(
                fasta_path=fasta_path,
                target_acc=batch_acc,
                meta_lookup=meta_lookup,
                accepted_by_seq=accepted_by_seq,
            )
        except Exception as e:
            raise RuntimeError(f"FASTA extraction failed in round {round_num}: {e}") from e

        # update retention estimate based on actual new uniques vs batch
        realized = (new_unique / batch) if batch > 0 else 0.0
        retention_est = 0.7 * retention_est + 0.3 * max(realized, 0.05)

        log(f"seen_in_fasta={seen:,}/{batch:,}  new_unique={new_unique:,}  total_unique={len(accepted_by_seq):,}")
        if qc_fail:
            top = sorted(qc_fail.items(), key=lambda x: x[1], reverse=True)[:4]
            log("qc_fail_top: " + ", ".join([f"{k}={v}" for k, v in top]))

    # Convert to DataFrame (may be >20k)
    df_all = pd.DataFrame(list(accepted_by_seq.values()))
    log(f"\nCollected unique sequences: {len(df_all):,} (will downsample to exactly {TARGET_UNIQUE:,})")

    # Exact stratified downsample to 20k (within accepted set)
    try:
        final = stratified_sample_exact(
            df=df_all,
            target=TARGET_UNIQUE,
            exclude_accessions=set(),   # already deduped by sequence; we sample rows
            cap=MAX_PER_STRATUM,
            min_per=5,
            seed=RANDOM_SEED,
        )
    except Exception as e:
        # fallback: exact random (still exact, but warns)
        log(f"[warn] stratified downsample failed ({e}); falling back to random exact sample.")
        final = df_all.sample(n=TARGET_UNIQUE, random_state=RANDOM_SEED)

    if len(final) != TARGET_UNIQUE:
        raise RuntimeError(f"Final size contract violated: {len(final)} != {TARGET_UNIQUE}")

    return final


def write_outputs(df: pd.DataFrame) -> None:
    log("\n=== STEP 5: Write outputs ===")

    # FASTA
    try:
        with open(OUT_FASTA, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                header = f">{row['lineage']}|{row['collection_date']}|{row['accession']}"
                f.write(header + "\n")
                seq = row["sequence_ungapped"]
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")
        log(f"✓ wrote FASTA: {OUT_FASTA}  (n={len(df):,})")
    except Exception as e:
        raise RuntimeError(f"Failed writing FASTA: {e}") from e

    # metadata CSV (no sequence column)
    try:
        cols = [
            "accession", "collection_date", "lineage", "variant_bucket",
            "location", "year", "quarter",
            "length_ungapped", "length_with_gaps",
            "gap_ranges_aligned_coords", "indel_count",
            "ambiguous_count", "ambiguous_fraction",
        ]
        df[cols].to_csv(OUT_META, index=False)
        log(f"✓ wrote metadata: {OUT_META}")
    except Exception as e:
        raise RuntimeError(f"Failed writing metadata CSV: {e}") from e


# 
# Main
# 

def main() -> None:
    # reset log
    try:
        if OUT_LOG.exists():
            OUT_LOG.unlink()
    except Exception:
        pass

    log("GISAID Spike Preprocess (simple) starting...")

    # 0) Ensure inputs (optionally extract tars)
    fasta_path = DEFAULT_FASTA_FILE
    meta_path = DEFAULT_META_FILE

    # try:
    #     if not fasta_path.exists() and DEFAULT_FASTA_TAR.exists():
    #         safe_extract_tar_xz(DEFAULT_FASTA_TAR, Path("spikeprot0723"))
    #     if not meta_path.exists() and DEFAULT_META_TAR.exists():
    #         safe_extract_tar_xz(DEFAULT_META_TAR, Path("metadata_pkg"))
    # except Exception as e:
    #     raise RuntimeError(f"Extraction step failed: {e}") from e

    # locate if still missing
    if not fasta_path.exists():
        cand = find_first_file(Path("."), (".fasta",))
        if cand:
            fasta_path = cand
    if not meta_path.exists():
        cand = find_first_file(Path("metadata_pkg"), (".tsv",))
        if cand:
            meta_path = cand

    log(f"FASTA: {fasta_path}")
    log(f"META : {meta_path}")

    # 1) Index FASTA
    fasta_acc = index_fasta_accessions(fasta_path)

    # 2) Filter metadata
    df_meta = filter_metadata(meta_path, fasta_acc)

    # 3) Add strata
    df_meta = add_strata(df_meta)

    # 4) Iterative collect
    final_df = iterative_collect_20k(df_meta, fasta_path)

    # 5) Write outputs
    write_outputs(final_df)

    # Summary
    log("\n=== DONE ===")
    log(f"Final sequences: {len(final_df):,}")
    log("Variant buckets:")
    for k, v in final_df["variant_bucket"].value_counts().items():
        log(f"  {k}: {v:,}")
    log("Years:")
    for k, v in final_df["year"].value_counts().sort_index().items():
        log(f"  {k}: {v:,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\nFATAL: {e}")
        sys.exit(1)
