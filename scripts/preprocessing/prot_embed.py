#!/usr/bin/env python3
"""
ProtT5 CPU embedding (hardened): correct checkpoint/resume via memmap + done mask.

Outputs in --outdir:
  embeddings.npy        (N, 1024) float32  <-- written incrementally (memmap)
  done_mask.npy         (N,) uint8         1=done, 0=not done, 2=skipped
  ids.txt               sequence IDs in row order
  skipped.tsv           reasons for skipped sequences
  state.json            atomic progress/status snapshot

Resume:
  Re-run with --resume; it will continue unfinished rows.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

# -----------------------------
# Logging / atomic helpers
# -----------------------------
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)

def atomic_write_json(path: Path, obj: dict):
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))

# -----------------------------
# FASTA
# -----------------------------
def read_fasta_ordered(fasta_path: Path):
    ids = []
    seqs = []
    cur_id = None
    cur = []

    with fasta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur))
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)

        if cur_id is not None:
            ids.append(cur_id)
            seqs.append("".join(cur))

    return ids, seqs

def clean_sequence_for_prott5(seq: str):
    # ProtT5 expects space-separated AAs; map rare letters to X
    s = seq.upper().replace("-", "").replace(".", "").rstrip("*")
    for ch in "UZOB":
        s = s.replace(ch, "X")
    return s

# -----------------------------
# Model / embedding
# -----------------------------
def load_model(model_name: str):
    from transformers import T5Tokenizer, T5EncoderModel
    tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    return tok, model

def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: (B, T, H), attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1).float()               # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)            # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                   # (B, 1)
    return (summed / counts).cpu().numpy()                    # (B, H)

def embed_batch(tokenizer, model, seqs_clean, max_len=None):
    # seqs_clean: list of strings with NO spaces yet
    # Convert to spaced format for tokenizer
    spaced = [" ".join(list(s[:max_len] if max_len else s)) for s in seqs_clean]

    import torch
    with torch.inference_mode():
        enc = tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True
        )
        # CPU inference
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return emb

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=2, help="Safe default for CPU. Will auto-shrink on OOM.")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=50, help="Flush + state write every N completed rows.")
    ap.add_argument("--max-aa", type=int, default=0, help="If >0, skip sequences longer than this (do NOT truncate).")
    ap.add_argument("--model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Recommend putting HF cache on fast local/project disk if possible.
    # os.environ.setdefault("HF_HOME", str(args.outdir / "hf_cache"))

    # Control CPU threading (best-effort)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)

    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    log(f"Reading FASTA: {args.fasta}")
    ids, seqs_raw = read_fasta_ordered(args.fasta)
    n = len(ids)
    if n == 0:
        raise RuntimeError("No sequences found in FASTA.")

    log(f"Loaded {n:,} sequences")

    # Paths
    emb_path = args.outdir / "embeddings.npy"
    done_path = args.outdir / "done_mask.npy"
    ids_path = args.outdir / "ids.txt"
    skipped_path = args.outdir / "skipped.tsv"
    state_path = args.outdir / "state.json"

    # Write ids.txt once (atomic)
    if (not ids_path.exists()) or (not args.resume):
        atomic_write_text(ids_path, "\n".join(ids) + "\n")

    # Prepare/Load memmaps
    if args.resume and emb_path.exists() and done_path.exists():
        log("Resuming from existing embeddings + done_mask")
        embeddings = np.load(emb_path, mmap_mode="r+")          # memmap
        done_mask = np.load(done_path, mmap_mode="r+")          # memmap
        if embeddings.shape != (n, 1024):
            raise RuntimeError(f"embeddings.npy shape {embeddings.shape} != {(n, 1024)}; FASTA/order changed?")
        if done_mask.shape != (n,):
            raise RuntimeError(f"done_mask.npy shape {done_mask.shape} != {(n,)}; FASTA/order changed?")
    else:
        log("Creating new embeddings.npy + done_mask.npy")
        embeddings = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, 1024))
        done_mask = open_memmap(done_path, mode="w+", dtype=np.uint8, shape=(n,))
        done_mask[:] = 0
        embeddings[:] = 0.0

    # Load model
    log(f"Loading model/tokenizer: {args.model}")
    tokenizer, model = load_model(args.model)
    log("Model loaded (CPU). This may be slow.")

    # Open skipped log append-only
    if not skipped_path.exists():
        skipped_path.write_text("row\tid\treason\tdetail\n")

    completed = int((done_mask == 1).sum())
    skipped = int((done_mask == 2).sum())
    log(f"Starting work. Already done={completed:,}, skipped={skipped:,}, remaining={n - completed - skipped:,}")

    # Build list of remaining indices
    remaining = [i for i in range(n) if done_mask[i] == 0]

    batch_size = max(1, args.batch_size)
    last_ckpt_done = completed + skipped

    # We’ll process remaining indices in order; if you want randomization, do it outside.
    idx_ptr = 0
    while idx_ptr < len(remaining):
        # Select a candidate batch
        b_inds = remaining[idx_ptr: idx_ptr + batch_size]

        # Pre-filter outliers/empties (mark skipped)
        batch_clean = []
        batch_rows = []
        for i in b_inds:
            s = clean_sequence_for_prott5(seqs_raw[i])
            if len(s) == 0:
                done_mask[i] = 2
                with skipped_path.open("a") as f:
                    f.write(f"{i}\t{ids[i]}\tempty_after_clean\tlen=0\n")
                continue
            if args.max_aa and len(s) > args.max_aa:
                done_mask[i] = 2
                with skipped_path.open("a") as f:
                    f.write(f"{i}\t{ids[i]}\ttoo_long\tlen={len(s)}\n")
                continue
            batch_clean.append(s)
            batch_rows.append(i)

        if len(batch_rows) == 0:
            idx_ptr += batch_size
            continue

        # Try embedding; on OOM-ish errors, shrink batch and retry
        try:
            emb = embed_batch(tokenizer, model, batch_clean)
            if emb.shape != (len(batch_rows), 1024):
                raise RuntimeError(f"Unexpected embedding shape {emb.shape}")

            # Store results
            for k, row_i in enumerate(batch_rows):
                embeddings[row_i] = emb[k]
                done_mask[row_i] = 1

            idx_ptr += batch_size  # advance only after successful embedding

        except (MemoryError, RuntimeError) as e:
            msg = str(e).lower()
            # CPU can throw various memory / allocator errors; treat them similarly
            if "out of memory" in msg or "alloc" in msg or "memory" in msg:
                if batch_size > 1:
                    new_bs = max(1, batch_size // 2)
                    log(f"Memory error; reducing batch_size {batch_size} -> {new_bs} and retrying")
                    batch_size = new_bs
                    continue
            # If batch_size==1 or not a memory-ish error: log and skip those rows
            log(f"Batch failed at rows {batch_rows[:5]}... error={e}. Marking as skipped.")
            with skipped_path.open("a") as f:
                for row_i in batch_rows:
                    done_mask[row_i] = 2
                    f.write(f"{row_i}\t{ids[row_i]}\tembed_failed\t{repr(e)}\n")
            idx_ptr += batch_size

        except Exception as e:
            log(f"Unexpected error: {e}. Marking batch as skipped.")
            with skipped_path.open("a") as f:
                for row_i in batch_rows:
                    done_mask[row_i] = 2
                    f.write(f"{row_i}\t{ids[row_i]}\tembed_failed\t{repr(e)}\n")
            idx_ptr += batch_size

        # Periodic checkpoint: flush memmaps + write atomic state
        done_now = int((done_mask == 1).sum())
        skipped_now = int((done_mask == 2).sum())
        progressed = (done_now + skipped_now) - last_ckpt_done

        if progressed >= args.checkpoint_every:
            embeddings.flush()
            done_mask.flush()
            last_ckpt_done = done_now + skipped_now
            atomic_write_json(state_path, {
                "total": n,
                "done": done_now,
                "skipped": skipped_now,
                "remaining": n - done_now - skipped_now,
                "batch_size_current": batch_size,
                "threads": args.threads,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            log(f"Checkpoint: done={done_now:,} skipped={skipped_now:,} remaining={n - done_now - skipped_now:,}")

    # Final flush + final state
    embeddings.flush()
    done_mask.flush()
    done_final = int((done_mask == 1).sum())
    skipped_final = int((done_mask == 2).sum())
    atomic_write_json(state_path, {
        "total": n,
        "done": done_final,
        "skipped": skipped_final,
        "remaining": n - done_final - skipped_final,
        "batch_size_final": batch_size,
        "threads": args.threads,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    log(f"Finished. done={done_final:,} skipped={skipped_final:,} total={n:,}")
    log(f"Embeddings: {emb_path}")
    log(f"Done mask:  {done_path}")
    log(f"IDs:        {ids_path}")
    log(f"Skipped:    {skipped_path}")
    log(f"State:      {state_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)
