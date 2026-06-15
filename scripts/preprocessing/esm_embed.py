#!/usr/bin/env python3
"""
CPU ESM-2 embedding (resume-safe, checkpointed) for protein FASTA.

Model: facebook/esm2_t33_650M_UR50D
Output:
  outdir/
    embeddings.npy        (memmapped float32, shape [N, D])
    done.npy              (bool mask, shape [N])
    ids.txt               (sequence IDs in row order)
    lengths.txt           (raw AA lengths after cleaning)
    run.log               (simple log)

Pooling:
  mean over residue tokens (excludes BOS + EOS + PAD)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple

# ---- set thread env vars BEFORE importing torch (helps on some systems)
def _set_thread_env(n: int):
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(k, str(n))

def log(msg: str, log_path: Path = None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

def clean_seq(s: str) -> str:
    """
    Clean protein sequence for ESM2:
      - uppercase
      - remove gaps/dots
      - strip terminal '*'
      - map U,Z,O,B -> X
    """
    s = (s or "").upper()
    s = s.replace("-", "").replace(".", "")
    s = s.rstrip("*")
    for ch in "UZOB":
        s = s.replace(ch, "X")
    return s

def read_fasta_simple(fasta_path: Path, require_len: int = 0, limit: int = 0) -> Tuple[List[str], List[str], List[int]]:
    """
    Minimal FASTA reader
    Returns ids, cleaned_seqs, lengths
    """
    ids: List[str] = []
    seqs: List[str] = []
    lens: List[int] = []

    cur_id = None
    cur = []

    with open(fasta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    raw = "".join(cur)
                    c = clean_seq(raw)
                    if (require_len == 0) or (len(c) == require_len):
                        ids.append(cur_id)
                        seqs.append(c)
                        lens.append(len(c))
                        if limit and len(ids) >= limit:
                            break
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)

        # flush last record (if not limited early)
        if cur_id is not None and (not limit or len(ids) < limit):
            raw = "".join(cur)
            c = clean_seq(raw)
            if (require_len == 0) or (len(c) == require_len):
                ids.append(cur_id)
                seqs.append(c)
                lens.append(len(c))

    return ids, seqs, lens

def atomic_save_npy(path: Path, arr):
    tmp = path.with_suffix(path.suffix + ".tmp")
    import numpy as np
    np.save(tmp, arr)
    tmp.replace(path)

def build_batches(order: List[int], lengths: List[int], max_tokens: int) -> List[List[int]]:
    """
    Build batches by token budget.
    Approx tokens per seq ~ len + 2 (BOS/EOS).
    """
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0

    for idx in order:
        tok = lengths[idx] + 2
        if cur and (cur_tokens + tok > max_tokens):
            batches.append(cur)
            cur = []
            cur_tokens = 0
        cur.append(idx)
        cur_tokens += tok

    if cur:
        batches.append(cur)

    return batches

def pool_mean_excluding_specials(last_hidden_state, input_ids, attention_mask, eos_token_id: int):
    """
    Mean pool per sequence excluding BOS (position 0), EOS tokens, and PAD tokens.
    """
    import torch

    # mask starts from attention_mask (PAD already 0)
    mask = attention_mask.clone().bool()

    # exclude BOS
    if mask.size(1) > 0:
        mask[:, 0] = False

    # exclude EOS tokens wherever they appear
    if eos_token_id is not None:
        eos_positions = (input_ids == eos_token_id)
        mask = mask & (~eos_positions)

    mask_f = mask.unsqueeze(-1).float()  # [B, L, 1]
    summed = (last_hidden_state * mask_f).sum(dim=1)  # [B, D]
    denom = mask_f.sum(dim=1).clamp_min(1.0)          # [B, 1]
    return summed / denom

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--cache-dir", default="", help="Optional HF cache dir (faster filesystem is better)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=32768, help="Batch token budget (len+2 summed)")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Flush/save done mask every N batches")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--require-length", type=int, default=0, help="If >0 keep only sequences of exactly this AA length")
    ap.add_argument("--limit", type=int, default=0, help="For testing: only embed first N sequences after filtering")
    args = ap.parse_args()

    _set_thread_env(args.threads)

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "run.log"

    log(f"FASTA: {args.fasta}", log_path)
    log(f"OUTDIR: {args.outdir}", log_path)
    log(f"MODEL: {args.model}", log_path)
    log(f"THREADS: {args.threads}", log_path)
    log(f"MAX_TOKENS: {args.max_tokens}", log_path)
    if args.require_length:
        log(f"FILTER: require_length={args.require_length}", log_path)
    if args.limit:
        log(f"LIMIT: {args.limit}", log_path)

    # torch threading
    try:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, args.threads // 2))
    except Exception:
        pass

    # ---- read FASTA
    t0 = time.time()
    ids, seqs, lens = read_fasta_simple(args.fasta, require_len=args.require_length, limit=args.limit)
    n = len(ids)
    if n == 0:
        log("ERROR: no sequences found after filtering.", log_path)
        sys.exit(2)
    log(f"Loaded {n} sequences in {time.time()-t0:.1f}s", log_path)

    # ---- write ids/lengths once
    ids_path = args.outdir / "ids.txt"
    lengths_path = args.outdir / "lengths.txt"
    if (not ids_path.exists()) or (not args.resume):
        with open(ids_path, "w", encoding="utf-8") as f:
            for x in ids:
                f.write(x + "\n")
        with open(lengths_path, "w", encoding="utf-8") as f:
            for L in lens:
                f.write(str(L) + "\n")

    # ---- load model/tokenizer
    cache_kwargs = {}
    if args.cache_dir:
        cache_kwargs["cache_dir"] = args.cache_dir

    log("Loading tokenizer...", log_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **cache_kwargs)
    log("Loading model (CPU)...", log_path)
    model = AutoModel.from_pretrained(args.model, **cache_kwargs)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    # dimension
    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "hidden_dim", None)
    if hidden_size is None:
        # fallback: probe
        with torch.inference_mode():
            tmp = tokenizer(["A"], return_tensors="pt")
            out = model(**tmp)
            hidden_size = out.last_hidden_state.shape[-1]
    d = int(hidden_size)
    log(f"Embedding dim: {d}", log_path)

    # ---- outputs
    emb_path = args.outdir / "embeddings.npy"
    done_path = args.outdir / "done.npy"
    meta_path = args.outdir / "meta.json"

    # done mask
    if args.resume and done_path.exists():
        done = np.load(done_path)
        if done.shape[0] != n:
            log("WARNING: done.npy length mismatch; ignoring and starting fresh.", log_path)
            done = np.zeros((n,), dtype=bool)
    else:
        done = np.zeros((n,), dtype=bool)

    # embeddings memmap
    if args.resume and emb_path.exists():
        # open existing memmap
        embeddings = np.load(emb_path, mmap_mode="r+")
        if embeddings.shape != (n, d):
            log("WARNING: embeddings.npy shape mismatch; recreating.", log_path)
            embeddings = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, d))
            embeddings[:] = 0.0
            done[:] = False
    else:
        embeddings = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, d))
        embeddings[:] = 0.0

    # save meta
    meta = {
        "model": args.model,
        "n": n,
        "d": d,
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "require_length": args.require_length,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    # ---- batching order: process longest first (reduces padding waste), but write back to original index
    order = list(range(n))
    order.sort(key=lambda i: lens[i], reverse=True)

    # remove already-done indices
    remaining = [i for i in order if not bool(done[i])]
    if not remaining:
        log("Nothing to do: all sequences already embedded (done.npy is all True).", log_path)
        sys.exit(0)

    batches = build_batches(remaining, lens, args.max_tokens)
    log(f"Planned {len(batches)} batches (remaining sequences: {len(remaining)})", log_path)

    eos_id = getattr(tokenizer, "eos_token_id", None)

    # ---- embed loop
    start = time.time()
    embedded_this_run = 0
    failed_this_run = 0

    def embed_indices(batch_indices: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Embed a batch of indices. Returns (indices_kept, emb_matrix)
        May raise exception; caller handles.
        """
        batch_seqs = [seqs[i] for i in batch_indices]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc)
            h = out.last_hidden_state  # [B, L, D]
            pooled = pool_mean_excluding_specials(
                h, enc["input_ids"], enc["attention_mask"], eos_id
            )  # [B, D]
        return batch_indices, pooled.cpu().numpy().astype(np.float32)

    for b_idx, batch in enumerate(batches, start=1):
        t_batch = time.time()
        try:
            idxs, embs = embed_indices(batch)
            # store
            for row, i in enumerate(idxs):
                embeddings[i, :] = embs[row]
                done[i] = True
            embedded_this_run += len(idxs)

        except Exception as e:
            # batch failed -> fall back to per-sequence
            log(f"Batch {b_idx}/{len(batches)} failed ({len(batch)} seqs): {type(e).__name__}: {e}", log_path)
            for i in batch:
                if done[i]:
                    continue
                try:
                    idxs, embs = embed_indices([i])
                    embeddings[i, :] = embs[0]
                    done[i] = True
                    embedded_this_run += 1
                except Exception as e2:
                    failed_this_run += 1
                    done[i] = False
                    log(f"  Single failed id={ids[i]} len={lens[i]}: {type(e2).__name__}: {e2}", log_path)

        # checkpoint
        if (b_idx % args.checkpoint_every) == 0 or (b_idx == len(batches)):
            try:
                embeddings.flush()
            except Exception:
                pass
            try:
                atomic_save_npy(done_path, done)
            except Exception as e3:
                log(f"WARNING: could not save done.npy checkpoint: {e3}", log_path)

        # progress
        done_count = int(done.sum())
        elapsed = time.time() - start
        rate = done_count / max(elapsed, 1e-6)
        log(
            f"Batch {b_idx}/{len(batches)} | done={done_count}/{n} | "
            f"this_run_ok={embedded_this_run} fail={failed_this_run} | "
            f"rate={rate:.3f} seq/s | batch_time={time.time()-t_batch:.1f}s",
            log_path
        )

    # final flush
    try:
        embeddings.flush()
    except Exception:
        pass
    try:
        atomic_save_npy(done_path, done)
    except Exception:
        pass

    log(f"Finished. Total done={int(done.sum())}/{n}. This run ok={embedded_this_run}, failed={failed_this_run}.", log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl-C received, exiting cleanly.", flush=True)
        sys.exit(130)
