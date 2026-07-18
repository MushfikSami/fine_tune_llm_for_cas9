#!/usr/bin/env python
"""Step 4: automated sgRNA designer for rice CRISPR/Cas9.

Given a target gene (MSU Gene ID resolved against the genome, or a FASTA file),
this script:

  1. Scans both strands for SpCas9 'NGG' PAM sites and extracts the 20 bp spacer
     immediately upstream of each PAM (Biopython).
  2. Scores each spacer for on-target efficiency with the LoRA-fine-tuned
     DNABERT-2 regressor, using the spacer plus 50 bp of flanking context.
  3. Runs each 20 bp spacer against the genome bowtie2 index to estimate an
     off-target penalty.
  4. Emits a ranked ``designed_sgrnas.csv`` (best guide first).

Examples:
    python 04_design_sgrna.py --gene LOC_Os01g20160
    python 04_design_sgrna.py --fasta my_target.fa
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq

from dnabert_utils import (GENOME_FASTA, GENOME_GFF3, INDEX_PREFIX, LORA_DIR,
                           load_finetuned_regressor)

FLANK = 50          # bp of context on each side of the spacer for scoring
SPACER_LEN = 20
GENE_PAD = 60       # extra genomic bp pulled around a gene so edge guides get context


# --------------------------------------------------------------------------
# Target acquisition
# --------------------------------------------------------------------------
def target_from_gene_id(gene_id: str) -> tuple[str, str]:
    """Return (name, sequence) for an MSU gene id, padded with GENE_PAD flanks."""
    seqid = start = end = None
    with open(GENOME_GFF3) as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            f = line.rstrip("\n").split("\t")
            attrs = f[8]
            if f"ID={gene_id};" in attrs or attrs.endswith(f"ID={gene_id}"):
                seqid, start, end = f[0], int(f[3]), int(f[4])
                break
    if seqid is None:
        sys.exit(f"Gene id {gene_id} not found in {GENOME_GFF3.name}")

    fasta = SeqIO.index(str(GENOME_FASTA), "fasta")
    chrom = fasta[seqid].seq
    lo = max(0, start - 1 - GENE_PAD)
    hi = min(len(chrom), end + GENE_PAD)
    seq = str(chrom[lo:hi]).upper()
    print(f"Target {gene_id}: {seqid}:{start}-{end} (+/-{GENE_PAD} bp), {len(seq)} bp")
    return gene_id, seq


def target_from_fasta(path: str) -> tuple[str, str]:
    rec = next(SeqIO.parse(path, "fasta"))
    seq = str(rec.seq).upper()
    print(f"Target {rec.id}: {len(seq)} bp (from {path})")
    return rec.id, seq


# --------------------------------------------------------------------------
# PAM scan
# --------------------------------------------------------------------------
def find_guides(seq: str) -> list[dict]:
    """Locate all NGG-PAM 20 bp spacers on both strands.

    Returns dicts with spacer, pam, strand, 0-based position, and a
    ``context`` string (spacer +/- FLANK bp, oriented 5'->3' with the spacer).
    """
    guides: list[dict] = []
    n = len(seq)

    # Forward strand: PAM = positions p..p+2 with seq[p+1:p+3] == 'GG'.
    for p in range(1, n - 2):
        if seq[p + 1] == "G" and seq[p + 2] == "G":
            s0 = p - SPACER_LEN
            if s0 < 0:
                continue
            spacer = seq[s0:p]
            if "N" in spacer:
                continue
            c_lo, c_hi = s0 - FLANK, p + 3 + FLANK
            context = seq[max(0, c_lo):min(n, c_hi)]
            guides.append(dict(spacer=spacer, pam=seq[p:p + 3], strand="+",
                               position=s0, context=context))

    # Reverse strand: forward 'CCN' at positions q..q+2 => reverse-strand NGG.
    for q in range(0, n - 2):
        if seq[q] == "C" and seq[q + 1] == "C":
            e = q + 3 + SPACER_LEN
            if e > n:
                continue
            proto = seq[q + 3:e]           # forward-strand protospacer
            if "N" in proto:
                continue
            spacer = str(Seq(proto).reverse_complement())
            pam = str(Seq(seq[q:q + 3]).reverse_complement())
            c_lo, c_hi = q - FLANK, e + FLANK
            context = str(Seq(seq[max(0, c_lo):min(n, c_hi)]).reverse_complement())
            guides.append(dict(spacer=spacer, pam=pam, strand="-",
                               position=q, context=context))
    return guides


# --------------------------------------------------------------------------
# On-target scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def score_on_target(guides: list[dict], batch_size: int = 64) -> list[float]:
    tokenizer, model = load_finetuned_regressor(LORA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    scores: list[float] = []
    contexts = [g["context"] for g in guides]
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=96).to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out["logits"] if isinstance(out, dict) else out.logits
        scores.extend(logits.float().flatten().tolist())
    return scores


# --------------------------------------------------------------------------
# Off-target penalty (bowtie2)
# --------------------------------------------------------------------------
def offtarget_counts(guides: list[dict], threads: int) -> list[int]:
    """Align each 20 bp spacer to the genome; count hits beyond the intended one."""
    if not Path(str(INDEX_PREFIX) + ".1.bt2").exists():
        sys.exit(f"Missing bowtie2 index at {INDEX_PREFIX}.* — run 02_preprocess_dnabert.py")

    with tempfile.TemporaryDirectory() as td:
        reads = Path(td) / "spacers.fa"
        with open(reads, "w") as fh:
            for i, g in enumerate(guides):
                fh.write(f">g{i}\n{g['spacer']}\n")
        sam = Path(td) / "aln.sam"
        cmd = [
            "bowtie2", "-x", str(INDEX_PREFIX), "-f", "-U", str(reads),
            "-k", "50", "-L", "10", "-N", "1",
            "--score-min", "L,-0.6,-0.9",   # tolerate ~3-4 mismatches across 20 bp
            "--no-unal", "--no-hd", "-p", str(threads), "-S", str(sam),
        ]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

        hits = [0] * len(guides)
        with open(sam) as fh:
            for line in fh:
                if line.startswith("@"):
                    continue
                cols = line.split("\t")
                if len(cols) < 3:
                    continue
                flag = int(cols[1])
                if flag & 4:  # unmapped
                    continue
                idx = int(cols[0][1:])
                hits[idx] += 1
    # The intended on-target site contributes one perfect hit; the rest are off-targets.
    return [max(0, h - 1) for h in hits]


def gc_content(s: str) -> float:
    return round(100 * (s.count("G") + s.count("C")) / len(s), 1)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gene", help="MSU gene id, e.g. LOC_Os01g20160")
    src.add_argument("--fasta", help="path to a target gene FASTA")
    ap.add_argument("--out", default="designed_sgrnas.csv")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--w-on", type=float, default=1.0, help="on-target weight")
    ap.add_argument("--w-off", type=float, default=0.1, help="off-target penalty weight")
    ap.add_argument("--top", type=int, default=0, help="keep only top-N rows (0=all)")
    args = ap.parse_args()

    name, seq = (target_from_gene_id(args.gene) if args.gene
                 else target_from_fasta(args.fasta))

    guides = find_guides(seq)
    print(f"Found {len(guides)} candidate spacers (both strands).")
    if not guides:
        sys.exit("No NGG PAM sites with a full 20 bp spacer were found.")

    print("Scoring on-target efficiency with fine-tuned DNABERT-2...")
    on_scores = score_on_target(guides)

    print("Estimating off-target penalty with bowtie2...")
    off_counts = offtarget_counts(guides, args.threads)

    rows = []
    for g, on, off in zip(guides, on_scores, off_counts):
        final = args.w_on * on - args.w_off * off
        rows.append(dict(
            gene=name, spacer=g["spacer"], pam=g["pam"], strand=g["strand"],
            position=g["position"], gc_percent=gc_content(g["spacer"]),
            on_target=round(on, 4), offtarget_count=off,
            final_score=round(final, 4),
        ))
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False,
                                        ignore_index=True)
    if args.top:
        df = df.head(args.top)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} ranked guides to {args.out}")
    print(df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
