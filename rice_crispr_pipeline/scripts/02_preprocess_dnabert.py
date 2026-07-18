#!/usr/bin/env python
"""Step 2: build the off-target index and tokenize training data.

Task A -- Off-target index: run ``bowtie2-build`` on the genome FASTA (``all.con``)
          to create the alignment index in ``data/index/``.

Task B -- Efficiency regression data (primary): download the Doench/Azimuth 2016
          empirical Cas9 efficiency dataset (30-mer context + normalized
          ``score_drug_gene_rank`` in [0, 1]), tokenize with the DNABERT-2
          tokenizer, and save a train/test HuggingFace Arrow dataset in
          ``data/processed/efficiency/``.

Task B2 -- Optional MLM domain-adaptation data (``--with-mlm``): extract rice
          genic sequences from ``all.gff3`` + ``all.con`` and tokenize them into
          ``data/processed/rice_genic/`` for optional pretraining.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import pandas as pd
import requests
from datasets import Dataset, DatasetDict

from dnabert_utils import (GENOME_FASTA, GENOME_GFF3, INDEX_DIR, INDEX_PREFIX,
                           PROCESSED_DIR, load_tokenizer)

# Doench/Azimuth 2016 dataset: 30-mer guides with normalized efficiency ranks.
DOENCH_URL = (
    "https://raw.githubusercontent.com/MicrosoftResearch/Azimuth/master/"
    "azimuth/data/FC_plus_RES_withPredictions.csv"
)
SEQ_COL = "30mer"
LABEL_COL = "score_drug_gene_rank"  # already in [0, 1]
MAX_TOKEN_LEN = 96


# --------------------------------------------------------------------------
# Task A: off-target index
# --------------------------------------------------------------------------
def build_bowtie2_index(threads: int) -> None:
    print("[Task A] Building bowtie2 index...")
    if not GENOME_FASTA.exists():
        sys.exit(f"Missing genome FASTA: {GENOME_FASTA}. Run 01_download_genome.py first.")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(INDEX_DIR.glob("rice_msu7*.bt2*"))
    if existing:
        print(f"  [skip] index already exists ({len(existing)} .bt2 files)")
        return
    if shutil.which("bowtie2-build") is None:
        sys.exit("bowtie2-build not found on PATH (conda install -c bioconda bowtie2).")

    cmd = ["bowtie2-build", "--threads", str(threads),
           str(GENOME_FASTA), str(INDEX_PREFIX)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    built = list(INDEX_DIR.glob("rice_msu7*.bt2*"))
    print(f"  [ok] built {len(built)} index files in {INDEX_DIR}")


# --------------------------------------------------------------------------
# Task B: efficiency regression dataset
# --------------------------------------------------------------------------
def build_efficiency_dataset(tokenizer, test_frac: float, seed: int) -> None:
    print("[Task B] Building efficiency regression dataset (Doench 2016)...")
    out_dir = PROCESSED_DIR / "efficiency"
    if (out_dir / "dataset_dict.json").exists():
        print(f"  [skip] {out_dir} already exists")
        return

    raw = PROCESSED_DIR / "doench2016.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print(f"  [get] {DOENCH_URL}")
        r = requests.get(DOENCH_URL, timeout=60)
        r.raise_for_status()
        raw.write_bytes(r.content)
    df = pd.read_csv(raw)
    if SEQ_COL not in df.columns or LABEL_COL not in df.columns:
        sys.exit(f"Unexpected columns in Doench CSV: {list(df.columns)}")

    df = df[[SEQ_COL, LABEL_COL]].dropna()
    df = df.rename(columns={SEQ_COL: "sequence", LABEL_COL: "label"})
    df["sequence"] = df["sequence"].str.upper().str.strip()
    df = df[df["sequence"].str.fullmatch(r"[ACGT]+")]
    df["label"] = df["label"].astype("float32")
    print(f"  {len(df)} labelled guides, label range "
          f"[{df.label.min():.3f}, {df.label.max():.3f}]")

    def tok(batch):
        return tokenizer(batch["sequence"], truncation=True,
                         max_length=MAX_TOKEN_LEN)

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.map(tok, batched=True)
    split = ds.train_test_split(test_size=test_frac, seed=seed)
    DatasetDict(train=split["train"], test=split["test"]).save_to_disk(str(out_dir))
    print(f"  [ok] saved {out_dir} "
          f"(train={split['train'].num_rows}, test={split['test'].num_rows})")


# --------------------------------------------------------------------------
# Task B2: optional MLM domain-adaptation dataset from genic regions
# --------------------------------------------------------------------------
def build_genic_mlm_dataset(tokenizer, max_genes: int, chunk: int, seed: int) -> None:
    print("[Task B2] Building rice genic MLM dataset...")
    from Bio import SeqIO

    out_dir = PROCESSED_DIR / "rice_genic"
    if (out_dir / "dataset_info.json").exists():
        print(f"  [skip] {out_dir} already exists")
        return

    print("  Indexing genome FASTA...")
    fasta = SeqIO.index(str(GENOME_FASTA), "fasta")

    seqs: list[str] = []
    n_genes = 0
    with open(GENOME_GFF3) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            seqid, start, end = f[0], int(f[3]), int(f[4])
            if seqid not in fasta:
                continue
            sub = str(fasta[seqid].seq[start - 1:end]).upper()
            # window the gene into fixed-size chunks for the LM
            for i in range(0, max(1, len(sub) - chunk + 1), chunk):
                piece = sub[i:i + chunk]
                if len(piece) >= chunk // 2 and set(piece) <= set("ACGT"):
                    seqs.append(piece)
            n_genes += 1
            if max_genes and n_genes >= max_genes:
                break
    print(f"  {n_genes} genes -> {len(seqs)} sequence chunks")

    ds = Dataset.from_dict({"sequence": seqs})
    ds = ds.map(lambda b: tokenizer(b["sequence"], truncation=True,
                                    max_length=MAX_TOKEN_LEN), batched=True)
    ds.save_to_disk(str(out_dir))
    print(f"  [ok] saved {out_dir} ({ds.num_rows} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-mlm", action="store_true",
                    help="also build the optional genic MLM domain-adaptation set")
    ap.add_argument("--max-genes", type=int, default=2000,
                    help="cap genes for the MLM set (0 = all)")
    ap.add_argument("--mlm-chunk", type=int, default=300)
    args = ap.parse_args()

    build_bowtie2_index(args.threads)
    tokenizer = load_tokenizer()
    build_efficiency_dataset(tokenizer, args.test_frac, args.seed)
    if args.with_mlm:
        build_genic_mlm_dataset(tokenizer, args.max_genes, args.mlm_chunk, args.seed)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
