#!/usr/bin/env python
"""Step 1: download the MSU Rice Genome Annotation Project (Release 7.0) files.

Downloads the full genome FASTA (``all.con``) and gene annotation (``all.gff3``)
into ``data/msu_raw/``. The MSU host serves these over HTTPS (the ``http://``
URLs 301-redirect), so we let ``requests`` follow redirects.
"""
from __future__ import annotations

import sys

import requests

from dnabert_utils import GENOME_FASTA, GENOME_GFF3, MSU_RAW_DIR

BASE_URL = (
    "https://rice.uga.edu/pub/data/Eukaryotic_Projects/o_sativa/"
    "annotation_dbs/pseudomolecules/version_7.0/all.dir/"
)
FILES = {
    "all.con": GENOME_FASTA,   # full genome FASTA (~382 MB)
    "all.gff3": GENOME_GFF3,   # gene annotations
}
CHUNK = 1 << 20  # 1 MiB


def download(url: str, dest, expected_prefix: bytes | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size:,} bytes)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r         {done:,}/{total:,} bytes ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    if expected_prefix is not None:
        with open(tmp, "rb") as fh:
            head = fh.read(len(expected_prefix))
        if head != expected_prefix:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"{dest.name} did not start with {expected_prefix!r} (got {head!r})"
            )
    tmp.rename(dest)
    print(f"  [ok  ] saved {dest} ({dest.stat().st_size:,} bytes)")


def main() -> int:
    MSU_RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading MSU Rice Genome Annotation Project v7.0 files...")
    download(BASE_URL + "all.con", GENOME_FASTA, expected_prefix=b">Chr1")
    download(BASE_URL + "all.gff3", GENOME_GFF3)

    # Quick sanity report: count FASTA records (chromosomes/pseudomolecules).
    n_seqs = 0
    with open(GENOME_FASTA) as fh:
        for line in fh:
            if line.startswith(">"):
                n_seqs += 1
    print(f"\nVerification:")
    print(f"  all.con : {GENOME_FASTA.stat().st_size:,} bytes, {n_seqs} sequences")
    print(f"  all.gff3: {GENOME_GFF3.stat().st_size:,} bytes")
    if GENOME_GFF3.stat().st_size == 0:
        print("  ERROR: all.gff3 is empty", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
