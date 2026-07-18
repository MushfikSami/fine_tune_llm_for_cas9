# Rice CRISPR/Cas9 sgRNA Design Pipeline

End-to-end ML pipeline to **evaluate and rank CRISPR/Cas9 sgRNAs for rice**
(*Oryza sativa*, Nipponbare **MSU Release 7.0**), aimed at heat/salt-tolerance
gene editing.

The pipeline combines two independent scores:

- **On-target efficiency** — a genomic foundation model (**DNABERT-2-117M**) with
  a mean-pooled regression head, **LoRA**-fine-tuned on empirical Cas9 efficiency
  data (Doench/Azimuth 2016).
- **Off-target safety** — a **bowtie2** index of the full rice genome used to
  count promiscuous binding sites for each candidate spacer.

Results are written as a ranked CSV and an interactive, offline HTML report.

For the biology and chemistry behind each step (PAM recognition, R-loop
formation, HNH/RuvC cleavage, why borrowed efficiency labels transfer), see
[`WORKFLOW.md`](WORKFLOW.md).

---

## Layout

```
rice_crispr_pipeline/
├── data/
│   ├── msu_raw/     # downloaded MSU FASTA (all.con) + GFF3 (all.gff3)
│   ├── processed/   # Doench 2016 CSV + tokenized HuggingFace Arrow datasets
│   └── index/       # bowtie2 off-target index (rice_msu7.*.bt2)
├── scripts/
│   ├── dnabert_utils.py         # shared: paths + DNABERT-2 loader (compat patches)
│   ├── 01_download_genome.py    # fetch MSU 7.0 genome + annotation
│   ├── 02_preprocess_dnabert.py # bowtie2-build + tokenize training data
│   ├── 03_finetune_model.py     # LoRA regression fine-tune of DNABERT-2
│   ├── 04_design_sgrna.py       # scan PAMs, score, off-target, rank -> CSV
│   └── 05_visualize.py          # CSV -> interactive HTML report
├── models/dnabert_lora/         # saved LoRA adapter + tokenizer
├── requirements.txt
├── WORKFLOW.md                  # detailed workflow + biological/chemical mechanism
└── README.md
```

---

## Setup

Requires an NVIDIA GPU for fine-tuning/inference (validated on an RTX A6000).

```bash
conda create -n rice_crispr python=3.10 -y
conda install -n rice_crispr -c bioconda bowtie2 -y
conda run -n rice_crispr pip install -r requirements.txt
```

---

## Usage

Run the five stages in order:

```bash
# 1. Download the MSU 7.0 genome (all.con ~382 MB) + annotation (all.gff3)
conda run -n rice_crispr python scripts/01_download_genome.py

# 2. Build the bowtie2 off-target index + tokenize the Doench 2016 efficiency data
#    (add --with-mlm for optional rice genic domain-adaptation data)
conda run -n rice_crispr python scripts/02_preprocess_dnabert.py --threads 8

# 3. LoRA fine-tune DNABERT-2 as an efficiency regressor -> models/dnabert_lora/
conda run -n rice_crispr python scripts/03_finetune_model.py

# 4. Design + rank guides for a target gene (MSU id or FASTA)
conda run -n rice_crispr python scripts/04_design_sgrna.py --gene LOC_Os01g20160
#   or:  --fasta path/to/target_gene.fa

# 5. Render the interactive report
conda run -n rice_crispr python scripts/05_visualize.py   # -> designed_sgrnas.html
```

### Outputs

- **`designed_sgrnas.csv`** — ranked best→worst, columns: `gene, spacer, pam,
  strand, position, gc_percent, on_target, offtarget_count, final_score`.
- **`designed_sgrnas.html`** — self-contained interactive report (no internet
  needed): headline stat tiles, an efficiency-vs-off-target scatter with hover
  tooltips, an on-target histogram, and a sortable/filterable guide table with
  per-guide safety tiers. Supports light/dark mode.

### Key CLI options

| Script | Option | Purpose |
|---|---|---|
| `02_preprocess_dnabert.py` | `--threads N`, `--with-mlm`, `--test-frac` | index threads; build optional genic MLM set; train/test split |
| `03_finetune_model.py` | `--epochs`, `--lr`, `--batch-size`, `--lora-r` | training hyperparameters |
| `04_design_sgrna.py` | `--gene` / `--fasta`, `--w-on`, `--w-off`, `--top` | target; on/off-target weights; keep top-N |
| `05_visualize.py` | `--csv`, `--out` | input CSV / output HTML path |

The aggregate is `final_score = w_on * on_target − w_off * offtarget_count`
(defaults `w_on=1.0`, `w_off=0.1`).

---

## Model

- **Base:** `Zhihan1996/DNABERT-2-117M` (BPE tokenizer, vocab 4096), loaded with
  `trust_remote_code=True`.
- **Head:** masked mean-pooling over token embeddings → linear regressor (1
  output). *DNABERT-2's built-in CLS pooler is untrained, so a CLS-based head
  collapses to the label mean — mean-pooling is required.*
- **Adaptation:** LoRA (r=16, α=32) on `Wqkv` and `dense`; ~0.9 M trainable
  params (0.77%). The regression head is fully trained.
- **Data / metric:** Doench/Azimuth 2016 (train 4779 / test 531). A representative
  run reaches **Spearman ≈ 0.36 / Pearson ≈ 0.36** on the held-out test — moderate
  but real (a frozen-embedding Ridge baseline gives ~0.29).

---

## Notes & caveats

- **Library pins matter.** DNABERT-2's custom code is incompatible with
  transformers 5.x (it reads `config.pad_token_id`), so `transformers>=4.44,<5`
  is pinned. Its bundled Triton flash-attention kernel is also broken on modern
  Triton; `dnabert_utils.py` disables it at runtime and falls back to the model's
  pure-PyTorch attention path. Inference/training must run on CUDA.
- **On-target scores are a transferable-biochemistry prior** (Doench 2016 was
  measured in mammalian cells), **not** rice-validated ground truth — use them to
  *rank*, not as absolute cut rates. Retrain `03_finetune_model.py` on
  plant-specific efficiency data (e.g. PlantCRISPR / PC-Score) when available.
- **Off-target counting** via bowtie2 approximates but does not weight
  seed-vs-distal mismatch position or bulges the way CRISPR-specific tools
  (CFD, Cas-OFFinder) do; the count is capped by the bowtie2 `-k` limit.
- The MSU files are served over **HTTPS** (the `http://` URLs 301-redirect).
- Example target `LOC_Os01g20160` is **OsHKT1;5 (SKC1)**, a salt-tolerance Na⁺
  transporter.
