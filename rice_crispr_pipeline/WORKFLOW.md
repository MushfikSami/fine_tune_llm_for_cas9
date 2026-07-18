# WORKFLOW — Rice CRISPR/Cas9 sgRNA Design Pipeline

A step-by-step account of the pipeline, interleaving the **computational workflow**
with the **biological and chemical mechanisms** each step models. The goal is a
ranked list of single-guide RNAs (sgRNAs) that will drive efficient, specific
SpCas9 cleavage inside rice (*Oryza sativa*) heat/salt-tolerance genes.

---

## 0. Biological & chemical background

### 0.1 The CRISPR/Cas9 system

CRISPR/Cas9 is an RNA-guided DNA endonuclease adapted from the *Streptococcus
pyogenes* adaptive immune system. Two molecular parts do the work:

- **Cas9 protein** — a bilobed (REC + NUC) DNA endonuclease. The NUC lobe carries
  two nuclease domains: **RuvC** and **HNH**.
- **Single-guide RNA (sgRNA)** — an engineered fusion of the natural crRNA and
  tracrRNA. Its 5′ end contains a **20-nucleotide "spacer"** whose sequence is
  chosen by the experimenter; this is what our pipeline designs.

### 0.2 The chemistry of target recognition and cleavage

1. **PAM scanning.** Cas9 cannot bind DNA on spacer complementarity alone. It first
   interrogates the double helix for a **Protospacer Adjacent Motif (PAM)** —
   for SpCas9 this is **5′-NGG-3′** immediately 3′ of the 20 bp target
   ("protospacer") on the *non-target* strand. Two arginine residues
   (Arg1333/Arg1335) read the major-groove edges of the two G:C base pairs. No
   PAM → no stable binding. **This is why our designer scans for NGG.**

2. **R-loop formation.** Once a PAM is engaged, Cas9 locally melts the duplex and
   the sgRNA spacer base-pairs with the **target strand** (Watson–Crick H-bonds),
   forming an **R-loop** (RNA:DNA heteroduplex + displaced ssDNA). Pairing
   nucleates at the PAM-proximal "**seed**" region (~10 bp nearest the PAM);
   mismatches here abolish cleavage, whereas PAM-distal mismatches are often
   tolerated. **This asymmetry is the physical basis of off-target behaviour.**

3. **Concerted double-strand break.** Full R-loop formation triggers a
   conformational change that docks the HNH domain onto the target strand.
   - **HNH** cleaves the **target** strand.
   - **RuvC** cleaves the **non-target** strand.
   Both cut ~**3 bp upstream (5′) of the PAM**, leaving a predominantly blunt
   double-strand break (DSB). Catalysis is a metal-dependent phosphodiester
   hydrolysis (Mg²⁺-coordinated), breaking the sugar-phosphate backbone.

4. **Repair → edit.** The plant cell repairs the DSB mainly by
   **non-homologous end joining (NHEJ)**, which is error-prone and introduces
   indels that frameshift/knock out the gene — the usual goal when editing a
   tolerance gene — or by **homology-directed repair (HDR)** if a donor template
   is supplied for precise edits.

### 0.3 What makes a *good* guide — and what the model learns

Two independent properties determine a usable guide:

- **On-target efficiency** — how reliably Cas9 cleaves the intended site.
  Empirically this depends on the *sequence* of the spacer and its flanks:
  nucleotide identity at specific positions, GC content (very low GC → weak
  RNA:DNA duplex; very high GC → sticky/structured), absence of poly-T
  (`TTTT` is a Pol III terminator that truncates the sgRNA), and secondary
  structure of the spacer. These are **biochemical/thermodynamic rules** that
  are largely *organism-independent*, which is why a model trained on mammalian
  data transfers to plant targets.
- **Off-target specificity** — how many *other* genomic sites the guide could
  cleave, governed by seed-region complementarity elsewhere in the genome.

The pipeline scores these two axes separately (DNABERT-2 for on-target,
bowtie2 for off-target) and combines them.

### 0.4 Why a genomic language model (DNABERT-2)

**DNABERT-2** is a BERT-style transformer pretrained on multi-species genomes with
a masked-language-modelling objective over **Byte-Pair-Encoded (BPE)** DNA tokens.
Through pretraining it internalises statistical/structural regularities of DNA
(motifs, k-mer context, compositional bias). We exploit that representation: a
small **regression head** maps a guide's contextual embedding to a continuous
efficiency score. This is transfer learning — the biochemistry of Cas9 loading is
encoded implicitly in sequence, and the model has already learned to represent
sequence.

---

## 1. Step 1 — Environment setup

**Computational.** Create an isolated `rice_crispr` conda environment
(Python 3.10), install `bowtie2` (bioconda) for genome alignment and the Python
ML/bio stack (`torch`, `transformers<5`, `peft`, `datasets`, `accelerate`,
`biopython`, `pandas`, `scikit-learn`, `einops`) from `requirements.txt`.

**Why the pins.** DNABERT-2 ships custom modelling code written for the
transformers 4.x API and a Triton flash-attention kernel that both break on the
newest libraries; the pins and a runtime patch (see `scripts/dnabert_utils.py`)
keep it working. Fine-tuning runs on GPU (RTX A6000).

---

## 2. Step 2 — MSU genome acquisition (`01_download_genome.py`)

**Computational.** Stream-download the **MSU Rice Genome Annotation Project,
Release 7.0** (Nipponbare reference):
- `all.con` — the full genome FASTA (~382 MB, 12 chromosomes + ChrSy/ChrUn).
- `all.gff3` — gene/mRNA/exon/CDS/UTR annotations (~82 MB).

**Biological role.** The **reference genome** is the substrate for *both* halves of
the design:
1. It provides the **target-gene sequence** (via annotation coordinates) in which
   we search for guides.
2. It is the **entire searchable space for off-targets** — a guide's specificity
   is only meaningful relative to the whole genome it will be delivered into.
Using the Nipponbare MSU 7.0 build ensures coordinates and sequence match the
cultivar most rice functional-genomics work is anchored to.

---

## 3. Step 3 — Indexing & training-data tokenization (`02_preprocess_dnabert.py`)

### Task A — Off-target index (`bowtie2-build`)

**Computational.** Build a Burrows–Wheeler / FM-index of `all.con`
(`data/index/rice_msu7.*.bt2`).

**Biological role.** This compressed, searchable index lets us later ask, in
milliseconds, "**where else in the rice genome could this 20-mer bind?**" — i.e.
enumerate potential off-target protospacers. It is the genomic "search engine"
backing the specificity check.

### Task B — Efficiency regression dataset (Doench/Azimuth 2016)

**Computational.** Download the **Doench et al. 2016** dataset: thousands of
sgRNAs each annotated with an empirically measured, normalised (0–1) knockout
efficiency (`score_drug_gene_rank`). Each guide is provided as a **30-mer**
(4 bp 5′ context + 20 bp protospacer + 3 bp PAM + 3 bp 3′ context). Tokenize with
the DNABERT-2 BPE tokenizer and save as a train/test HuggingFace Arrow dataset.

**Biological role — why borrowed mammalian labels are valid.** No large,
labelled *rice* efficiency dataset exists. Doench 2016 measured efficiency by
flow-cytometry knockout assays in human/mouse cells. The signal it captures —
how spacer sequence and immediate flanks affect Cas9 loading, R-loop stability,
and cleavage — is **biochemistry of the Cas9–sgRNA–DNA complex**, not
species-specific gene regulation. Those rules (position-specific nucleotide
preferences, GC balance, avoidance of Pol III terminators) transfer across
organisms. The 30-mer window matters: the flanking bases carry part of the PAM
and nucleosome/context signal that modulate cleavage.

### Task B2 — Optional genic MLM domain-adaptation (`--with-mlm`)

**Computational.** Extract `gene`-feature sequences from `all.gff3` + `all.con`,
chunk and tokenize them.

**Biological role.** Optional continued pretraining on *rice* sequence nudges
DNABERT-2's representation toward rice's compositional/codon/motif statistics
before the efficiency head is trained — domain adaptation, not efficiency signal.

---

## 4. Step 4 — LoRA fine-tuning (`03_finetune_model.py`)

**Computational.**
- Load the DNABERT-2 backbone and attach a **masked mean-pooling + linear
  regression head** (one output). Mean-pooling over token embeddings is used
  because DNABERT-2's built-in CLS pooler is untrained and collapses.
- Wrap the attention/projection matrices (`Wqkv`, `dense`) with **LoRA**
  (Low-Rank Adaptation): freeze the 117 M backbone weights and train only small
  rank-16 update matrices plus the head (~0.9 M params, 0.77%). This slashes VRAM
  and prevents catastrophic overfitting of the small dataset.
- Optimise **mean-squared error** between predicted and measured efficiency, with
  warmup + cosine schedule, weight decay, and bf16. Evaluate with
  **Spearman/Pearson correlation** (rank agreement is what matters for ranking
  guides) and RMSE.

**Biological interpretation.** The head learns a function
`sequence context → probability the guide cleaves efficiently`. Because the input
includes the 20 bp spacer *and* flanks, the model can weigh seed-region
composition, GC content, and terminator motifs — the same physical determinants
described in §0.3. The saved LoRA adapter (`models/dnabert_lora/`) is the trained
"efficiency sense" applied later to rice guides.

---

## 5. Step 5 — Automated sgRNA designer (`04_design_sgrna.py`)

Given a target gene (MSU Gene ID resolved against the genome, or a FASTA):

### 5.1 PAM scan (Biopython)

**Computational.** Slide over both strands of the target sequence:
- **Forward strand:** every position where the dinucleotide after a base is `GG`
  (i.e. `NGG`) defines a PAM; the **20 bp immediately 5′** of it is the spacer.
- **Reverse strand:** a forward-strand `CCN` is an `NGG` on the complementary
  strand; the spacer is the reverse-complement of the 20 bp 3′ of the `CC`.

**Biological mechanism.** This directly mirrors Cas9's obligatory **PAM search**
(§0.2 step 1). Scanning both strands is essential because Cas9 can load onto
either strand — the PAM simply has to be on the strand opposite the one the sgRNA
pairs with. Each candidate is a physically real place Cas9 *could* engage.

### 5.2 On-target scoring (fine-tuned DNABERT-2)

**Computational.** For each candidate, extract the **20 bp spacer with 50 bp of
genomic flanking context**, orient it 5′→3′ with the spacer, tokenize, and run a
batched GPU forward pass through the LoRA regressor → an efficiency score.

**Biological mechanism.** The 50 bp context lets the model evaluate the local
sequence environment that influences chromatin accessibility, R-loop
thermodynamics, and the sequence-intrinsic cleavage propensity — the learned
proxy for "how well will Cas9 actually cut here?"

### 5.3 Off-target safety check (bowtie2)

**Computational.** Write all 20 bp spacers to FASTA and align them to the genome
index with mismatch-tolerant settings (`-N 1 -L 10 --score-min L,-0.6,-0.9`,
`-k 50`) so that near-matches (~up to 3–4 mismatches) are reported. Count
alignments per spacer; the one intended on-target locus is subtracted, leaving an
**off-target count**.

**Biological mechanism.** This enumerates other genomic loci where the guide's
seed region could still form a functional R-loop and cause an unintended DSB.
Because seed mismatches are the strongest determinant of specificity (§0.2 step
2), a permissive-but-bounded alignment approximates the set of biochemically
plausible off-target cut sites. A guide with many hits risks collateral edits and
is penalised.

### 5.4 Aggregate ranking

**Computational.** Combine the two axes:

```
final_score = w_on * on_target − w_off * offtarget_count
```

(default `w_on = 1.0`, `w_off = 0.1`), sort descending, and write
`designed_sgrnas.csv` with `spacer, pam, strand, position, gc_percent,
on_target, offtarget_count, final_score`.

**Biological rationale.** A useful genome-editing reagent must be **both potent
and specific**. The linear trade-off rewards high predicted cleavage while
subtracting a penalty for each extra genomic site the guide could hit —
operationalising the twin experimental goals of maximal on-target editing and
minimal off-target damage.

---

## 6. End-to-end data flow

```
MSU 7.0 genome (all.con, all.gff3)
        │
        ├─ bowtie2-build ───────────────► genome index  ──┐
        │                                                   │  (off-target search)
        │                                                   ▼
Doench 2016 (empirical Cas9 efficiency) ─ tokenize ─► HF dataset
        │                                                   │
        │                              LoRA fine-tune       │
        ▼                                    │              │
   DNABERT-2 backbone ──────────────────────┘              │
        │                                                   │
        ▼                                                   │
  LoRA efficiency regressor                                 │
        │                                                   │
Target gene ─ PAM scan (both strands) ─► candidate spacers ─┤
        │                                    │              │
        │              on-target score ◄─────┘              │
        │              off-target count ◄──────────────────┘
        ▼
  aggregate + rank ─► designed_sgrnas.csv
```

---

## 7. Interpreting the output

- **`on_target`** — model-predicted efficiency (higher = more likely to cut).
  A transferred biochemical prior, **not** a rice-validated absolute; use it to
  *rank*, not as a guaranteed cut rate.
- **`offtarget_count`** — number of additional genomic near-matches
  (0 is ideal; capped at the bowtie2 `-k` limit).
- **`gc_percent`** — sanity flag; ~40–70% is generally favourable for duplex
  stability without over-stickiness.
- **`strand` / `position`** — where in the target the guide sits, for cloning and
  validation.
- Always confirm top guides don't contain `TTTT` (Pol III terminator) and, for
  knockouts, prefer early-exon / conserved-domain positions.

---

## 8. Limitations & biological caveats

- **Species transfer.** Efficiency labels are mammalian; plant chromatin,
  temperature, and delivery differ. Treat scores as a prior. Retrain
  `03_finetune_model.py` on plant efficiency data (e.g. PlantCRISPR / PC-Score)
  when available.
- **Off-target model.** bowtie2 mismatch counting approximates, but does not
  weight, position-specific (seed vs distal) or bulge tolerances that
  CRISPR-specific tools (e.g. CFD score, Cas-OFFinder) model more faithfully.
- **No chromatin/epigenome features.** Real *in planta* efficiency also depends
  on DNA methylation and nucleosome occupancy not captured here.
- **Downstream biology.** A perfect cut still relies on the cell's NHEJ/HDR
  machinery and the resulting indel actually disrupting gene function — validate
  edits experimentally (e.g. amplicon sequencing) before drawing conclusions.
