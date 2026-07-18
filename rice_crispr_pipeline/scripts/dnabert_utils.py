"""Shared helpers for loading DNABERT-2 and locating pipeline paths.

DNABERT-2-117M ships custom modeling code that is incompatible with two parts of
a modern stack:

1. It reads ``config.pad_token_id`` -> requires ``transformers < 5``.
2. Its bundled Triton flash-attention kernel uses APIs removed in recent Triton
   (``tl.dot(trans_b=...)``) and crashes on the CUDA forward pass.

``load_dnabert`` handles (2) by nulling the module-global ``flash_attn_qkvpacked_func``
after the model is constructed, which routes attention through the model's own
pure-PyTorch fallback path. See ``bert_layers.BertUnpadSelfAttention.forward``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_NAME = "Zhihan1996/DNABERT-2-117M"

# --- Pipeline paths -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MSU_RAW_DIR = DATA_DIR / "msu_raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = ROOT / "models"
LORA_DIR = MODELS_DIR / "dnabert_lora"

GENOME_FASTA = MSU_RAW_DIR / "all.con"
GENOME_GFF3 = MSU_RAW_DIR / "all.gff3"
INDEX_PREFIX = INDEX_DIR / "rice_msu7"

# LoRA target modules for DNABERT-2's fused attention (verified via named_modules)
LORA_TARGET_MODULES = ["Wqkv", "dense"]

# Keep HF quiet/offline-friendly by default
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _disable_broken_triton_flash_attn() -> None:
    """Force DNABERT-2's PyTorch attention path (bundled Triton kernel is broken)."""
    patched = False
    for name, mod in list(sys.modules.items()):
        if name.endswith("bert_layers") and hasattr(mod, "flash_attn_qkvpacked_func"):
            mod.flash_attn_qkvpacked_func = None
            patched = True
    return patched


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


class DNABERTRegressor:
    """Namespace holder; the real class is built lazily to avoid importing torch early."""


def _build_regressor_class():
    import torch
    import torch.nn as nn

    class _DNABERTRegressor(nn.Module):
        """DNABERT-2 backbone + masked mean-pooling + linear regression head.

        DNABERT-2's built-in CLS pooler is randomly initialised (its MLM
        pretraining never trained a pooler), so a CLS-based classification head
        collapses to predicting the label mean. Mean-pooling the token
        embeddings gives a usable sequence representation for regression.
        """

        def __init__(self, backbone, hidden_size: int, dropout: float = 0.1):
            super().__init__()
            self.backbone = backbone
            self.dropout = nn.Dropout(dropout)
            self.regressor = nn.Linear(hidden_size, 1)
            self.config = backbone.config

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out[0]  # (B, T, H)
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            logits = self.regressor(self.dropout(pooled)).squeeze(-1)
            loss = None
            if labels is not None:
                loss = nn.functional.mse_loss(logits, labels.to(logits.dtype))
            return {"loss": loss, "logits": logits}

    return _DNABERTRegressor


def load_dnabert(num_labels: int = 1, model_name: str = MODEL_NAME, dropout: float = 0.1):
    """Load DNABERT-2 as a mean-pooled regression model with the Triton fix applied.

    Returns ``(tokenizer, model)``. ``num_labels`` is accepted for API symmetry;
    only single-output regression (efficiency score) is implemented.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    _disable_broken_triton_flash_attn()
    hidden_size = backbone.config.hidden_size
    RegressorCls = _build_regressor_class()
    model = RegressorCls(backbone, hidden_size, dropout=dropout)
    return tokenizer, model


def load_finetuned_regressor(adapter_dir=LORA_DIR, model_name: str = MODEL_NAME):
    """Rebuild the base regressor and load the saved LoRA adapter for inference.

    Returns ``(tokenizer, model)`` with the adapter merged/applied and the model
    set to eval mode. Load the tokenizer from the adapter dir so it matches training.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    adapter_dir = str(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    _, base = load_dnabert(model_name=model_name)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model
