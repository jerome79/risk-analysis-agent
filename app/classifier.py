import os
import math
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.taxonomy import canonical_labels

# ---- Perf/control knobs (safe defaults; override in .env) -------------------
MODEL_ID   = os.getenv("ZSL_MODEL", "facebook/bart-large-mnli")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = int(os.getenv("ZSL_MAX_LEN", "512"))
LBL_BATCH  = int(os.getenv("ZSL_LABEL_BATCH", "16"))   # how many labels to score per forward pass
TORCH_NUM  = int(os.getenv("TORCH_NUM_THREADS", "4"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(TORCH_NUM)


class ZeroShotRisk:
    """
    Zero-shot risk classifier using an MNLI model (e.g., BART-MNLI).
    For each text, we score entailment for each taxonomy label:
        premise  = text chunk
        hypothesis = "This text is about <LABEL>."
    We batch labels to speed up inference.
    """

    def __init__(self, labels: List[str] | None = None, model_id: str | None = None):
        self.labels = labels or canonical_labels()
        mid = model_id or MODEL_ID
        self.tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(mid).to(DEVICE).eval()

    # ------------------------ internal helpers -------------------------------

    @torch.inference_mode()
    def _score_one_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Score all labels for a single text, batching labels for speed.
        Returns: list of (label, entailment_prob) for every label (unsorted).
        """
        scores: list[tuple[str, float]] = []
        n = len(self.labels)
        # batch labels to avoid huge single forward + keep memory steady
        for start in range(0, n, LBL_BATCH):
            chunk_labels = self.labels[start : start + LBL_BATCH]
            # Build paired inputs: (premise=text, hypothesis="This text is about <label>.")
            premises   = [text] * len(chunk_labels)
            hypotheses = [f"This text is about {lab}." for lab in chunk_labels]

            enc = self.tok(
                premises,
                hypotheses,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}
            logits = self.mdl(**enc).logits  # shape [B, 3] = [contradiction, neutral, entailment]
            probs  = torch.softmax(logits, dim=-1)[:, -1].tolist()  # entailment prob per label

            scores.extend(zip(chunk_labels, map(float, probs)))

        return scores

    # ------------------------ public APIs ------------------------------------

    @torch.inference_mode()
    def classify(self, texts: List[str], top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Backwards-compatible with your current UI:
          returns, for each text, the top_k labels sorted by score desc.
        """
        out: list[list[tuple[str, float]]] = []
        for t in texts:
            scores = self._score_one_text(t)
            scores.sort(key=lambda x: x[1], reverse=True)
            out.append(scores[:top_k])
        return out

    @torch.inference_mode()
    def classify_threshold(
        self,
        texts: List[str],
        threshold: float = 0.5,
        max_labels: int | None = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Multi-label style: keep all labels with score >= threshold.
        Optionally cap the number returned with max_labels.
        """
        out: list[list[tuple[str, float]]] = []
        for t in texts:
            scores = self._score_one_text(t)
            keep = [(lab, sc) for lab, sc in scores if sc >= threshold]
            keep.sort(key=lambda x: x[1], reverse=True)
            if max_labels is not None:
                keep = keep[:max_labels]
            out.append(keep)
        return out
