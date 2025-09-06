import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.taxonomy import canonical_labels

MODEL_ID = os.getenv("ZSL_MODEL", "facebook/bart-large-mnli")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ZeroShotRisk:
    """
    Zero-shot classifier using an MNLI model (e.g., BART-MNLI).
    For each label in the taxonomy, compute entailment scores.
    """
    def __init__(self, labels=None, model_id: str | None = None):
        self.labels = labels or canonical_labels()
        mid = model_id or MODEL_ID
        self.tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(mid).to(DEVICE).eval()

    @torch.inference_mode()
    def classify(self, texts: list[str], top_k: int = 3):
        out = []
        for t in texts:
            scores = []
            for lab in self.labels:
                hyp = f"This text is about {lab}."
                enc = self.tok(t, hyp, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                logits = self.mdl(**enc).logits[0]  # [contradiction, neutral, entailment]
                prob = torch.softmax(logits, dim=-1)[-1].item()
                scores.append((lab, float(prob)))
            scores.sort(key=lambda x: x[1], reverse=True)
            out.append(scores[:top_k])
        return out
