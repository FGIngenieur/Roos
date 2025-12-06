# semantic_mapping.py

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class SemanticMapping:
    """
    A minimal mapping model:
    - Train: store (sentence, numeric target) pairs
    - Predict: return the target of the most similar sentence
    - Save & load model, dataset, embeddings
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

        self.sentences: List[str] = []
        self.targets: np.ndarray = np.array([], dtype=float)

        # cache of embeddings (torch.Tensor)
        self._embeddings_cache: torch.Tensor = None

    # -------------------------------------
    # Embedding helper
    # -------------------------------------
    def _embed(
        self,
        sentences: List[str],
        batch_size: int = 64,
        convert_to_tensor: bool = True,
    ) -> torch.Tensor:
        if not sentences:
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty((0, dim), device=self.device)

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=convert_to_tensor,
            device=self.device,
        )

    # -------------------------------------
    # Train
    # -------------------------------------
    def train(self, pairs: List[Tuple[str, float]], batch_size: int = 64):
        if not pairs:
            return

        sents, targs = zip(*pairs)
        self.sentences.extend(list(sents))

        if self.targets.size == 0:
            self.targets = np.array(targs, dtype=float)
        else:
            self.targets = np.concatenate([self.targets, np.array(targs, dtype=float)])

        # compute embeddings
        self._embeddings_cache = self._embed(self.sentences, batch_size=batch_size)

    # -------------------------------------
    # Predict
    # -------------------------------------
    def predict(
        self,
        new_sentence: str,
        batch_size: int = 64,
    ) -> Dict[str, Any]:

        if len(self.sentences) == 0:
            raise RuntimeError("Model has no training data. Call train().")

        emb = self._embed([new_sentence], batch_size=batch_size)[0]

        sims = util.cos_sim(emb, self._embeddings_cache).cpu().numpy().flatten()

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_target = float(self.targets[best_idx])
        best_sentence = self.sentences[best_idx]

        return {
            "prediction": best_target,
            "similarity": best_sim,
            "nearest_sentence": best_sentence,
        }

    # -------------------------------------
    # Batch prediction
    # -------------------------------------
    def predict_batch(
        self,
        sentences: List[str],
        batch_size: int = 64,
    ) -> List[Dict[str, Any]]:

        if len(self.sentences) == 0:
            raise RuntimeError("Model has no training data. Call train().")

        embs = self._embed(sentences, batch_size=batch_size)
        results = []

        for emb in embs:
            sims = util.cos_sim(emb, self._embeddings_cache).cpu().numpy().flatten()
            best_idx = int(np.argmax(sims))

            results.append({
                "prediction": float(self.targets[best_idx]),
                "similarity": float(sims[best_idx]),
                "nearest_sentence": self.sentences[best_idx],
            })

        return results

    # -------------------------------------
    # Save
    # -------------------------------------
    def save(self, path: str) -> None:
        """
        Save model, metadata, dataset, and embedding cache into a folder.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Metadata
        metadata = {
            "model_name": self.model_name,
            "device": self.device,
        }
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Dataset
        with open(p / "dataset.pkl", "wb") as f:
            pickle.dump({"sentences": self.sentences, "targets": self.targets.tolist()}, f)

        # Embedding cache
        if self._embeddings_cache is not None:
            torch.save(self._embeddings_cache, p / "embeddings.pt")

    # -------------------------------------
    # Load
    # -------------------------------------
    @classmethod
    def load(cls, path: str) -> "SemanticMapping":
        """
        Load saved model, dataset, and embeddings.
        """
        p = Path(path)

        # Metadata
        with open(p / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        obj = cls(
            model_name=metadata.get("model_name", "sentence-transformers/LaBSE"),
            device=metadata.get("device", None),
        )

        # Dataset
        with open(p / "dataset.pkl", "rb") as f:
            ds = pickle.load(f)
            obj.sentences = ds["sentences"]
            obj.targets = np.array(ds["targets"], dtype=float)

        # Embeddings
        emb_path = p / "embeddings.pt"
        if emb_path.exists():
            obj._embeddings_cache = torch.load(emb_path, map_location=obj.device)
        else:
            # Recompute if missing
            obj._embeddings_cache = obj._embed(obj.sentences)

        return obj
