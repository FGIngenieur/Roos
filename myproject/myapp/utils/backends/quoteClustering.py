from __future__ import annotations
#---------------------------------------------------------------------------
#   Title : _Quotation Section Libraries_
#---------------------------------------------------------------------------
import os
# semantic_clusterizer.py

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


class SemanticClusterizer:
    """
    Clean, opinionated SemanticClusterizer.

    Key features:
    - GPU-accelerated batch embeddings (SentenceTransformer)
    - Train from list of (sentence, numeric_target) pairs
    - Threshold-based greedy clustering OR DBSCAN (automatic cluster count)
    - Automatic threshold search (silhouette)
    - Incremental centroid updates, cluster merge + split heuristics
    - Weighted-KNN prediction *inside* the chosen cluster (cosine distance weight)
    - Batch prediction
    - Save / load model (metadata + clusters + centroids + dataset)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: Optional[str] = None,
        target_name: str = "target",
    ):
        # device auto-detect
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

        # dataset storage
        self.target_name = target_name
        self.sentences: List[str] = []
        self.targets: np.ndarray = np.array([], dtype=float)

        # cluster representation (indices into self.sentences)
        self.clusters: List[List[int]] = []
        self.centroids: List[torch.Tensor] = []

        # metadata
        self.method: Optional[str] = None
        self.threshold: Optional[float] = None

        # caches
        self._embeddings_cache: Optional[torch.Tensor] = None  # tensor on self.device

    # -------------------------
    # Embedding helpers
    # -------------------------
    def _embed(
        self,
        sentences: List[str],
        batch_size: int = 64,
        convert_to_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Wrapper around SentenceTransformer.encode, compatible with all models.
        Returns a tensor located on the configured device.
        """
        if not sentences:
            # return empty tensor of shape (0, dim)
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty((0, dim), device=self.device)

        emb = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=convert_to_tensor,
            device=self.device,
        )
        return emb

    def _ensure_embeddings_cache(self, batch_size: int = 64) -> torch.Tensor:
        """
        Compute and cache embeddings for self.sentences if not present.
        """
        if self._embeddings_cache is None:
            if not self.sentences:
                dim = self.model.get_sentence_embedding_dimension()
                self._embeddings_cache = torch.empty((0, dim), device=self.device)
            else:
                self._embeddings_cache = self._embed(self.sentences, batch_size=batch_size)
        return self._embeddings_cache

    # -------------------------
    # Dataset management
    # -------------------------
    def add_pairs(self, pairs: List[Tuple[str, float]]) -> None:
        """
        Append (sentence, numeric_target) pairs to the dataset.
        Call train(...) to recompute clusters.
        """
        if not pairs:
            return
        sents, targs = zip(*pairs)
        self.sentences.extend(sents)
        if self.targets.size == 0:
            self.targets = np.array(list(targs), dtype=float)
        else:
            self.targets = np.concatenate([self.targets, np.array(list(targs), dtype=float)])
        # clear embedding cache (will be recomputed on next train/predict)
        self._embeddings_cache = None

    # -------------------------
    # Clustering algorithms
    # -------------------------
    def _cluster_threshold(
        self,
        sentences: List[str],
        threshold: float,
        batch_size: int = 64,
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        Greedy clustering: iterate sentences, assign to first centroid with sim >= threshold,
        otherwise create a new cluster.
        Returns clusters (lists of indices) and centroids (tensor list).
        """
        embs = self._embed(sentences, batch_size=batch_size)  # tensor
        clusters: List[List[int]] = []
        centroids: List[torch.Tensor] = []

        for i, emb in enumerate(embs):
            assigned = False
            for cid, cent in enumerate(centroids):
                sim = util.cos_sim(emb, cent).item()
                if sim >= threshold:
                    clusters[cid].append(i)
                    # incremental centroid update
                    n = len(clusters[cid])
                    centroids[cid] = (cent * (n - 1) + emb) / n
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
                centroids.append(emb)
        return clusters, centroids

    def _cluster_dbscan(
        self,
        sentences: List[str],
        eps: float = 0.35,
        batch_size: int = 64,
    ) -> Tuple[List[List[int]], List[torch.Tensor]]:
        """
        DBSCAN clustering in embedding space using cosine metric (via sklearn).
        """
        embs = self._embed(sentences, batch_size=batch_size).cpu().numpy()
        db = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(embs)
        labels = db.labels_
        clusters_map: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            clusters_map.setdefault(int(lab), []).append(idx)
        clusters = list(clusters_map.values())

        centroids: List[torch.Tensor] = []
        for c in clusters:
            c_embs = torch.tensor(embs[c], device=self.device)
            centroids.append(c_embs.mean(dim=0))
        return clusters, centroids

    # -------------------------
    # Auto threshold search
    # -------------------------
    def auto_threshold_search(
        self,
        min_thr: float = 0.50,
        max_thr: float = 0.90,
        steps: int = 9,
        batch_size: int = 64,
    ) -> float:
        """
        Try multiple thresholds and pick the one with best silhouette score.
        Returns chosen threshold (defaults to 0.75 if no good threshold found).
        """
        if len(self.sentences) < 2:
            return 0.75
        embs = self._ensure_embeddings_cache(batch_size=batch_size).cpu().numpy()
        best_thr = 0.75
        best_score = -1.0
        for thr in np.linspace(min_thr, max_thr, steps):
            clusters, _ = self._cluster_threshold(self.sentences, threshold=thr, batch_size=batch_size)
            # convert clusters to label array
            labels = np.empty(len(self.sentences), dtype=int)
            for lab_idx, cluster in enumerate(clusters):
                for idx in cluster:
                    labels[idx] = lab_idx
            if len(set(labels)) < 2:
                continue
            try:
                score = silhouette_score(embs, labels, metric="cosine")
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        return best_thr

    # -------------------------
    # Merge / Split clusters
    # -------------------------
    def _merge_clusters(self, merge_threshold: float = 0.80) -> None:
        """
        Merge clusters whose centroids have cosine similarity >= merge_threshold.
        """
        merged = True
        while merged:
            merged = False
            n = len(self.clusters)
            for i in range(n):
                for j in range(i + 1, n):
                    if i >= len(self.centroids) or j >= len(self.centroids):
                        continue
                    sim = util.cos_sim(self.centroids[i], self.centroids[j]).item()
                    if sim >= merge_threshold:
                        # merge j into i
                        self.clusters[i].extend(self.clusters[j])
                        # recompute centroid from all members
                        idxs = self.clusters[i]
                        embs = self._ensure_embeddings_cache()[idxs]
                        self.centroids[i] = embs.mean(dim=0)
                        # remove j
                        del self.clusters[j]
                        del self.centroids[j]
                        merged = True
                        break
                if merged:
                    break

    def _split_clusters(self, split_threshold: float = 0.45, batch_size: int = 64) -> None:
        """
        Split clusters that have low average similarity to their centroid.
        Current fallback: split into singletons when below threshold.
        """
        new_clusters: List[List[int]] = []
        new_centroids: List[torch.Tensor] = []
        for idx, cluster in enumerate(self.clusters):
            if len(cluster) <= 1:
                new_clusters.append(cluster)
                new_centroids.append(self.centroids[idx])
                continue
            embs = self._ensure_embeddings_cache(batch_size=batch_size)[cluster]
            centroid = self.centroids[idx]
            sims = util.cos_sim(embs, centroid).cpu().numpy().flatten()
            avg_sim = float(np.mean(sims))
            if avg_sim >= split_threshold:
                new_clusters.append(cluster)
                new_centroids.append(centroid)
            else:
                # fallback: split into singletons
                for i in cluster:
                    new_clusters.append([i])
                    new_centroids.append(self._ensure_embeddings_cache(batch_size=batch_size)[i])
        self.clusters = new_clusters
        self.centroids = new_centroids

    # -------------------------
    # Train / Retrain
    # -------------------------
    def train(
        self,
        pairs: List[Tuple[str, float]],
        method: str = "threshold",
        threshold: Optional[float] = None,
        dbscan_eps: float = 0.35,
        auto_search_steps: int = 9,
        batch_size: int = 64,
        merge_threshold: float = 0.80,
        split_threshold: float = 0.45,
    ) -> List[List[int]]:
        """
        Append `pairs` to dataset and recompute clusters.
        Returns computed clusters (lists of indices).
        """
        # append data
        self.add_pairs(pairs)

        # ensure embeddings are present
        self._ensure_embeddings_cache(batch_size=batch_size)

        if method == "threshold":
            if threshold is None:
                threshold = self.auto_threshold_search(steps=auto_search_steps, batch_size=batch_size)
            clusters, centroids = self._cluster_threshold(self.sentences, threshold=threshold, batch_size=batch_size)
            self.method = "threshold"
            self.threshold = threshold
        elif method == "dbscan":
            clusters, centroids = self._cluster_dbscan(self.sentences, eps=dbscan_eps, batch_size=batch_size)
            self.method = "dbscan"
            self.threshold = None
        else:
            raise ValueError("method must be 'threshold' or 'dbscan'")

        # store clusters/centroids as indices into self.sentences
        self.clusters = clusters
        self.centroids = centroids

        # merge / split postprocessing
        self._merge_clusters(merge_threshold=merge_threshold)
        self._split_clusters(split_threshold=split_threshold, batch_size=batch_size)

        return self.clusters

    # -------------------------
    # Prediction (weighted KNN inside cluster)
    # -------------------------
    @staticmethod
    def _cosine_distance_from_sim(sim: float) -> float:
        """Distance in [0, 2] derived from cosine similarity in [-1,1]. We use 1 - sim."""
        return float(max(0.0, 1.0 - sim))

    def _choose_cluster_for_emb(self, emb: torch.Tensor) -> int:
        if not self.centroids:
            raise RuntimeError("No clusters available. Call train() first.")
        sims = [util.cos_sim(emb, c).item() for c in self.centroids]
        return int(np.argmax(sims))

    def predict(
        self,
        sentence: str,
        k: Optional[int] = None,
        batch_size: int = 64,
        epsilon: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Predict numeric target for a single sentence.
        Returns dict: {"prediction": float, "confidence": float, "closest": str}
        """
        emb = self._embed([sentence], batch_size=batch_size)[0]
        cluster_idx = self._choose_cluster_for_emb(emb)
        member_idxs = self.clusters[cluster_idx]
        member_embs = self._ensure_embeddings_cache(batch_size=batch_size)[member_idxs]
        member_targets = self.targets[np.array(member_idxs)]

        sims = util.cos_sim(emb, member_embs).cpu().numpy().flatten()
        dists = 1.0 - sims  # cosine distance

        # weights = 1/(distance + eps)
        weights = 1.0 / (dists + epsilon)

        if k is not None:
            top = np.argsort(dists)[:k]
            weights_k = weights[top]
            targets_k = member_targets[top]
            pred = float(np.sum(weights_k * targets_k) / np.sum(weights_k))
            confidence = float(np.mean(sims[top]))
            closest = self.sentences[member_idxs[int(top[0])]]
            return {"prediction": pred, "confidence": confidence, "closest": closest}
        else:
            pred = float(np.sum(weights * member_targets) / np.sum(weights))
            confidence = float(np.mean(sims))
            closest_idx = int(np.argmax(sims))
            closest = self.sentences[member_idxs[closest_idx]]
            return {"prediction": pred, "confidence": confidence, "closest": closest}

    def predict_batch(
        self,
        sentences: List[str],
        k: Optional[int] = None,
        batch_size: int = 64,
        epsilon: float = 1e-6,
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction returning list of dicts with keys:
        - prediction: float
        - confidence: float (mean similarity to used neighbors)
        - closest: str
        """
        embs = self._embed(sentences, batch_size=batch_size)
        results: List[Dict[str, Any]] = []

        # precompute member embeddings by cluster (cache)
        cluster_member_embs = [self._ensure_embeddings_cache(batch_size=batch_size)[c] for c in self.clusters]
        cluster_member_idxs = [c for c in self.clusters]

        for emb in embs:
            # choose best cluster by centroid similarity
            centroid_sims = [util.cos_sim(emb, c).item() for c in self.centroids]
            best_cluster = int(np.argmax(centroid_sims))
            member_idxs = cluster_member_idxs[best_cluster]
            member_embs = cluster_member_embs[best_cluster]
            member_targets = self.targets[np.array(member_idxs)]

            sims = util.cos_sim(emb, member_embs).cpu().numpy().flatten()
            dists = 1.0 - sims
            weights = 1.0 / (dists + epsilon)

            if k is not None:
                top = np.argsort(dists)[:k]
                weights_k = weights[top]
                targets_k = member_targets[top]
                pred = float(np.sum(weights_k * targets_k) / np.sum(weights_k))
                conf = float(np.mean(sims[top]))
                closest = self.sentences[member_idxs[int(top[0])]]
            else:
                pred = float(np.sum(weights * member_targets) / np.sum(weights))
                conf = float(np.mean(sims))
                closest = self.sentences[member_idxs[int(np.argmax(sims))]]

            results.append({"prediction": pred, "confidence": conf, "closest": closest})

        return results

    # -------------------------
    # Incremental centroid updates
    # -------------------------
    def incrementally_add_pairs_and_update(
        self,
        pairs: List[Tuple[str, float]],
        batch_size: int = 64,
    ) -> None:
        """
        Adds pairs to dataset and updates centroids incrementally for clusters where new
        points have been added (simple approach: re-run train for small datasets recommended).
        """
        start_len = len(self.sentences)
        self.add_pairs(pairs)
        # recompute embeddings cache
        self._ensure_embeddings_cache(batch_size=batch_size)
        # naive: recompute centroids fully for each cluster
        for cid, c in enumerate(self.clusters):
            embs = self._ensure_embeddings_cache(batch_size=batch_size)[c]
            self.centroids[cid] = embs.mean(dim=0)

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {
            "model_name": self.model_name,
            "device": self.device,
            "target_name": self.target_name,
            "method": self.method,
            "threshold": self.threshold,
            "sentences_count": len(self.sentences),
        }
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        with open(p / "clusters.pkl", "wb") as f:
            pickle.dump(self.clusters, f)

        # centroids
        torch.save(self.centroids, p / "centroids.pt")

        with open(p / "dataset.pkl", "wb") as f:
            pickle.dump({"sentences": self.sentences, "targets": self.targets.tolist()}, f)

    @classmethod
    def load(cls, path: str) -> "SemanticClusterizer":
        p = Path(path)
        with open(p / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(model_name=meta.get("model_name", "sentence-transformers/LaBSE"), device=meta.get("device", None), target_name=meta.get("target_name", "target"))
        obj.method = meta.get("method", None)
        obj.threshold = meta.get("threshold", None)

        with open(p / "clusters.pkl", "rb") as f:
            obj.clusters = pickle.load(f)

        obj.centroids = torch.load(p / "centroids.pt", map_location=obj.device)

        with open(p / "dataset.pkl", "rb") as f:
            ds = pickle.load(f)
            obj.sentences = ds["sentences"]
            obj.targets = np.array(ds["targets"], dtype=float)

        # precompute embeddings cache
        obj._embeddings_cache = None
        obj._ensure_embeddings_cache()
        return obj
