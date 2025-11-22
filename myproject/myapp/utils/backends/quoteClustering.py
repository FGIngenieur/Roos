import os
cwd = os.getcwd()
os.chdir('../libs/')
from quotesLibs import *
os.chdir(cwd)



class SemanticClusterizer:
    _MIN_THRESHOLD = .65
    _MAX_THRESHOLD = .90
    """
    Advanced multilingual semantic clustering engine with:
    - GPU acceleration
    - Automatic threshold search
    - Incremental centroid updates
    - Cluster merging and splitting logic
    - Saving / loading cluster models
    """

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    def __init__(self, model_name="sentence-transformers/LaBSE", device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

        self.dataset: List[str] = []
        self.clusters: List[List[str]] = []
        self.centroids: List[torch.Tensor] = []
        self.method: Optional[str] = None
        self.threshold: Optional[float] = None



    # =====================================================================
    # GPU-ACCELERATED EMBEDDING
    # =====================================================================

    def embed_parallel(self, sentences, batch_size=32, num_workers=4):
        return self.model.encode(
            sentences,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            num_workers=num_workers,
            device=self.device
        )



    # =====================================================================
    # AUTOMATIC THRESHOLD SEARCH
    # =====================================================================

    def auto_threshold_search(
        self,
        sentences: List[str],
        min_thr: float = self._MIN_THRESHOLD,
        max_thr: float = self._MAX_THRESHOLD,
        steps: int = 9
    ) -> float:
        """
        Tests multiple thresholds and selects the one producing the best
        silhouette score.
        """

        print("🔍 Running automatic threshold search...")

        thresholds = np.linspace(min_thr, max_thr, steps)
        embs = self.embed_parallel(sentences)

        best_thr = None
        best_score = -1

        for thr in thresholds:
            clusters = self.cluster_threshold(sentences, thr)
            labels = self._clusters_to_labels(sentences, clusters)

            # Skip threshold values with only 1 cluster
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embs.cpu().numpy(), labels, metric="cosine")

            if score > best_score:
                best_score = score
                best_thr = thr

        print(f"✔ Best threshold found: {best_thr:.3f} (silhouette={best_score:.3f})")
        return best_thr



    # Helper: convert clusters to label vector
    def _clusters_to_labels(self, sentences, clusters):
        label_map = {}
        for label, cluster in enumerate(clusters):
            for s in cluster:
                label_map[s] = label
        return [label_map[s] for s in sentences]



    # =====================================================================
    # THRESHOLD-BASED CLUSTERING
    # =====================================================================

    def cluster_threshold(self, sentences, threshold):
        embs = self.embed_parallel(sentences)
        clusters = []
        assigned = set()

        for i, s in enumerate(sentences):
            if i in assigned:
                continue

            cluster = [s]
            assigned.add(i)

            for j in range(i+1, len(sentences)):
                if j in assigned:
                    continue
                sim = util.cos_sim(embs[i], embs[j]).item()
                if sim >= threshold:
                    cluster.append(sentences[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters



    # =====================================================================
    # INCREMENTAL CENTROID UPDATE
    # =====================================================================

    def update_centroid(self, cluster_index: int, new_sentence: str):
        new_emb = self.embed_parallel([new_sentence])[0]
        old_centroid = self.centroids[cluster_index]
        n = len(self.clusters[cluster_index])

        updated = (old_centroid * n + new_emb) / (n + 1)
        self.centroids[cluster_index] = updated



    # =====================================================================
    # MERGING & SPLITTING LOGIC
    # =====================================================================

    def merge_clusters(self, merge_threshold: float = 0.80):
        merged = True

        while merged:
            merged = False
            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    sim = util.cos_sim(self.centroids[i], self.centroids[j]).item()
                    if sim >= merge_threshold:
                        # merge j into i
                        self.clusters[i].extend(self.clusters[j])
                        del self.clusters[j]
                        del self.centroids[j]
                        merged = True
                        break
                if merged:
                    break



    def split_clusters(self, split_threshold: float = 0.45):
        new_clusters = []
        new_centroids = []

        for idx, cluster in enumerate(self.clusters):
            if len(cluster) == 1:
                new_clusters.append(cluster)
                new_centroids.append(self.centroids[idx])
                continue

            embs = self.embed_parallel(cluster)
            centroid = self.centroids[idx]

            sims = util.cos_sim(embs, centroid).cpu().tolist()
            avg_sim = np.mean(sims)

            if avg_sim >= split_threshold:
                new_clusters.append(cluster)
                new_centroids.append(centroid)
            else:
                # Split: put each point in its own cluster
                for i, s in enumerate(cluster):
                    new_clusters.append([s])
                    new_centroids.append(embs[i])

        self.clusters = new_clusters
        self.centroids = new_centroids



    # =====================================================================
    # TRAIN METHOD (WITH ALL NEW LOGIC)
    # =====================================================================

    def train(
        self,
        new_sentences: List[str],
        method="threshold",
        threshold: Optional[float] = None,
        merge_threshold=0.80,
        split_threshold=0.45
    ):

        # Add new sentences to dataset
        self.dataset.extend(new_sentences)

        # Auto-select threshold if needed
        if method == "threshold":
            if threshold is None:
                threshold = self.auto_threshold_search(self.dataset)

            self.threshold = threshold
            clusters = self.cluster_threshold(self.dataset, threshold)

        elif method == "dbscan":
            clusters_dict = self.cluster_dbscan(self.dataset)
            clusters = list(clusters_dict.values())

        else:
            raise ValueError("method must be 'threshold' or 'dbscan'")

        # Compute centroids
        centroids = []
        for cluster in clusters:
            emb = self.embed_parallel(cluster)
            centroids.append(emb.mean(dim=0))

        self.clusters = clusters
        self.centroids = centroids
        self.method = method
        self.threshold = threshold

        # Apply merging & splitting
        self.merge_clusters(merge_threshold)
        self.split_clusters(split_threshold)

        return self.clusters



    # =====================================================================
    # DBSCAN CLUSTERING
    # =====================================================================

    def cluster_dbscan(self, sentences, eps=0.35):
        embs = self.embed_parallel(sentences).cpu().numpy()
        db = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit(embs)

        clusters = {}
        for i, label in enumerate(db.labels_):
            clusters.setdefault(label, []).append(sentences[i])

        return clusters



    # =====================================================================
    # SAVE / LOAD MODEL TO DISK
    # =====================================================================

    def save(self, path: str):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        # Save metadata
        meta = {
            "model_name": self.model_name,
            "device": self.device,
            "dataset": self.dataset,
            "clusters": self.clusters,
            "method": self.method,
            "threshold": self.threshold,
        }

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Save centroids
        torch.save(self.centroids, path / "centroids.pt")
        print(f"✔ Model saved to {path}")


    @classmethod
    def load(cls, path: str):
        path = Path(path)

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(meta["model_name"], meta["device"])
        obj.dataset = meta["dataset"]
        obj.clusters = meta["clusters"]
        obj.method = meta["method"]
        obj.threshold = meta["threshold"]

        obj.centroids = torch.load(path / "centroids.pt", map_location=obj.device)

        print(f"✔ Model loaded from {path}")
        return obj
