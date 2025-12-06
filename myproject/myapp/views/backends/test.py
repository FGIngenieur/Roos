#---------------------------------------------------------------------------
#   Title : _Quotation Section Libraries_
#---------------------------------------------------------------------------

import subprocess

try:
    from typing import (
        Literal, Optional, Tuple, Dict, List
    )
except ImportError:
    subprocess.call(["pip3", "install", "typing"])
    from typing import (
        Literal, Optional, Tuple, Dict, List
    )
try:
    import pdfplumber
except ImportError:
    subprocess.call(["pip3", "install", "pdfplumber"])
    import pdfplumber

try:
    import csv
except ImportError:
    subprocess.call(["pip3", "install", "csv"])
    import csv

try:
    import re
except ImportError:
    subprocess.call(["pip3", "install", "re"])
    import re

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    subprocess.call(["pip3", "install", "sentence_transformers"])
    from sentence_transformers import SentenceTransformer, util

try:
    import torch
except ImportError:
    subprocess.call(["pip3", "install", "torch"])
    import torch

try:
    import json
except ImportError:
    subprocess.call(["pip3", "install", "json"])
    import json

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
except ImportError:
    subprocess.call(["pip3", "install", "scikit-learn"])
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.cluster import DBSCAN
import os

# test_clusterizer.py
from quoteClustering import *  # adjust import if needed

def main():
    # Training dataset: (sentence, numeric_target)
    train_data = [
        ("Le chat dort sur le canapé.", 1.0),
        ("Un chat dort sur le sofa.", 0.9),
        ("La voiture est en panne.", 10.0),
        ("La voiture ne fonctionne plus.", 9.7),
        ("J'aime manger des pâtes.", 2.0),
        ("Les pâtes sont mon plat préféré.", 1.8),
        ("La machine est cassée.", 11.0),
        ("J'adore les spaghetti.", 2.2),
    ]

    print("\n=== Initializing SemanticClusterizer ===")
    sc = SemanticClusterizer(model_name="sentence-transformers/LaBSE")

    print("\n=== Training model ===")
    sc.train(
        pairs=train_data,
        method="threshold",
        threshold=None,            # auto threshold search
        merge_threshold=0.80,
        split_threshold=0.45,
        auto_search_steps=7
    )

    print("\n=== CLUSTERS ===")
    for idx, cluster in enumerate(sc.clusters):
        sentences = [sc.sentences[i] for i in cluster]
        print(f"\nCluster {idx}:")
        for s in sentences:
            print(f"  - {s}")

    print("\n=== SINGLE PREDICTION TEST ===")
    test_sentence = "Le félin se repose sur le divan."
    result = sc.predict(test_sentence)
    print("\nSentence:", test_sentence)
    print("Predicted target:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Closest training sentence:", result["closest"])

    print("\n=== BATCH PREDICTION TEST ===")
    batch_sentences = [
        "Ma voiture est cassée.",
        "Je prépare des spaghetti.",
        "Le chat se repose tranquillement.",
    ]
    batch_results = sc.predict_batch(batch_sentences)

    for sent, res in zip(batch_sentences, batch_results):
        print("\nSentence:", sent)
        print("Predicted target:", res["prediction"])
        print("Confidence:", res["confidence"])
        print("Closest training sentence:", res["closest"])

    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()
