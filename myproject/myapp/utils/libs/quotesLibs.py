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
