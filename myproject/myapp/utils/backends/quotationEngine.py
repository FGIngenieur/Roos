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
from quoteClustering import *
from quoteMapping import *

class QuoteEngine:
    """
        The class describes engine object corresponding to the backend object
        for quotes and estimates automation.
    """
    def __init__(
        self,
        client_id : int,
        project_id : str,
        engine_model : Literal["mapping", "clustering"] = "mapping",
        __engine = None
    ) -> None:
        self._client_id = client_id
        self._project_id = project_id
        self.engine_model = engine_model
        
        if self.engine_model == 'mapping':
            self.__engine = SemanticMapping()
        else:
            self.__engine = SemanticClusterizer()
    
    @staticmethod
    def _get_correct_dataset(dataset : pd.DataFrame, description_col = "Description", target_col = "Unit_price") -> List[Tuple[str, float]]:
        sub_dataset = dataset[[description_col, target_col]]
        output = [(row[description_col], row[target_col]) for idx, row in sub_dataset.iterrows()]
        return output

    def init_engine(self, dataset : pd.DataFrame):
        if self.engine_model == 'mapping':
            self.__engine.train(dataset)
        else:
            self.__engine_train(dataset)
