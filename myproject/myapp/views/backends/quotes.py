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

from quotationEngine import *
from dataLoader import *

class QuoteProject:
    """
    A template class for handling data, performing actions,
    and encapsulating related logic.
    """
    _PUBLIC_ID_REGEX = re.compile(r"^[0-9a-fA-F]{7}[0-9a-fA-F]{2}$")

    def __init__(
                self,
                project_name : str,
                client_id : int,
                project_type : Optional[Literal["estimate", "quote"]],
                project_id : int,
                _project_input_data : pd.DataFrame,
                _quote_engine: QuoteEngine
                ) -> None:
        """
        Initialize the class.

        Args:
            project_name (str): String referring to the project's name.
            project (str, optional): String that should be either "estimate" or "quote", defaults to "quote".
        """
        print(f"Initialization of the new project >>>")
        self._project_name = project_name
        
        if project_type is None:
            project_type = "quote"

        self._project_type = project_type
        self._project_id = project_id
        self._project_input_data = None
        self._quote_engine = QuoteEngine(client_id=self.client_id, project_id=self._project_id)
        
        # Optional: validate inputs
        self._validate()

        print(f"Project {project_name} successfully created!")

    def _validate(self):
        """Private method to validate instance attributes."""
        if not isinstance(self._project_name, str):
            raise ValueError("Project name must be str")
    
    def __generate_id(self) -> str:
        """
        Private method to generate a secure identifier of the project.
        Returns a unique hexadecimal public identifier (7 hex chars + 2 hex chars).
        """

        char_num = 7
        max_value = 16 ** (char_num + 1)  # 16^8
        max_evals = int(1e4)
        count = 0

        file_path = ".quotesIdentifiers.txt"

        # Read existing numbers
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                used = {line.strip() for line in f}
        except FileNotFoundError:
            used = set()

        # Generate until unique
        while count < max_evals:
            secret = np.random.randint(0, max_value)
            if str(secret) not in used:
                break
            count += 1

        if count >= max_evals:
            raise Exception("Project initialization timed out during project identifier generation...")

        # Compute public hex ID (7 + 2 hex chars)
        block = max_value // 16
        part1 = secret % block
        part2 = secret // block

        public_number = f"{part1:0{char_num}x}{part2:02x}"

        # Store secret number for future uniqueness checks
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(str(secret) + "\n")

        return public_number

    def __checkId(self, public: str) -> dict:
        if not self._PUBLIC_ID_REGEX.match(public):
            return False

        file_path = ".quotesIdentifiers.txt"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                used = {line.strip() for line in f}
        except FileNotFoundError:
            return False

        char_num = 7
        max_value = 16 ** (char_num + 1)
        block = max_value // 16

        part1 = int(public[:char_num], 16)
        part2 = int(public[char_num:], 16)

        secret = part2 * block + part1

        res = {
            "status" : False,
            "secret" : -1
        }

        if str(secret) in used:
            res["status"] = True
            res["secret"] = secret

        return res

    @property
    def getName(self):
        """Getter for param1."""
        return self._project_name

    def setProjectName(self, value):
        """Setter for param1 with optional validation."""
        self._project_name = value
        self._validate()

    def create_model(self) -> None:
        if self._project_input_data == None:
            raise Exception("A dataset should be added to the project first")
        self._quote_engine.init_engine(self._project_input_data)
    
    def readFile(self, file_path : str) -> None:
        loader = DataLoader(filepath = str(file_path))
        self._project_input_data = loader.getPandas()


    def __repr__(self):
        """Unambiguous string representation."""
        return f"QuoteProject <{self._project_id}>"
