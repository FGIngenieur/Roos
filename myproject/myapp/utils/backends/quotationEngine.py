import os
cwd = os.getcwd()
os.chdir('../libs/')
from quotesLibs import *
os.chdir(cwd)
from quoteClustering import *

class QuoteEngine:
    """
        The class describes engine object corresponding to the backend object
        for quotes and estimates automation.
    """
    def __init__(
        self,
        client_id,
        project_id,
        __,
        __engine
    ):
    self._client_id = client_id
    self._project_id = project_id
    self.__engine = SemanticClusterizer()

    @staticmethod
    def _fit_couple(
        str1,
        str2
    ):
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        s1 = "Le véhicule est en panne."
        s2 = "La voiture ne fonctionne plus."

        emb1 = model.encode(s1, convert_to_tensor=True)
        emb2 = model.encode(s2, convert_to_tensor=True)

        sim = util.cos_sim(emb1, emb2).item()
        return sim
    
    def _fit_set(
        self,
        find_text,
        in_list
    ):
        results_list = []

        for e in in_list:
            results_list.append(self._fit_couple(find_text, e))
        
        i = 0
