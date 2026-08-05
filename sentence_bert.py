
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np

#
# vectorise les documents convertis (Sentence-BERT)
#


@lru_cache(maxsize=1)
def get_model():
    print("Chargement du modèle...", flush=True)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def make_embeddings(text: str) -> np.ndarray:
    """
    Vectorise le texte en utilisant le modèle Sentence-BERT.

    Retourne l'embedding de type <numpy.ndarray>
    """
    model = get_model() # en cache pour éviter de recharger le modèle à chaque appel

    # Générer les embeddings
    return model.encode(text, convert_to_numpy=True)
