from pathlib import Path
from typing import Callable
from converter import doc2text
from utils import split_markdown_by_headers
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import pickle

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
  """Calcule la similarité cosinus entre deux vecteurs."""
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  if norm_vec1 == 0 or norm_vec2 == 0:
    return 0 # Éviter la division par zéro
  return dot_product / (norm_vec1 * norm_vec2)


#
# vectorise les documents convertis (Mistral embeddings)
#

def make_embeddings_Mistral(text: str) -> np.ndarray:
    pass

#
# vectorise les documents convertis (Sentence-BERT)
#


@lru_cache(maxsize=1)
def get_model_SentenceBERT():
    print("Chargement du modèle...", flush=True)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def make_embeddings_SentenceBERT(text: str) -> np.ndarray:
    """
    Vectorise le texte en utilisant le modèle Sentence-BERT.

    Retourne l'embedding de type <numpy.ndarray>
    """
    model = get_model_SentenceBERT() # en cache pour éviter de recharger le modèle à chaque appel

    # Générer les embeddings
    return model.encode(text, convert_to_numpy=True)

def make_embeddings(dossier: Path, embedding_function: Callable[[str], np.ndarray]):
    """
    Vectorise les textes en utilisant le modèle Sentence-BERT.

    embedding_function: fonction d'encodage qui prend un texte en entrée et retourne un embedding de type <numpy.ndarray>

    Retourne un dictionnaire :
    {
        "embedding": <numpy.ndarray>,
        "metadata": {
            "file": "Nom du fichier",
            "title": "Titre de la section"
        },
        "text": "Contenu du texte"
    }
    """

    embedding_docs = []

    # Embeddings des documents potentiels
    for element in dossier.rglob("*"):
        if not element.is_file():
            continue

        if not element.suffix.lower() == ".md":
            continue
        
        sections = split_markdown_by_headers(element)
        
        print(f"Vectorisation du fichier : {element.name} ({len(sections)} sections)", flush=True)
        for section in sections:
            embedding_docs.append({
                "embedding": embedding_function(section.get("content", "")),
                "text": section.get("content", ""),
                "metadata": {
                    "path": element.resolve(),
                    "file": section.get("filename", ""),
                    "title": section.get("title", ""),
                    #...
                }
            })

    return embedding_docs

def save_embeddings(embedding_docs: list, embedding_name: str):
    with open(embedding_name, "wb") as f:
        pickle.dump(embedding_docs, f)

def load_embeddings(embedding_name: str)-> list:
    with open(embedding_name, "rb") as f:
        return pickle.load(f)

#
# vectorise les documents convertis (FastText)
#

def make_embeddings_FastText(text: str) -> np.ndarray:
    pass

#
# MAIN
#

path = Path("./inputs")

# convertie les documents en textes Markdown
doc2text(path)

# convertie les textes Markdown en emmbeddings
if not Path("SentenceBERT.embeddings").exists():
    embedding_docs = make_embeddings(path, make_embeddings_SentenceBERT)
    save_embeddings(embedding_docs, "SentenceBERT.embeddings")
