from pathlib import Path
from typing import Callable
from converter import doc2text
from sentence_bert import make_embeddings as make_embeddings_SentenceBERT
from mistral import make_embeddings as make_embeddings_Mistral
from fast_text import make_embeddings as make_embeddings_FastText
from utils import split_markdown_by_headers
import numpy as np
import pickle

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
  """Calcule la similarité cosinus entre deux vecteurs."""
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  if norm_vec1 == 0 or norm_vec2 == 0:
    return 0 # Éviter la division par zéro
  return dot_product / (norm_vec1 * norm_vec2)


def make_embeddings(dossier: Path, embedding_function: Callable[[str], np.ndarray]):
    """
    Vectorise les textes en utilisant le modèle d'embedding fourni.

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
    """
    Sauvegarde l'embedding des documents dans un fichier pickle.
    """
    with open(embedding_name, "wb") as f:
        pickle.dump(embedding_docs, f)

def load_embeddings(embedding_name: str)-> list:
    """
    Charge l'embedding des documents depuis un fichier pickle.
    """
    with open(embedding_name, "rb") as f:
        return pickle.load(f)

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

if not Path("Mistral.embeddings").exists():
    embedding_docs = make_embeddings(path, make_embeddings_Mistral)
    save_embeddings(embedding_docs, "Mistral.embeddings")

if not Path("FastText.embeddings").exists():
    embedding_docs = make_embeddings(path, make_embeddings_FastText)
    save_embeddings(embedding_docs, "FastText.embeddings")

# Mesurer la similarité
# en utilisant la similarité cosinus sur 2 vecteurs A et B nous pouvons déterminer à quel point ils sont similaires.
# La valeur de similarité cosinus varie entre -1 et 1
# 1 signifie que les vecteurs sont identiques
# 0 signifie qu'ils sont orthogonaux (aucune similarité)
# -1 signifie qu'ils sont opposés.
#    
import numpy as np

embedding_docs = load_embeddings("SentenceBERT.embeddings")

vectors = np.array([
    item["embedding"]
    for item in embedding_docs
])

print(vectors.shape)


