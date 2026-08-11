from functools import lru_cache
import fasttext
import numpy as np
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve

@lru_cache(maxsize=1)
def get_model():
    """
    Charge le modèle FastText une seule fois.
    """
    print("Chargement du modèle FastText...", flush=True)
    
    file = Path("cc.fr.300.bin")
    archive = Path("cc.fr.300.bin.gz")
    if not file.exists():
        if not archive.exists():
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz"
            destination = "cc.fr.300.bin.gz"
            print("Téléchargement...")
            urlretrieve(url, destination)
            print("Téléchargement terminé.")

        print("Décompression du modèle FastText...", flush=True)
        with gzip.open(archive, "rb") as fin:
            with open("cc.fr.300.bin", "wb") as fout:
                shutil.copyfileobj(fin, fout)

    return fasttext.load_model("cc.fr.300.bin")


def make_embeddings(text: str) -> np.ndarray:
    """
    Vectorise un texte avec FastText.

    Retourne
    --------
    np.ndarray
        Embedding de dimension 300.
    """
    model = get_model()

    # Fast Text ne fonctionne pas avec des textes mais des mots
    # Ici on supprime les espaces et on ne garde que les mots
    text = " ".join(text.split())

    return model.get_sentence_vector(text)