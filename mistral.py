import os
import httpx
import numpy as np


def make_embeddings(text: str) -> np.ndarray:
    response = httpx.post(
        "https://api.mistral.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"
        },
        json={
            "model": "mistral-embed",
            "input": text
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return np.array(data["data"][0]["embedding"], dtype=np.float32)